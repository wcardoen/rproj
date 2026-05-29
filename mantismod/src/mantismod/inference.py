from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from .model import MultiTimeSeriesForecaster
from .utils import preprocess_input, DEFAULT_MEAN, DEFAULT_STD


# --------------------------------------------------------------------------- #
# Private checkpoint-handling helpers                                          #
# --------------------------------------------------------------------------- #

def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remove the ``"module."`` key prefix introduced by ``nn.DataParallel``.

    When a model is saved after being wrapped with ``torch.nn.DataParallel``
    every parameter name is prefixed with ``"module."``. This helper
    transparently strips that prefix so the weights can be loaded into an
    unwrapped model.

    Args:
        state_dict: Raw state dict loaded from a ``.pt`` checkpoint file.

    Returns:
        The same state dict with ``"module."`` prefixes removed from all
        keys, or the original dict unchanged if no such prefix was found.
    """
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _remap_ts_type_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Rename legacy ``ts_type_*`` keys to the current ``target_type_*`` names.

    Early training runs used ``ts_type_embedding`` / ``ts_type_norm`` for
    what is now called ``target_type_embedding`` / ``target_type_norm``.
    This helper performs the in-place rename so older checkpoints load
    without errors. Keys that already use the current naming are left
    unchanged.

    Args:
        state_dict: State dict, possibly containing legacy key names.

    Returns:
        The state dict with any legacy keys renamed to their current
        equivalents.
    """
    mapping: dict[str, str] = {
        "ts_type_embedding.weight": "target_type_embedding.weight",
        "ts_type_norm.weight":      "target_type_norm.weight",
        "ts_type_norm.bias":        "target_type_norm.bias",
    }
    for old, new in mapping.items():
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
    return state_dict


def _detect_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Infer the number of input channels the checkpoint was trained with.

    Inspects the weight shape of the first convolutional layer in
    ``MultiScaleConvEmbedding`` (``values_embedding.conv_short``, etc.) to
    determine whether the checkpoint expects a single input channel
    (no-covariate model) or two channels (covariate model).

    Args:
        state_dict: State dict loaded from a ``.pt`` checkpoint file.

    Returns:
        ``1`` for a no-covariate checkpoint, ``2`` for a covariate
        checkpoint, or ``1`` as a safe fallback if no matching key is
        found.
    """
    probe_keys: list[str] = [
        "values_embedding.conv_short.weight",
        "values_embedding.conv_med.weight",
        "values_embedding.conv_long.weight",
        "values_embedding.conv_vlong.weight",
    ]
    for k in probe_keys:
        if k in state_dict:
            return int(state_dict[k].shape[1])
    return 1  # safe fallback


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """
    Resolve a flexible device argument into a ``torch.device`` instance.

    Args:
        device: The desired compute device. Accepted forms:

            * ``None`` — automatically selects CUDA when available,
              otherwise falls back to CPU.
            * ``str`` — e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``,
              ``"mps"``.
            * ``torch.device`` — used directly without modification.

    Returns:
        A ``torch.device`` corresponding to the requested device.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

class Mantis:
    """
    High-level inference wrapper for the Mantis disease-forecasting model.

    Handles checkpoint loading, device placement, input preprocessing,
    model inference, and output denormalisation in a single, convenient
    interface. Runs on CPU or a single GPU.

    Example::

        model = Mantis(forecast_horizon=4, use_covariate=False, device="cuda")
        predictions = model.predict(weekly_deaths, target_type=2)
        # predictions: np.ndarray of shape [4, 9]
    """

    def __init__(
        self,
        forecast_horizon: int = 4,
        use_covariate: bool = True,
        model_dir: str = "models",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Load a Mantis checkpoint and prepare it for inference.

        Args:
            forecast_horizon: Number of weeks to forecast. Must be ``4``
                or ``8``.
            use_covariate: Set to ``True`` when you intend to supply a
                covariate series alongside the target. The chosen value
                must match the checkpoint variant (``*_cov.pt`` vs
                ``*_nocov.pt``); a ``ValueError`` is raised on mismatch.
            model_dir: Path to the directory that contains the ``.pt``
                checkpoint files. The file is expected to follow the
                naming convention
                ``mantis_{forecast_horizon}w_{cov|nocov}.pt``.
            device: Target compute device. Accepts ``None`` (auto-select
                CUDA when available, otherwise CPU), a device string such
                as ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or a
                ``torch.device`` object. Always uses at most one GPU.

        Raises:
            AssertionError: If *forecast_horizon* is not ``4`` or ``8``.
            FileNotFoundError: If the expected checkpoint file does not
                exist under *model_dir*.
            ValueError: If the requested *use_covariate* setting is
                inconsistent with the number of input channels found in
                the checkpoint.
        """
        assert forecast_horizon in [4, 8], "forecast_horizon must be 4 or 8"

        self.device: torch.device = _resolve_device(device)
        self.use_covariate: bool = use_covariate
        self.forecast_horizon: int = forecast_horizon

        # ------------------------------------------------------------------ #
        # Locate checkpoint                                                    #
        # ------------------------------------------------------------------ #
        suffix = "cov" if use_covariate else "nocov"
        filename = f"mantis_{forecast_horizon}w_{suffix}.pt"
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # ------------------------------------------------------------------ #
        # Load checkpoint directly onto the target device                     #
        # ------------------------------------------------------------------ #
        state_dict: dict[str, torch.Tensor] = torch.load(
            model_path, map_location=self.device
        )
        state_dict = _strip_module_prefix(state_dict)
        state_dict = _remap_ts_type_keys(state_dict)

        ckpt_in_ch: int = _detect_input_channels(state_dict)

        if self.use_covariate and ckpt_in_ch == 1:
            raise ValueError(
                "You requested use_covariate=True but the checkpoint expects 1 input "
                "channel (nocov). Load a *_cov.pt checkpoint or set use_covariate=False."
            )
        if (not self.use_covariate) and ckpt_in_ch == 2:
            raise ValueError(
                "You requested use_covariate=False but the checkpoint expects 2 input "
                "channels (cov). Load a *_nocov.pt checkpoint or set use_covariate=True "
                "and provide a covariate."
            )

        # ------------------------------------------------------------------ #
        # Build model, load weights, move to device                           #
        # ------------------------------------------------------------------ #
        self.model: MultiTimeSeriesForecaster = (
            MultiTimeSeriesForecaster(
                input_window=112,
                forecast_horizon=forecast_horizon,
                hidden_dim=1024,
                ffn_dim=2048,
                n_layers=16,
                n_heads=16,
                n_quantiles=9,
                disease_embed_dim=64,
                pop_embed_dim=64,
                binary_feat_dim=32,
                dropout=0.2,
                values_input_dim=ckpt_in_ch,
            )
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)

        self.model.eval()

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _move_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Move all tensor values in *inputs* to ``self.device``.

        Non-tensor values (e.g. plain Python scalars) are left unchanged.

        Args:
            inputs: Dictionary returned by :func:`~utils.preprocess_input`,
                mapping string keys to ``torch.Tensor`` objects.

        Returns:
            A new dictionary with the same keys where every
            ``torch.Tensor`` value has been moved to ``self.device``.
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

    # ---------------------------------------------------------------------- #
    # Public inference API                                                     #
    # ---------------------------------------------------------------------- #

    def predict(
        self,
        time_series: npt.ArrayLike,
        covariate: Optional[npt.ArrayLike] = None,
        population: Optional[float] = None,
        target_type: int = 2,
        covariate_type: Optional[int] = None,
    ) -> npt.NDArray[np.float32]:
        """
        Run a full inference pass on a single weekly time series.

        The method preprocesses raw inputs, moves them to the model's
        device, runs a forward pass under ``torch.no_grad()``, applies
        quantile widening, and returns denormalised predictions on CPU.

        Args:
            time_series: 1-D array-like of raw weekly target values. The
                series should cover the full input window expected by the
                model (112 weeks by default).
            covariate: 1-D array-like of raw weekly covariate values with
                the same length as *time_series*, or ``None``. Must be
                provided when the instance was created with
                ``use_covariate=True``; must be ``None`` (or will be
                ignored) otherwise.
            population: Raw (un-transformed) population count for the
                forecast region, or ``None``. Passed through
                ``log1p``-transformation internally.
            target_type: Integer code for the target signal:
                ``0`` = cases, ``1`` = hospitalizations, ``2`` = deaths.
                Determines the denormalisation statistics applied to the
                model output. Defaults to ``2``.
            covariate_type: Integer code for the covariate signal using
                the same coding as *target_type*. When ``None``, the
                covariate is assumed to share the type of *target_type*.

        Returns:
            A ``float32`` NumPy array of shape ``[H, 9]`` where ``H`` is
            the forecast horizon (4 or 8 weeks) and the 9 columns
            correspond to the predicted quantiles
            ``[0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95]``
            after quantile widening and denormalisation. Values are on the
            original (raw) scale of the target signal.

        Raises:
            ValueError: If the instance was created with
                ``use_covariate=True`` but no *covariate* is supplied.
        """
        if self.use_covariate and covariate is None:
            raise ValueError(
                "This model expects a covariate input, but none was provided."
            )

        # Preprocess on CPU; tensors are moved to device in _move_inputs
        inputs: dict[str, torch.Tensor] = preprocess_input(
            time_series=time_series,
            covariate=covariate if self.use_covariate else None,
            population=population,
            target_type=target_type,
            covariate_type=covariate_type if self.use_covariate else None,
            # Optionally pass pop stats here for exact training-style standardisation:
            # mean_std={"pop_mean": 14.1607, "pop_std": 1.9670}
        )

        inputs = self._move_inputs(inputs)

        with torch.no_grad():
            pred: torch.Tensor = self.model.predict(**inputs)  # [1, H, 9]

        # Bring results back to CPU before numpy conversion
        pred_np: npt.NDArray[np.float32] = pred.squeeze(0).cpu().numpy()  # [H, 9]

        # Widen all quantiles symmetrically away from the median
        median: npt.NDArray[np.float32] = pred_np[:, 4:5]
        widened: npt.NDArray[np.float32] = median + 1.15 * (pred_np - median)
        widened[:, 4] = median[:, 0]  # preserve exact median

        # Denormalise: reverse z-score then reverse log1p
        mean: float = DEFAULT_MEAN[target_type]
        std: float = DEFAULT_STD[target_type]
        denorm: npt.NDArray[np.float32] = np.expm1(widened * std + mean)

        return denorm
