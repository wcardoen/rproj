from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

# Hardcoded normalization stats (match training values exactly)
DEFAULT_MEAN: dict[int, float] = {
    0: 5.55,  # cases
    1: 3.84,  # hospitalizations
    2: 2.57,  # deaths
}
DEFAULT_STD: dict[int, float] = {
    0: 3.63,
    1: 3.15,
    2: 2.59,
}


def preprocess_input(
    time_series: npt.ArrayLike,
    covariate: Optional[npt.ArrayLike] = None,
    population: Optional[float] = None,
    target_type: int = 2,
    covariate_type: Optional[int] = None,
    mean_std: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    """
    Convert raw weekly time series (and an optional covariate) into
    model-ready tensors.

    All tensors are created on CPU. The caller is responsible for moving
    them to the appropriate device before passing them to the model.

    Args:
        time_series: 1-D array-like of raw weekly target values. Will be
            cast to ``float32``.
        covariate: 1-D array-like of raw weekly covariate values, or
            ``None`` when no covariate is used. Must have the same length
            as *time_series* when provided. Will be cast to ``float32``.
        population: Raw (un-transformed) population count for the region
            being forecast, or ``None``. The value is ``log1p``-transformed
            internally and optionally z-scored when *mean_std* contains
            population statistics.
        target_type: Integer code that identifies the target signal:
            ``0`` = cases, ``1`` = hospitalizations, ``2`` = deaths.
            Selects the default normalisation statistics when *mean_std*
            is not supplied. Defaults to ``2``.
        covariate_type: Integer code for the covariate signal using the
            same coding as *target_type*. When ``None`` the covariate is
            treated as having the same type as *target_type*.
        mean_std: Optional dictionary that overrides the default
            normalisation statistics. Recognised keys:

            * ``"mean"`` / ``"std"`` — target z-score parameters.
            * ``"cov_mean"`` / ``"cov_std"`` — covariate z-score
              parameters.
            * ``"pop_mean"`` / ``"pop_std"`` — population log-value
              z-score parameters.

            Any missing keys fall back to the hard-coded defaults.

    Returns:
        A dictionary with the following keys, ready to be unpacked into
        ``MultiTimeSeriesForecaster.predict(**inputs)``:

        * ``"values"`` — ``FloatTensor`` of shape ``[1, T, C]`` where
          ``C`` is 1 (no covariate) or 2 (with covariate).
        * ``"disease_type"`` — ``LongTensor`` of shape ``[1]``, currently
          always ``0`` (placeholder).
        * ``"target_type"`` — ``LongTensor`` of shape ``[1]``.
        * ``"population"`` — ``FloatTensor`` of shape ``[1]``.
        * ``"day_indices"`` — ``LongTensor`` of shape ``[1, T]`` with
          absolute day indices starting at 1098.
        * ``"valid_mask"`` — ``BoolTensor`` of shape ``[1, T]``, all
          ``True`` (no padding).
    """
    # Convert inputs to float32 numpy arrays
    x: npt.NDArray[np.float32] = np.asarray(time_series, dtype=np.float32)
    cov: Optional[npt.NDArray[np.float32]] = (
        np.asarray(covariate, dtype=np.float32) if covariate is not None else None
    )

    # ------------------------------------------------------------------ #
    # Normalise target                                                     #
    # ------------------------------------------------------------------ #
    if mean_std is not None and "mean" in mean_std and "std" in mean_std:
        target_mean = float(mean_std["mean"])
        target_std = float(mean_std["std"])
    else:
        target_mean = DEFAULT_MEAN[target_type]
        target_std = DEFAULT_STD[target_type]

    x = (np.log1p(x) - target_mean) / (target_std + 1e-7)

    # ------------------------------------------------------------------ #
    # Normalise covariate (if provided) and assemble feature matrix       #
    # ------------------------------------------------------------------ #
    if cov is not None:
        cov_type: int = covariate_type if covariate_type is not None else target_type
        if mean_std is not None and "cov_mean" in mean_std and "cov_std" in mean_std:
            cov_mean = float(mean_std["cov_mean"])
            cov_std = float(mean_std["cov_std"])
        else:
            cov_mean = DEFAULT_MEAN[cov_type]
            cov_std = DEFAULT_STD[cov_type]

        cov = (np.log1p(cov) - cov_mean) / (cov_std + 1e-7)
        feats: npt.NDArray[np.float32] = np.stack([x, cov], axis=1)  # [T, 2]
    else:
        feats = x[:, None]  # [T, 1] — true single-channel path

    # ------------------------------------------------------------------ #
    # Build tensors                                                        #
    # ------------------------------------------------------------------ #
    values: torch.Tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # [1, T, C]
    T: int = values.shape[1]
    valid_mask: torch.Tensor = torch.ones(T, dtype=torch.bool).unsqueeze(0)        # [1, T]
    day_indices: torch.Tensor = torch.arange(T, dtype=torch.long).unsqueeze(0) + 1098  # [1, T]

    # ------------------------------------------------------------------ #
    # Normalise population                                                 #
    # ------------------------------------------------------------------ #
    pop_log: float = float(np.log1p(float(population))) if population is not None else 0.0
    if mean_std is not None and "pop_mean" in mean_std and "pop_std" in mean_std:
        population_scaled: float = (pop_log - float(mean_std["pop_mean"])) / (
            float(mean_std["pop_std"]) + 1e-7
        )
    else:
        population_scaled = pop_log  # fallback: no standardisation applied

    population_tensor: torch.Tensor = torch.tensor([population_scaled], dtype=torch.float32)

    return {
        "values": values,
        "disease_type": torch.tensor([0], dtype=torch.long),  # placeholder
        "target_type": torch.tensor([target_type], dtype=torch.long),
        "population": population_tensor,
        "day_indices": day_indices,
        "valid_mask": valid_mask,
    }
