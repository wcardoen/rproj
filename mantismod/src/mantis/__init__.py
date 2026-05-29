"""
Mantis: probabilistic disease-outbreak forecasting.

Public API
----------
Mantis
    High-level inference wrapper. Load a checkpoint and call
    :meth:`~inference.Mantis.predict` to obtain multi-quantile forecasts.

Example::

    from mantis import Mantis

    model = Mantis(forecast_horizon=4, use_covariate=False, device="cuda")
    predictions = model.predict(weekly_deaths, target_type=2)
    # predictions: np.ndarray of shape [4, 9]
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mantis")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .inference import Mantis

__all__ = ["Mantis"]
