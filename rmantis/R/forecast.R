#' Forecast with Mantis
#'
#' Runs the Mantis deep-learning forecast model on a numeric time series,
#' optionally using a covariate series. Python and the \code{mantis} module
#' are validated before any calls are made via \code{rmantis_check_python()}.
#'
#' @param time_series Numeric vector of observed counts (required).
#' @param model_dir Character. Directory containing the pretrained \code{.pt}
#'   weight files (required). Use \code{\link{mantis_download_weights}} to
#'   populate this directory.
#' @param covariate Optional numeric vector of covariate values aligned with
#'   \code{time_series}. Pass \code{NULL} (default) when no covariate is used.
#' @param target_type Integer. Type of target series:
#'   \code{0} = cases, \code{1} = hospitalisations, \code{2} = deaths (default).
#' @param covariate_type Integer or \code{NULL}. Type code for the covariate
#'   series; ignored when \code{covariate} is \code{NULL}.
#' @param horizon Integer forecast horizon; must be \code{4} or \code{8}.
#'   Default is \code{4}.
#' @param use_covariate Logical. Whether to load the covariate-enabled model
#'   weights. Default is \code{TRUE}.
#' @param device Character. The device to be used. Default is
#'   \code{'cuda'}.
#' @param debug Logical. When \code{TRUE}, prints diagnostic messages about
#'   the Python executable and model directory being used. Default is
#'   \code{FALSE}.
#'
#' @return A numeric matrix of forecast samples returned by the Mantis model.
#'
#' @seealso \code{\link{mantis_download_weights}} to fetch the required weights.
#'
#' @export
mantis_forecast <- function(time_series,
                            model_dir,
                            covariate      = NULL,
                            target_type    = 2L,
                            covariate_type = NULL,
                            horizon        = 4L,
                            use_covariate  = TRUE,
			    device         = "cuda",
                            debug          = FALSE) {

  # --- Input validation -------------------------------------------------------
  #     WRC: There should be an input validation for the device	
  if (!is.numeric(time_series) || length(time_series) == 0) {
    stop("'time_series' must be a non-empty numeric vector.", call. = FALSE)
  }
  if (!horizon %in% c(4L, 8L)) {
    stop("'horizon' must be either 4 or 8.", call. = FALSE)
  }
  if (!dir.exists(model_dir)) {
    stop(sprintf("Model directory not found: %s", model_dir), call. = FALSE)
  }

  # --- Python / module check --------------------------------------------------
  cfg <- rmantis_check_python()
  if (debug) cat(sprintf("Python executable: %s\n", cfg$python))
  if (debug) cat(sprintf("model_dir        : %s\n", model_dir))
  if (debug) cat(sprintf("device           : %s\n", device))

  # --- Run forecast -----------------------------------------------------------
  mantis <- reticulate::import("mantis")

  if (debug) cat("Creating Mantis model object ...\n")
  model <- mantis$Mantis(
    forecast_horizon = as.integer(horizon),
    use_covariate    = use_covariate,
    model_dir        = model_dir,
    device           = device
  )

  if (debug) cat("Calling mantis$predict() ...\n")
  preds <- model$predict(
    time_series    = as.integer(time_series),
    covariate      = if (!is.null(covariate)) as.integer(covariate) else NULL,
    target_type    = as.integer(target_type),
    covariate_type = covariate_type
  )

  as.matrix(preds)
}
