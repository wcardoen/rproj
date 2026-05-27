#' Check that Python and the mantis module are available
#'
#' Validates that a Python executable exists in the current reticulate
#' configuration and that the \code{mantis} Python module is installed.
#' Called internally by \code{\link{mantis_forecast}} before any Python
#' calls are made, so users receive a clear, actionable error message
#' rather than a raw Python traceback.
#'
#' @return Invisibly returns the reticulate Python config list on success.
#' @keywords internal
rmantis_check_python <- function() {
  cfg <- reticulate::py_config()

  if (!file.exists(cfg$python)) {
    stop(
      "Python executable not found at: ", cfg$python, "\n",
      "Configure your environment with reticulate::use_python() or ",
      "reticulate::use_virtualenv() before calling mantis_forecast().",
      call. = FALSE
    )
  }

  if (!reticulate::py_module_available("mantis")) {
    stop(
      "The Python 'mantis' module is not installed in the current environment.\n",
      "Install it with: reticulate::py_install('mantis')",
      call. = FALSE
    )
  }

  invisible(cfg)
}
