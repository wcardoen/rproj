#' Sets the Python executable
#'
#' @param exe Character. Path to the Python executable to use.
#' @return Invisibly returns the reticulate Python config list on success.
#' @keywords internal
mantis_setup_python <- function(exe) {
  if (!is.character(exe) || nchar(trimws(exe)) == 0) {
    stop("'exe' must be a non-empty character string.", call. = FALSE)
  }
  if (!file.exists(exe)) {
    stop(
      "Python executable not found at: ", exe, "\n",
      "Please provide a valid path to a Python binary.",
      call. = FALSE
    )
  }
  reticulate::use_python(exe, required = TRUE)
  invisible(reticulate::py_config())
}



#' Check that Python and the mantismod module are available
#'
#' Validates that a Python executable exists in the current reticulate
#' configuration and that the \code{mantismod} Python module is installed.
#' Called internally by \code{\link{mantis_forecast}} before any Python
#' calls are made, so users receive a clear, actionable error message
#' rather than a raw Python traceback.
#'
#' @return Invisibly returns the reticulate Python config list on success.
#' @keywords internal
mantis_check_python <- function() {
  cfg <- reticulate::py_config()

  if (!file.exists(cfg$python)) {
    stop(
      "Python executable not found at: ", cfg$python, "\n",
      "Configure your environment with reticulate::use_python() or ",
      "reticulate::use_virtualenv() before calling mantis_forecast().",
      call. = FALSE
    )
  }

  if (!reticulate::py_module_available("mantismod")) {
    stop(
      "The Python 'mantismod' module is not installed in the current environment.\n",
      "Install it with: reticulate::py_install('mantismod')",
      call. = FALSE
    )
  }

  invisible(cfg)
}
