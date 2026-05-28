#' rmantismod: modifieds mantis (GPU)
#'
#' This is a modified version of rmantis.
#' It has GPU support for 1 CUDA device.
#'
#' @section Main functions:
#' The core functions of `mantismod` are:
#' \itemize{
#'   \item \code{\link{mantis_cudacheck}} - Checks CUDA availability and lists GPU devices
#'   \item \code{\link{mantis_download_weights}} - Downloads the weights 
#'   \item \code{\link{mantis_forecast}} - Performs a forecast 
#' }
#'
#' @section Getting started:
#' A typical workflow looks like:
#' ```r
#' library(rmantismod)
#' # Verify your GPU environment first
#' mantis_cudacheck("/path/to/python")
#' res <- mantis_forecast
#' ```
#'
#' @author
#' Wim R.M. Cardoen \email{wcardoen@gmail.com}
#'
#' @references
#' Any papers, books, or URLs your package is based on.
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL
