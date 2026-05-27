#' Download pretrained Mantis model weights
#'
#' Downloads the required \code{.pt} weight files from the official GitHub
#' Releases page and saves them into a local directory so
#' \code{\link{mantis_forecast}} can load them.
#'
#' @param horizon Integer. Forecast horizon; must be \code{4} or \code{8}.
#'   Default is \code{4}.
#' @param use_covariate Logical. Whether to download the covariate-enabled
#'   weights (\code{TRUE}, default) or the no-covariate weights (\code{FALSE}).
#' @param dest_dir Character. Directory in which to save the weight file.
#'   Created recursively if it does not exist. Default is \code{"models"}.
#' @param version Character. GitHub release tag to download from.
#'   Default is \code{"mantis-v1.0"}.
#'
#' @return Invisibly returns the local path to the downloaded weight file.
#'
#' @seealso \code{\link{mantis_forecast}} to run forecasts using the weights.
#'
#' @export
mantis_download_weights <- function(horizon       = 4L,
                                    use_covariate = TRUE,
                                    dest_dir      = "models",
                                    version       = "mantis-v1.0") {

  if (!horizon %in% c(4L, 8L)) {
    stop("horizon must be either 4 or 8.", call. = FALSE)
  }

  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE)
  }

  suffix    <- if (use_covariate) "cov" else "nocov"
  fname     <- sprintf("mantis_%dw_%s.pt", as.integer(horizon), suffix)
  base_url  <- sprintf(
    "https://github.com/carsondudley1/Mantis/releases/download/%s", version
  )
  url       <- paste(base_url, fname, sep = "/")
  dest_file <- file.path(dest_dir, fname)

  message("Downloading: ", url)
  utils::download.file(url, destfile = dest_file, mode = "wb")
  message("Saved model to: ", dest_file)

  invisible(dest_file)
}
