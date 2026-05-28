#' Check CUDA Availability via PyTorch
#'
#' Validates whether CUDA is available through the configured Python environment's
#' \code{torch} library. If available, it lists the number of detected GPU devices 
#' along with their names.
#'
#' @param python_exe Character. Path to the Python executable (required).
#'
#' @return A logical value: \code{TRUE} if CUDA is available and PyTorch can 
#'   access a GPU, \code{FALSE} otherwise.
#'
#' @seealso \code{\link{mantis_forecast}} to run forecasts using the GPU.
#'
#' @export
mantis_cudacheck <- function(python_exe) {

    # Setup PYTHON
    mantis_setup_python(python_exe)	

    # Import the Python torch library
    torch <- reticulate::import("torch")

    cat(sprintf("\nChecking CUDA\n"))
    hasCuda <- torch$cuda$is_available()
    if (hasCuda) {
        cat(sprintf("  CUDA available!\n"))
        num_devices <- torch$cuda$device_count()
        cat(sprintf("  #GPU devices found:%d\n", num_devices))
        for (idevice in seq(from = 0, to = (num_devices - 1))) {
            cat(sprintf("  Device:%d -> %s\n", idevice, 
            torch$cuda$get_device_name(as.integer(idevice)))) 
        }
    } else {
        cat(sprintf("NO CUDA available!\n"))
    }
    cat(sprintf("End CUDA check!\n\n"))
    return(hasCuda)
}
