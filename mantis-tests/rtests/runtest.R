library(rmantis)

cudaCheck <- function(){

    library(reticulate)
    # Import the Python torch library
    torch <- import("torch")

    cat(sprintf("\nChecking CUDA\n"))
    hasCuda <- torch$cuda$is_available()
    if (hasCuda){
        cat(sprintf("  CUDA available!\n"))
        num_devices = torch$cuda$device_count()
        cat(sprintf("  #GPU devices found:%d\n", num_devices))
        for(idevice in seq(from=0,to=(num_devices-1))){
            cat(sprintf("    Device:%d -> %s\n", idevice, 
            torch$cuda$get_device_name(as.integer(idevice)))) 
        }
    }else{
        cat(sprintf("NO CUDA available!\n"))
    }
    cat(sprintf("End CUDA check!\n\n"))
    return(hasCuda)
}


ts <- Sys.time()
cat(sprintf("Simulation started at: '%s'\n",ts))
hasCuda <- cudaCheck()

# PART 1:
# ------
# Input: Directory where the models where stored
model_dir = "../models"

input_file = "../data/example.csv"

cat(sprintf("Input data:\n"))
cat(sprintf("  Example file:%s\n", input_file))
cat(sprintf("  Model dir.  :%s\n", model_dir))

device <- "cuda"
if(!hasCuda) device <- "cpu"
cat(sprintf("  Device      :%s\n\n", device))

mydf = read.csv(input_file, header=T)
head(mydf)


# PART 2:
# ------
debug <- F
mat <- rmantis::mantis_forecast(time_series = mydf$hospitalizationCount,
                       model_dir = model_dir,   # NEW <-- REQUIRED
                       target_type = 1L,        # 0 = cases, 1 = hosp, 2 = deaths
                       horizon = 4L,
                       use_covariate = FALSE,
		       device = device,
                       debug=debug) 
cat(sprintf("Results::\n"))
print(mat)

te <- Sys.time()
time_spent <- difftime(te, ts, units = "secs")
cat(sprintf("\n\nSimulation ended at: '%s'\n",te))
cat(sprintf("Walltime (sec): %12.4f\n",time_spent <- difftime(te, ts, units = "secs")))

