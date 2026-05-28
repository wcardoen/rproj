library(rmantismod)

# HERE COMES YOU PYTHONEXE 
PYTHONEXE <- "/home/sleipnir/Devel/python/mypy/.venv/bin/python3"


ts <- Sys.time()
cat(sprintf("Simulation started at: '%s'\n",ts))
hasCuda <- mantis_cudacheck(PYTHONEXE)
device <- "cuda"
if(!hasCuda) device <- "cpu"


# PART 1:
# ------
# Input: Directory where the models where stored
model_dir = "../models"
input_file = "../data/example.csv"

cat(sprintf("Input data:\n"))
cat(sprintf("  Example file:%s\n", input_file))
cat(sprintf("  Model dir.  :%s\n", model_dir))
cat(sprintf("  Device      :%s\n\n", device))

mydf = read.csv(input_file, header=T)
head(mydf)


# PART 2:
# ------
debug <- F
mat <- mantis_forecast(time_series = mydf$hospitalizationCount,
                       model_dir = model_dir,   # NEW <-- REQUIRED
		       python_exe = PYTHONEXE,  # NEW <-- REQUIRED
                       target_type = 1L,        # 0 = cases, 1 = hosp, 2 = deaths
                       horizon = 4L,
                       use_covariate = FALSE,
		       device = device,
                       debug=debug) 
cat(sprintf("\nResults::\n"))
print(mat)

te <- Sys.time()
time_spent <- difftime(te, ts, units = "secs")
cat(sprintf("\n\nSimulation ended at: '%s'\n",te))
cat(sprintf("Walltime (sec): %12.4f\n",time_spent <- difftime(te, ts, units = "secs")))

