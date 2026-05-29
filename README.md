# mantisproj

* `mantismod`: modified `mantis` Python code (CPU + GPU)
  - The **installation** instructions can be found in: `mantismod/README.md`
* `rmantismod`: R package based on the modified `mantismod` Python code.
  - Prerequisites: 
    + The installation of `mantismod` (Python module)
    + The following `R` libraries: `devtools`, `reticulate`
  - Installation:
    ```R
    R>library(devtools)
    R>install_github("wcardoen/mantisproj", subdir = "rmantismod")
    ```
  - Check:
    ```R
    R>library(rmantismod)
    ```

  - Help:
    ```R
    R>library(rmantismod)
    R>help(rmantismod)
    R>help(mantis_cudacheck)
    R>help(mantis_download_weights)
    R>help(mantis_forecast)
    ```
