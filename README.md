# mantisproj

* `mantismod`: modified `mantis` `Python` package (CPU + GPU)
  - the **installation** instructions can be found in: `mantismod/README.md`
* `rmantismod`: R package based on the modified `mantismod` Python code.
  - prerequisites: 
    + the installation of `mantismod` (Python module)
    + the following `R` libraries: `devtools`, `reticulate`
  - installation:
    ```R
    R>library(devtools)
    R>install_github("wcardoen/mantisproj", subdir = "rmantismod")
    ```
  - check:
    ```R
    R>library(rmantismod)
    ```

  - help:
    ```R
    R>library(rmantismod)
    R>help(rmantismod)
    R>help(mantis_cudacheck)
    R>help(mantis_download_weights)
    R>help(mantis_forecast)
    ```
