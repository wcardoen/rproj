# Modified version of mantis i.e. mantismod

## Original code

The original code was downloaded from (Carson Dudley's Github)[https://github.com/carsondudley1/Mantis]

## Modifications
The original code only supported CPUs. It now supports GPUs through Torch.
Several versions of Torch are supported:
* `cpu`
* `cu121` 
* `cu124` 
* `cu126` 
* `cu128` 
* `cu130` 
* `cu132`

### Installation of mantis in your new python setup  

Let `DIR` be installation directory

- `module load uv`
- `cd $DIR`
- `uv init mypy`
- `cd mypy`
- `uv python install 3.13 -v`   
- `uv python pin 3.13 -v`      # If you have experience an error, make sure requires-python >= pin_version
- `uv add "mantis[cu128] @ git+https://github.com/wcardoen/rproj.git#subdirectory=mantis"`

```bash
uv run python3
Python 3.13.5 (main, Jul  1 2025, 18:37:36) [Clang 20.1.4 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mantis
>>> mantis.__version__
'0.1.0'
```

Let `DIR` be directory where `mypy` is installed.
Then 
```bash
export PYTHONEXE=$DIR/.venv/bin/python3
which $PYTHONEXE
```

### Check the code

In order to do the test, you need to add `pandads` to the installation
- `cd $DIR`
- `uv add pandas`
