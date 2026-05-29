# Modified version of mantis i.e. mantismod

## Original code

The original code was downloaded from [Carson Dudley's Github](https://github.com/carsondudley1/Mantis)

## Modifications

The original code only supported CPUs. The modified version now supports GPUs through Torch.<br>
Several versions of Torch are supported:
* `cpu`
* `cu121` 
* `cu124` 
* `cu126` 
* `cu128` 
* `cu130` 
* `cu132`

## Installation of `mantis` in your new python setup  

Let `DIR` be the base installation directory

- `module load uv`
- `cd $DIR`
- `uv init -p 3.12 mypy`
- `cd mypy`
- `uv python install 3.13 -v`   
- `uv python pin 3.13 -v`      # If you have experience an error, make sure requires-python >= pin_version
- installation of different versions of mantis (**Choose one!**)
  * `cu126` : [5.x, 6.x, 7.0, 7.5, 8.0, 8.6, 9.0] i.e. [Maxwell, ..., Hopper]
    + `cu126`: `uv add "mantismod[cu126] @ git+https://github.com/wcardoen/mantisproj.git#subdirectory=mantismod"`
  * `cu128` : [7.0, 7.5, 8.0, 8.6, 9.0, 10.0, 12.0] i.e. [Volta, ..., Blackwell]
    + `cu128`: `uv add "mantismod[cu128] @ git+https://github.com/wcardoen/mantisproj.git#subdirectory=mantismod"`
  * `cpu`:
    + `cpu`: `uv add "mantismod[cpu] @ git+https://github.com/wcardoen/mantisproj.git#subdirectory=mantismod"`
- `uv add pandas`  # `pandas` (for subsequent testing)

The `python` executable can be invoked into 2 different ways:

- Using `uv run python` in `$DIR/mypy` and subdirectories
```bash
cd $DIR/mypy
uv run python3
```

- By prepending the `PATH` variable with the directory of the binary
```bash
export PATH=$DIR/mypy/.venv/bin/:$PATH
which python3
```

### Check installation

We installed the version `cu126` (We ran the code on a `Maxwell` architecture)

```bash
[u0253283@kp298:mypy]$ uv run python3
Python 3.13.12 (main, Mar  3 2026, 15:01:51) [Clang 21.1.4 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'2.12.0+cu126'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce GTX TITAN X'
>>> import mantismod
>>> mantismod.__version__
'0.1.0'
```

