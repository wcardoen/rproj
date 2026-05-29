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
- `uv init -p 3.12 mypy`
- `cd mypy`
- `uv python install 3.13 -v`   
- `uv python pin 3.13 -v`      # If you have experience an error, make sure requires-python >= pin_version
- installation of different versions of mantis (Choose one)
  `cu126` : [5.x, 6.x, 7.0, 7.5, 8.0, 8.6, 9.0] i.e. [Maxwell, ..., Hopper]
  `cu128` : [7.0, 7.5, 8.0, 8.6, 9.0, 10.0, 12.0] i.e. [Volta, ..., Blackwell]
- thus:
  * `uv add "mantismod[cu126] @ git+https://github.com/wcardoen/mantisproj.git#subdirectory=mantismod"`
  * `uv add "mantismod[cu128] @ git+https://github.com/wcardoen/mantisproj.git#subdirectory=mantismod"`
  * `uv add "mantismod[cpu] @ git+https://github.com/wcardoen/mantisproj.git#subdirectory=mantismod"`


```bash
uv run python3
Python 3.13.5 (main, Jul  1 2025, 18:37:36) [Clang 20.1.4 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mantismod
>>> mantismod.__version__
'0.1.0'
```

Let `DIR` be directory where `mypy` is installed.
Then 
```bash
export PATH=$DIR/mypy/.venv/bin/:$PATH
which python3
```

### Some simple tests

```bash
[u0253283@kp298:mypy]$ python3
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
```

