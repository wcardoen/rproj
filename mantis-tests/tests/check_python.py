import platform
import sys
import importlib.metadata
import importlib.util

print(f"*** Check Python Installation ***")

print(f"\nPython version:{platform.python_version()}\n")

print(f"Check packages:")
packages= ["numpy", "pandas", "torch", "mantis"]
try:
    for pkg in packages:
        found = importlib.util.find_spec(pkg) is not None
        if found:
            version = importlib.metadata.version(pkg)
            print(f"  {pkg:<12} ({version:8s}) {'-> installed'}")
        else:
            print(f"  {pkg}: {'-> NOT installed'}")
        

    import torch
    print(f"\nCheck CUDA")
    if torch.cuda.is_available()==True:
        print(f"  Torch CUDA available?: True")
        print(f"  #CUDA device found:{torch.cuda.device_count()}")
        for idevice in range(torch.cuda.device_count()):
            print(f"    device_id:{idevice+1} -> {torch.cuda.get_device_name(idevice)}")

    else:
        print(f"CUDA is NOT available!\n")

except Exception as err:
    sys.exit(f"ERROR:: {err}")

print(f"Done with Python checks!")
