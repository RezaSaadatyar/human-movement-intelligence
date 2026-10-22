# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Inspect compute resources and choose a PyTorch CUDA or CPU device."""

import platform
import shutil
import subprocess


# ============================ 1. Computer Device =============================
def computer_device(min_vram_gb=14.0):
    """
    Report operating-system, CPU, RAM and NVIDIA/PyTorch GPU information.

    Args:
        min_vram_gb (float | None): Optional total-VRAM warning threshold.
            This is guidance, not a minimum requirement for computation.

    Returns:
        torch.device: CUDA device when available to PyTorch; otherwise CPU.

    Notes:
        PyTorch availability determines the selected device. An NVIDIA
        command-line tool alone cannot establish CUDA runtime readiness.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "computer_device requires PyTorch in the active environment."
        ) from exc

    if min_vram_gb is not None and min_vram_gb < 0:
        raise ValueError("min_vram_gb must be non-negative or None.")

    print('Compute device and system information')

    # ------------------------ 1. Inspect System Resources ------------------------
    cpu_name = platform.processor() or platform.machine()
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {cpu_name}")
    try:
        import psutil
    except ImportError:
        print("Installed RAM: unavailable (optional psutil not installed)")
    else:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"Installed RAM: {ram_gb:.1f} GiB")

    # ------------------- 2. Inspect NVIDIA Driver Information --------------------
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result = subprocess.run(
                [smi, '--query-gpu=name,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=False, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                for record in result.stdout.strip().splitlines():
                    name, memory_mib = record.rsplit(',', 1)
                    print(f"NVIDIA GPU: {name.strip()}")
                    try:
                        size_gib = float(memory_mib.strip()) / 1024
                        print(f"Total VRAM: {size_gib:.2f} GiB")
                    except ValueError:
                        print(f"Reported VRAM: {memory_mib.strip()} MiB")
            else:
                print("NVIDIA driver information is unavailable.")
        except (OSError, subprocess.TimeoutExpired):
            print("NVIDIA driver query could not be completed.")
    else:
        print("nvidia-smi is not available in this environment.")

    # -------------------- 3. Check PyTorch and Select Device ---------------------
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"PyTorch total VRAM: {vram_gib:.2f} GiB")
        if min_vram_gb is not None and vram_gib < min_vram_gb:
            print(
                f"VRAM is below the optional {min_vram_gb:g} GiB threshold; "
                "check the memory needs of the current experiment."
            )
    else:
        device = torch.device("cpu")
        print("CUDA is not available to PyTorch; using CPU.")
    print(f"Selected compute device: {device}")
    return device
