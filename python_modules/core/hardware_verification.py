"""Inspect available hardware through the shared PyTorch device utility."""

from .computer_device import computer_device


def verify_hardware(min_vram_gb: float | None = 14.0):
    """Print system and GPU details and return an available PyTorch device.

    The optional VRAM threshold produces a warning, not an execution block.
    PyTorch CUDA availability determines whether CUDA or CPU is selected.
    """
    return computer_device(min_vram_gb=min_vram_gb)
