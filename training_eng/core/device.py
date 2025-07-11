import torch
from rich import print


def get_device(CPU_only=False, verbose=False):
    """
    Returns the appropriate PyTorch device to use, either "cpu", "cuda", or "xpu" if available.

    Args:
        CPU_only (bool, optional): If True, forces the use of the CPU device even if other devices are available. Defaults to False.
        verbose (bool, optional): If True, prints information about the chosen device. Defaults to False.

    Returns:
        torch.device: The PyTorch device to use, either "cpu", "cuda", or "xpu".
    """
    if CPU_only:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Chosen device: [bold green]{device}")
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
            device_capability = torch.cuda.get_device_capability(device)
            print(f"[bold green]CUDA [reset]│ Device Name: [bold white]{device_name}")
            print(
                f"[bold green]CUDA [reset]│ Device Capability: [bold white]{device_capability}"
            )
        elif device.type == "xpu":
            device_name = torch.xpu.get_device_name(device)
            print(f"[bold cyan]XPU [reset]│ Device Name: [bold white]{device_name}")

    return device


def check_device(model_or_tensor):
    """
    Returns the device type (either "cpu", "cuda", or "xpu") for the given PyTorch model or tensor.

    Args:
        model_or_tensor (torch.nn.Module or torch.Tensor): The PyTorch Module or tensor to get the device for.

    Returns:
        str: The device type as a string, either "cpu", "cuda", or "xpu".
    """
    device = (
        next(model_or_tensor.parameters()).device
        if isinstance(model_or_tensor, torch.nn.Module)
        else model_or_tensor.device
    )
    return str(device).split(":")[0]


def move_optimizer_to_device(optimizer, device):
    """
    Moves all optimizer parameters and their associated state to the specified device.
    (Skips gradient tensors)

    Args:
        optimizer: A torch.optim optimizer (e.g., SGD, Adam).
        device: Target device (e.g., "cuda:0" or "cpu").

    Returns:
        None (modifies optimizer in-place).
    """
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)

        elif isinstance(param, dict):
            for sub_param in param.values():
                if isinstance(sub_param, torch.Tensor):
                    sub_param.data = sub_param.data.to(device)
