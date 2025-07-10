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
    Move the optimizer's state tensors to the specified device.

    This function iterates through the optimizer's state dictionary and
    moves all tensors to the given device. It is useful when you need to
    change the device of the optimizer's state after initialization,
    especially when working with mixed precision training or device changes.

    Parameters:
    optimizer (torch.optim.Optimizer): The optimizer instance to move.
    device (torch.device or str): The target device (e.g., 'cuda' or 'cpu').

    Returns:
    None: The optimizer is modified in place.

    Raises:
    AttributeError: If the optimizer does not have a 'state_dict' method.
    KeyError: If the optimizer's state_dict does not contain a 'state' key.
    """
    try:
        optimizer_state = optimizer.state_dict()
    except AttributeError:
        raise AttributeError("Optimizer has no state_dict() method.")

    if "state" not in optimizer_state:
        raise KeyError("Optimizer state_dict does not contain 'state' key.")

    device = torch.device(device)

    for state in optimizer_state["state"].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
