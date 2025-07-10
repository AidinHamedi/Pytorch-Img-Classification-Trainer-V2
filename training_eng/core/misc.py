import math
import torch
import random
from typing import List, Any, Tuple, Union
from torch.utils.data import DataLoader


def format_seconds(seconds: int) -> str:
    """
    Converts a given number of seconds into a human-readable time string.

    Parameters:
        seconds (int): The total number of seconds.

    Returns:
        str: A string representing the time in the format of hs ms s,
             where h, m, and s are hours, minutes, and seconds respectively.
             Only includes units with non-zero values.
    """
    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    seconds = round(remaining % 60)

    time_parts = []
    if hours > 0:
        time_parts.append(f"{int(hours)}h")
    if minutes > 0:
        time_parts.append(f"{int(minutes)}m")
    if seconds > 0:
        time_parts.append(str(seconds) + "s")

    if time_parts == []:
        return "0s"

    return " ".join(time_parts)


def retrieve_samples(
    data_loader: DataLoader,
    num_samples: int = 50,
    selection_method: str = "fully_random",
    seed: int = 42,
    return_labels: bool = False,
) -> Union[List[Any], List[Tuple[Any, Any]]]:
    """
    Retrieves a list of images from a DataLoader without iterating through it.

    Parameters:
        data_loader (DataLoader): The DataLoader instance from which to retrieve samples.
        num_samples (int, optional): The number of samples to retrieve. Defaults to 50.
        selection_method (str, optional): The method to select samples.
            Can be 'fully_random', 'random', 'from_start', or 'from_end'. Defaults to 'fully_random'.
        seed (int, optional): The seed for random number generator for reproducibility.
            Only used when selection_method is 'random'. Defaults to 42.
        return_labels (bool, optional): Whether to return labels along with images.
            Defaults to False.

    Returns:
        List[Any] or List[Tuple[Any, Any]]:
            - If return_labels is False: A list of image tensors.
            - If return_labels is True: A list of tuples, each containing an image tensor and its corresponding label.

    Raises:
        IndexError: If the dataset does not have enough samples.
        ValueError: If an invalid selection_method is provided.
    """
    dataset = data_loader.dataset
    total_samples = len(dataset)

    if num_samples > total_samples:
        raise IndexError(
            f"Cannot retrieve {num_samples} samples from a dataset of size {total_samples}"
        )

    if selection_method == "random":
        random.seed(seed)
        indices = random.sample(range(total_samples), num_samples)
    elif selection_method == "from_start":
        indices = range(num_samples)
    elif selection_method == "from_end":
        indices = range(total_samples - num_samples, total_samples)
    elif selection_method != "fully_random":
        raise ValueError(
            "Invalid selection_method. Choose 'random', 'from_start', or 'from_end'."
        )

    samples = [dataset[i] for i in indices]

    if isinstance(samples[0], (tuple, list)):
        if return_labels:
            return samples
        else:
            return [sample[0] for sample in samples]
    else:
        return samples


def make_grid(
    tensor,
    nrow=8,
    padding=2,
    normalize=False,
    scale_each=False,
    pad_value=0,
    format="CHW",
):
    """
    Arrange a tensor of images into a grid layout.

    Parameters:
        tensor (Tensor): 4D tensor of shape (B, C, H, W) or 3D tensor of shape (C, H, W).
        nrow (int): Number of images displayed in each row of the grid.
        padding (int): Padding between images.
        normalize (bool): Whether to normalize tensor values to the range [0, 1].
        scale_each (bool): Whether to normalize each image in the batch individually.
        pad_value (float): Value used to pad the grid image.
        format (str): Output format of the grid, 'CHW' or 'HWC'.

    Returns:
        Tensor: Grid image tensor in the specified format.

    Example:
        grid = make_grid(tensor, nrow=4, padding=2, normalize=True, format='HWC')
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError("Input tensor should be 3D or 4D.")

    B, C, H, W = tensor.size()
    nimg = B
    nrows = nrow
    ncols = int(math.ceil(float(nimg) / float(nrows)))

    if normalize:
        if scale_each:
            tensors = [(x - x.min()) / (x.max() - x.min()) for x in tensor]
            tensor = torch.stack(tensors)
        else:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    pad_h = padding * (ncols - 1)
    pad_w = padding * (nrows - 1)

    grid_height = H * nrows + pad_h
    grid_width = W * ncols + pad_w
    grid = torch.full(
        (C, grid_height, grid_width),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    for i in range(nimg):
        row = i // nrows
        col = i % nrows
        top = row * (H + padding)
        left = col * (W + padding)
        grid[:, top : top + H, left : left + W] = tensor[i]

    if format == "HWC":
        grid = grid.permute(1, 2, 0)
    elif format != "CHW":
        raise ValueError("Invalid format specified. Choose 'CHW' or 'HWC'.")

    return grid
