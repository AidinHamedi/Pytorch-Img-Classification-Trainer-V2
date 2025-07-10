import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import turbojpeg

    turbojpeg_loaded = True
except ImportError:
    turbojpeg_loaded = False


BACKEND_SUPPORT = {
    "opencv": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
    "pil": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
    "turbojpeg": {".jpg", ".jpeg"},
}


def compute_class_weights_one_hot(y: np.ndarray, weighting: str = "linear"):
    """Computes normalized class weights for multi-label binary one-hot encoded labels.

    This computes the inverse frequency of each class in the provided
    multi-label binary one-hot encoded labels, applies the specified weighting scheme,
    and returns the normalized class weights.

    Parameters:
    y (np.ndarray): Multi-label binary one-hot encoded labels.
    weighting (str): The weighting scheme to apply. Options are 'square', 'sqrt', '1p5_Power', '1p2_Power', 'cube', 'log', and 'linear'.

    Returns:
    np.ndarray: The normalized class weights.

    Intended for computing loss weighting to handle class imbalance in multi-label classification.
    """
    # Count the number of samples in each class
    class_sample_counts = y.sum(axis=0)

    # Compute the inverse of each class count
    class_weights = 1.0 / class_sample_counts.astype(np.float32)

    # Apply the specified weighting scheme
    if weighting == "square":
        class_weights = np.square(class_weights)
    elif weighting == "none":
        class_weights = np.ones_like(class_sample_counts)
    elif weighting == "sqrt":
        class_weights = np.sqrt(class_weights)
    elif weighting == "cube":
        class_weights = np.power(class_weights, 3)
    elif weighting == "1p5_Power":
        class_weights = np.power(class_weights, 1.5)
    elif weighting == "1p2_Power":
        class_weights = np.power(class_weights, 1.2)
    elif weighting == "log":
        class_weights = np.log(class_weights)
    elif weighting != "linear":
        raise ValueError(f"Unknown weighting scheme '{weighting}'")

    # Normalize the class weights so that they sum to 1
    class_weights_normalized = class_weights / np.sum(class_weights)

    # Return the normalized class weights
    return class_weights_normalized


def load_image_opencv(
    img_path: str,
    img_size: tuple = None,
    color_mode: str = "rgb",
    raise_on_error: bool = False,
    default_img: np.ndarray = None,
) -> np.ndarray:
    """
    Load an image using OpenCV with multi-format support and optional resizing.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple, optional): Target image dimensions (width, height). Defaults to None.
        color_mode (str, optional): Color mode ('rgb', 'grayscale'). Defaults to "rgb".
        raise_on_error (bool, optional): Whether to raise an error if the image fails to load. Defaults to False.
        default_img (np.ndarray, optional): The image to return if loading the image file fails. Defaults to None.

    Returns:
        np.ndarray: Loaded image as a NumPy array.

    Raises:
        ValueError: If the color mode is unsupported.
        FileNotFoundError: If the image fails to load and `raise_on_error` is True.
    """
    img = cv2.imread(img_path)
    if img is None:
        if raise_on_error:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        return default_img

    match color_mode:
        case "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        case "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        case _:
            raise ValueError(f"Unsupported color mode: {color_mode}")

    if img_size:
        img = cv2.resize(img, img_size)

    if color_mode == "grayscale":
        return np.expand_dims(img, axis=-1)

    return img


def load_image_pil(
    img_path: str,
    img_size: tuple = None,
    color_mode: str = "rgb",
    raise_on_error: bool = False,
    default_img: np.ndarray = None,
) -> np.ndarray:
    """
    Load an image using PIL with multi-format support and optional resizing.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple, optional): Target image dimensions (width, height). Defaults to None.
        color_mode (str, optional): Color mode ('rgb', 'rgba', 'grayscale'). Defaults to "rgb".
        raise_on_error (bool, optional): Whether to raise an error if the image fails to load. Defaults to False.
        default_img (np.ndarray, optional): The image to return if loading the image file fails. Defaults to None.

    Returns:
        np.ndarray: Loaded image as a NumPy array.

    Raises:
        ValueError: If the color mode is unsupported.
        FileNotFoundError: If the image fails to load and `raise_on_error` is True.
    """
    try:
        img = Image.open(img_path)
    except Exception as e:
        if raise_on_error:
            raise FileNotFoundError(f"Failed to load image: {img_path}") from e
        return default_img

    match color_mode:
        case "grayscale":
            img = img.convert("L")
        case "rgb":
            img = img.convert("RGB")
        case "rgba":
            img = img.convert("RGBA")
        case _:
            raise ValueError(f"Unsupported color mode: {color_mode}")

    if img_size:
        img = img.resize(img_size)

    if color_mode == "grayscale":
        return np.expand_dims(img, axis=-1)

    return np.array(img)


def load_image_turbojpeg(
    img_path: str,
    img_size: tuple = None,
    color_mode: str = "rgb",
    raise_on_error: bool = False,
    default_img: np.ndarray = None,
) -> np.ndarray:
    """
    Load an image using Turbojpeg with multi-format support and optional resizing.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple, optional): Target image dimensions (width, height). Defaults to None.
        color_mode (str, optional): Color mode ('rgb', 'grayscale'). Defaults to "rgb".
        raise_on_error (bool, optional): Whether to raise an error if the image fails to load. Defaults to False.
        default_img (np.ndarray, optional): The image to return if loading the image file fails. Defaults to None.

    Returns:
        np.ndarray: Loaded image as a NumPy array.

    Raises:
        ValueError: If the color mode is unsupported.
        FileNotFoundError: If the image fails to load and `raise_on_error` is True.
    """
    if not turbojpeg_loaded:
        raise RuntimeError("Turbojpeg is not loaded.")

    try:
        with open(img_path, "rb") as fr:
            match color_mode:
                case "grayscale":
                    img = np.array(
                        turbojpeg.decompress(fr.read(), pixelformat=turbojpeg.GRAY)
                    )
                case "rgb":
                    img = np.array(turbojpeg.decompress(fr.read()))
                case _:
                    raise ValueError(f"Unsupported color mode: {color_mode}")
    except Exception as e:
        if raise_on_error:
            raise FileNotFoundError(f"Failed to load image: {img_path}") from e
        return default_img

    if img_size:
        img = cv2.resize(img, img_size)

    if color_mode == "grayscale":
        return np.expand_dims(img, axis=-1)

    return img


def is_supported_file(filename: str, backend: str = "opencv") -> bool:
    """
    Validate if the file extension is supported by the specified backend.

    Args:
        filename (str): Name of the file.
        backend (str, optional): Backend to check support for ('opencv', 'pil', 'turbojpeg'). Defaults to "opencv".

    Returns:
        bool: True if the file extension is supported, False otherwise.
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in BACKEND_SUPPORT.get(backend, set())


class Torch_ImgDataloader(Dataset):
    """
    PyTorch image dataloader with multi-backend support and customizable preprocessing.

    Args:
        data_pairs (list): List of [one-hot label, image path] pairs.
        backend (str, optional): Image loading backend ('opencv', 'pil', 'turbojpeg'). Defaults to "opencv".
        color_mode (str, optional): Color mode ('rgb', 'grayscale', 'rgba'). Defaults to "rgb".
        transforms (callable, optional): Custom transform pipeline. Defaults to None.
        normalize (bool, optional): Whether to normalize pixel values to [0, 1]. Defaults to True.
        dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
        transform_timing (str, optional): When to apply transforms ('pre_norm', 'post_norm'). Defaults to "post_norm".
        raise_on_error (bool, optional): Whether to raise an error on any problem. Defaults to False.

    Raises:
        ValueError: If the backend is unsupported or an image fails to load.
    """

    def __init__(
        self,
        data_pairs,
        backend="opencv",
        color_mode="rgb",
        transforms=None,
        normalize=True,
        dtype=torch.float32,
        transform_timing="post_norm",
        raise_on_error=False,
    ):
        # Initialize instance variables
        self.data_pairs = data_pairs
        self.backend = backend
        self.color_mode = color_mode
        self.transforms = transforms
        self.normalize = normalize
        self.dtype = dtype
        self.transform_timing = transform_timing
        self.raise_on_error = raise_on_error
        match backend:
            case "opencv":
                self.load_func = load_image_opencv
            case "turbojpeg":
                self.load_func = load_image_turbojpeg
            case "pillow":
                self.load_func = load_image_pil
            case _:
                raise ValueError(f"Unsupported backend: {backend}")

    def _process_image(self, img):
        """
        Process the loaded image through the transformation pipeline.

        Args:
            img (np.ndarray): Raw image array.

        Returns:
            torch.Tensor: Processed and transformed image tensor.

        Raises:
            ValueError: If the image is None or has an unsupported shape.
        """
        if img is None:
            raise ValueError("Image is None, cannot process.")

        # Convert the NumPy array to a PyTorch tensor
        img = torch.from_numpy(img).type(self.dtype, non_blocking=True)

        # Change from HWC to CHW format
        img = img.permute(2, 0, 1)

        # Apply transforms before normalization if specified
        if self.transforms and self.transform_timing == "pre_norm":
            img = self.transforms(img)

        # Normalize pixel values to [0, 1] if specified
        if self.normalize:
            img = img / 255.0

        # Apply transforms after normalization if specified
        if self.transforms and self.transform_timing == "post_norm":
            img = self.transforms(img)

        return img

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """Return the processed image tensor and label for the given index."""
        label, img_path = self.data_pairs[idx]
        img = self.load_func(
            img_path, color_mode=self.color_mode, raise_on_error=self.raise_on_error
        )
        if img is None and self.raise_on_error:
            raise ValueError(f"Failed to load image: {img_path}")
        img_tensor = self._process_image(img)
        return img_tensor, label


def make_data_pairs(
    train_dir: str,
    val_dir: str = None,
    auto_split: bool = True,
    split_ratio: float = 0.2,
    class_weighting_method: str = "linear",
) -> dict:
    """
    Prepares and splits image dataset pairs for training and evaluation.

    This function scans a directory of images organized in subdirectories, where each
    subdirectory represents a class. It creates one-hot encoded labels and pairs them
    with their corresponding image paths. The dataset can be automatically split into
    training and evaluation sets, or a separate validation directory can be provided.
    It also computes class weights to handle potential class imbalance in the training set.

    Args:
        train_dir (str): Path to the training directory, with subdirectories for each class.
        val_dir (str, optional): Path to the validation directory. Used if `auto_split` is False. Defaults to None.
        auto_split (bool, optional): If True, automatically splits data from `train_dir` into
            training and validation sets. Defaults to True.
        split_ratio (float, optional): The ratio of the dataset to be used for the
            validation set when `auto_split` is True. Defaults to 0.2.
        class_weighting_method (str, optional): The method for calculating class weights
            (e.g., 'linear', 'sqrt'). Defaults to "linear".

    Raises:
        ValueError: If `auto_split` is False and `val_dir` is not found, or if the class
            labels in `train_dir` and `val_dir` do not match.

    Returns:
        dict: A dictionary containing the processed data and metadata:
            - "data_pairs" (dict): Contains 'train' and 'eval' lists of [one-hot_label, image_path] pairs.
            - "stats" (dict): Contains dataset statistics like image counts and split ratio.
            - "class_weights" (torch.Tensor): The computed class weights for the training set.
            - "num_classes" (int): The total number of classes.
            - "labels" (list): A list of the class label names.
    """
    # Create one-hot encodings for labels using PyTorch
    label_dirs = os.listdir(train_dir)
    labels = [lable.capitalize() for lable in label_dirs]
    label_to_onehot = {
        label.capitalize(): torch.eye(len(label_dirs))[i]
        for i, label in enumerate(label_dirs)
    }

    # Create pairs of [one-hot label, image path]
    data_pairs = []
    for label_dir in label_dirs:
        label_onehot = label_to_onehot[label_dir.capitalize()]
        img_paths = os.listdir(os.path.join(train_dir, label_dir))
        for img_path in img_paths:
            full_path = os.path.join(train_dir, label_dir, img_path)
            data_pairs.append([label_onehot, full_path])

    # Shuffle the pairs
    random.shuffle(data_pairs)

    # Get dataset stats
    num_classes = len(label_dirs)
    image_count = len(data_pairs)

    if auto_split:
        split_idx = int(image_count * split_ratio)
        train_pairs = data_pairs[:split_idx]
        eval_pairs = data_pairs[split_idx:]
        del data_pairs
    else:
        # Verify eval directory exists
        if not os.path.exists(val_dir):
            raise ValueError(f"Evaluation data directory not found: {val_dir}")

        # Verify matching labels
        eval_label_dirs = os.listdir(val_dir)
        if set(eval_label_dirs) != set(label_dirs):
            raise ValueError("Mismatch between training and evaluation labels")

        # Create eval pairs using same label encoding
        eval_pairs = []
        for label_dir in eval_label_dirs:
            label_onehot = label_to_onehot[label_dir.capitalize()]
            img_paths = os.listdir(os.path.join(val_dir, label_dir))
            for img_path in img_paths:
                full_path = os.path.join(val_dir, label_dir, img_path)
                eval_pairs.append([label_onehot, full_path])

        train_pairs = data_pairs
        del data_pairs

    # Split statistics
    eval_count = len(eval_pairs)
    train_count = len(train_pairs)
    total_count = eval_count + train_count
    split_ratio = train_count / total_count

    # Compute the class weights
    class_weights = torch.from_numpy(
        compute_class_weights_one_hot(
            torch.stack([pair[0] for pair in train_pairs]).numpy(),
            weighting=class_weighting_method,
        )
    )

    return {
        "data_pairs": {
            "train": train_pairs,
            "eval": eval_pairs,
        },
        "stats": {
            "main_dir_image_count": image_count,
            "split_ratio": split_ratio,
            "train_count": train_count,
            "eval_count": eval_count,
            "total_count": total_count,
        },
        "class_weights": class_weights,
        "num_classes": num_classes,
        "labels": labels,
    }
