from torchvision.transforms import v2 as v2_transforms


def make_augmentor(
    step: int,
    target_steps: int,
    target_magnitude: int,
    img_dim: tuple,
    magnitude_base: int = 4,
    max_magnitude: int = 30,
    aug_num_ops: int = 2,
    norm_mean: tuple = (0.485, 0.456, 0.406),
    norm_std: tuple = (0.229, 0.224, 0.225),
):
    """Creates a torchvision transform pipeline with scheduled augmentation.

    This function constructs a transform pipeline that includes resizing, scheduled
    RandAugment, and normalization. The magnitude of the RandAugment transformations
    is dynamically adjusted based on the current training step, allowing for a
    gradual increase in augmentation strength as training progresses. This can
    help stabilize training in the early stages while still providing strong
    regularization later on.

    Args:
        step (int): The current training step or epoch.
        target_steps (int): The total number of steps over which the augmentation
            magnitude should be scaled.
        target_magnitude (int): The final target magnitude for RandAugment.
        img_dim (tuple): The target image dimensions (height, width) for resizing.
        magnitude_base (int, optional): The initial base magnitude for RandAugment.
            Defaults to 4.
        max_magnitude (int, optional): The absolute maximum magnitude for RandAugment.
            Defaults to 30.
        aug_num_ops (int, optional): The number of augmentation operations to apply
            in RandAugment. Defaults to 2.
        norm_mean (tuple, optional): The mean for normalization.
            Defaults to (0.485, 0.456, 0.406).
        norm_std (tuple, optional): The standard deviation for normalization.
            Defaults to (0.229, 0.224, 0.225).

    Raises:
        ValueError: If `target_magnitude` is greater than `magnitude_base`.

    Returns:
        v2_transforms.Compose: A torchvision transform pipeline.
    """
    if target_magnitude > magnitude_base:
        raise ValueError("target_magnitude can't be higher than magnitude_base")

    return v2_transforms.Compose(
        [
            v2_transforms.Resize(img_dim),
            v2_transforms.RandAugment(
                num_ops=aug_num_ops,
                magnitude=min(
                    round(step / (target_steps / (target_magnitude - magnitude_base))),
                    max_magnitude - magnitude_base,
                )
                + magnitude_base,
            ),
            v2_transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
