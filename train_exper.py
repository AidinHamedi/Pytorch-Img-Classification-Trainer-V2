import pytorch_optimizer as TP_optim
import torch
from custom_onecyclelr import scheduler
from rich import print
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.transforms import v2 as v2_transforms

from training_eng.core.callback_arg import CallbackWrapper
from training_eng.data_utils.data_loader import ImgLoader, make_data_pairs
from training_eng.data_utils.data_proc import make_augmentor
from training_eng.trainer import fit

main_data_dir = "./dataset/train"  # Main dataset dir
eval_data_dir = (
    "./dataset/validation"  # Eval dataset dir (Not needed if "auto_split" is True)
)
img_res = (224, 224)  # img loading resolution (for val)
img_format = "rgb"  # rgb, grayscale
dl_backend = "opencv"  # pil, opencv or turbojpeg (for faster data loading)
dtype = torch.float32  # data type
auto_split = False  # Auto split dataset (Will auto split the data in "main_data_dir" to Train and Test, Wont use "eval_data_dir")
split_ratio = 0.8  # Split (Train&Test) ~ auto_split==True
class_weighting_method = "linear"  # class weighting method
dataLoader_num_workers = 8  # Number of workers for data loading
debug_model_structure = False  # Trace and show a summary of the model


def train(exper_args: dict):  # This is an Example function
    print("[bold green]Starting...")

    train_batchsize = 64
    eval_batchsize = 64
    train_gradient_accumulation = 0

    data_pairs = make_data_pairs(
        train_dir=main_data_dir,
        val_dir=eval_data_dir,
        auto_split=auto_split,
        split_ratio=split_ratio,
        class_weighting_method=class_weighting_method,
    )
    print("[yellow]Data pairs info:")
    for key in data_pairs["stats"]:
        print(f" - {key}: {data_pairs['stats'][key]}")

    eval_dataloader = DataLoader(
        dataset=ImgLoader(
            data_pairs["data_pairs"]["eval"],
            backend=dl_backend,
            color_mode=img_format,
            dtype=dtype,
            transforms=v2_transforms.Compose(
                [
                    v2_transforms.Resize(img_res),
                    v2_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        batch_size=eval_batchsize,
        shuffle=False,
        num_workers=dataLoader_num_workers,
        persistent_workers=False,
        prefetch_factor=2,
        timeout=120,
        pin_memory=True,
        drop_last=False,
    )

    def gen_train_dataloader(**env_args):
        train_dataloader = DataLoader(
            dataset=ImgLoader(
                data_pairs["data_pairs"]["train"],
                backend=dl_backend,
                color_mode=img_format,
                dtype=dtype,
                transforms=make_augmentor(
                    step=env_args["epoch"],
                    target_steps=45,
                    target_magnitude=14,
                    img_dim=img_res,
                    magnitude_base=4,
                    max_magnitude=20,
                ),
            ),
            batch_size=train_batchsize,
            shuffle=True,
            num_workers=dataLoader_num_workers,
            persistent_workers=False,
            prefetch_factor=2,
            timeout=120,
            pin_memory=True,
            drop_last=True,
        )
        return train_dataloader

    print("[bold green]Making the model...")
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_name(
        exper_args["model_name"],
        include_top=True,
        num_classes=data_pairs["num_classes"],
        in_channels=3 if img_format == "rgb" else 1,
    )

    if debug_model_structure:
        print("[yellow]Model summary:")
        print(
            summary(
                model,
                input_size=(1, 3 if img_format == "rgb" else 1, *img_res),
                verbose=0,
                depth=5,
            )
        )

    optimizer_params = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if all(keyword not in name for keyword in ["bias", "bn"])
            ]
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(keyword in name for keyword in ["bias", "bn"])
            ],
            "weight_decay": 0,
        },
    ]
    optimizer = TP_optim.GrokFastAdamW(
        optimizer_params,
        lr=0.01,
        weight_decay=0.02,
    )
    optimizer = TP_optim.Lookahead(optimizer, k=5, alpha=0.5, pullback_momentum="none")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.006)

    print("[bold green]Training the model...")
    fit(
        model,
        CallbackWrapper(gen_train_dataloader),
        CallbackWrapper(lambda _: _, default_value=eval_dataloader, constant=True),
        optimizer,
        loss_fn,
        max_epochs=512,
        mixed_precision=True,
        gradient_accumulation=bool(train_gradient_accumulation),
        gradient_accumulation_steps=CallbackWrapper(
            lambda **env_args: min(3, int(env_args["epoch"] / 10) + 1),
            default_value=train_gradient_accumulation,
            constant=False,
        ),
        early_stopping_cnf={
            "patience": 16,
            "monitor": "Cohen's Kappa",
            "mode": "max",
            "min_delta": 0.00001,
        },
        lr_scheduler={
            "scheduler": scheduler.OneCycleLr(
                optimizer,
                warmup_iters=6,
                lr_idling_iters=16,
                annealing_iters=38,
                decay_iters=80,
                max_lr=0.01,
                annealing_lr_min=0.004,
                decay_lr_min=0.001,
                warmup_type="linear",
            ),
            "enable": True,
            "batch_mode": False,
        },
        model_trace_input=torch.randn(1, 3 if img_format == "rgb" else 1, *img_res),
        experiment_name=exper_args["exper_name"],
        grad_centralization=False,
        cuda_compile=True,
        cuda_compile_config={
            "dynamic": False,
            "fullgraph": True,
            "backend": "inductor",
            # "mode": "max-autotune-no-cudagraphs",
        },
    )
