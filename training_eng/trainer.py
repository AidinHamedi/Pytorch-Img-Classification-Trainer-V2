import gc
import os
import shutil
import time
from contextlib import suppress
from functools import partial

import numpy as np
import pytorch_optimizer as po
import shortuuid
import torch
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import GradScaler, autocast, nn
from torch.utils.tensorboard import SummaryWriter

from .core.callback_arg import CallbackWrapper
from .core.device import get_device, move_optimizer_to_device
from .core.misc import format_seconds, make_grid
from .core.misc import retrieve_samples as dl_retrieve_samples
from .train_utils.early_stopping import EarlyStopping
from .train_utils.model_eval import calc_metrics
from .train_utils.model_eval import eval as eval_model


def fit(
    model: nn.Module,
    train_dataloader: CallbackWrapper,
    test_dataloader: CallbackWrapper,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    max_epochs: int = 512,
    early_stopping_cnf: dict = {
        "patience": 24,
        "monitor": "Cohen's Kappa",
        "mode": "max",
        "min_delta": 0.00001,
    },
    train_eval_portion: float = 0.1,
    gradient_accumulation: bool = True,
    gradient_accumulation_steps: CallbackWrapper = CallbackWrapper(
        lambda x: x, default_value=4, constant=True
    ),
    mixed_precision: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    lr_scheduler: dict = {
        "scheduler": None,
        "enable": False,
        "batch_mode": False,
    },
    experiment_name: str = "!auto",
    cache_dir: str = "./cache",
    model_export_path: str = "./models",
    tensorboard_logs_path: str = "./logs",
    model_trace_input: torch.Tensor = None,  # type: ignore
    cuda_compile: bool = False,
    grad_centralization: bool = False,
    cuda_compile_config: dict = {
        "dynamic": False,
        "fullgraph": True,
        "backend": "inductor",
    },
    log_debugging: bool = True,
    min_training_epochs: int = 2,
    force_cpu: bool = False,
):
    """
    Trains a PyTorch model with support for various features like early stopping, mixed precision, and logging.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (CallbackWrapper): A wrapper for the training dataloader that allows for dynamic updates.
        test_dataloader (CallbackWrapper): A wrapper for the testing dataloader.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        loss_fn (nn.Module): The loss function.
        max_epochs (int, optional): The maximum number of epochs to train. Defaults to 512.
        early_stopping_cnf (dict, optional): Configuration for early stopping. Defaults to {"patience": 24, "monitor": "Cohen's Kappa", "mode": "max", "min_delta": 0.00001}.
        train_eval_portion (float, optional): The portion of training data to use for evaluation. Defaults to 0.1.
        gradient_accumulation (bool, optional): Whether to use gradient accumulation. Defaults to True.
        gradient_accumulation_steps (CallbackWrapper, optional): The number of gradient accumulation steps. Defaults to 4.
        mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to True.
        mixed_precision_dtype (torch.dtype, optional): The data type for mixed precision. Defaults to torch.bfloat16.
        lr_scheduler (dict, optional): Configuration for the learning rate scheduler. Defaults to {"scheduler": None, "enable": False, "batch_mode": False}.
        experiment_name (str, optional): The name of the experiment. Defaults to "!auto".
        cache_dir (str, optional): The directory to cache intermediate files. Defaults to "./cache".
        model_export_path (str, optional): The path to save the trained models. Defaults to "./models".
        tensorboard_logs_path (str, optional): The path to save TensorBoard logs. Defaults to "./logs".
        model_trace_input (torch.Tensor, optional): An example input for model tracing. Defaults to None.
        cuda_compile (bool, optional): Whether to compile the model using `torch.compile`. Defaults to False.
        grad_centralization (bool, optional): Whether to use gradient centralization. Defaults to False.
        cuda_compile_config (dict, optional): Configuration for `torch.compile`. Defaults to {"dynamic": False, "fullgraph": True, "backend": "inductor"}.
        log_debugging (bool, optional): Whether to log debugging information. Defaults to True.
        min_training_epochs (int, optional): The minimum number of epochs to train before saving the model. Defaults to 2.
        force_cpu (bool, optional): Whether to force CPU usage. Defaults to False.

    Returns:
        dict: A dictionary containing the best model and the history of metrics.
    """
    console = Console()

    if experiment_name == "!auto":
        experiment_name = f"{time.strftime('%Y-%m-%d %H-%M-%S')}"
    else:
        experiment_name = f"{shortuuid.ShortUUID().random(length=8)}~{experiment_name}"  # Avoid duplicates

    console.print(
        f"[bold green]Initializing... [default](Experiment name: [yellow]{experiment_name}[default])"
    )

    start_time = time.time()

    device = get_device(verbose=True, CPU_only=force_cpu)
    # console.print(f"Chosen device: [bold green]{device}[default]")
    device_str = str(device)

    model = model.to(device, non_blocking=True)
    move_optimizer_to_device(optimizer, device)

    tb_log_dir = os.path.join(tensorboard_logs_path, "runs", experiment_name)
    console.print(f"Tensorboard log dir: [green]{tb_log_dir}")
    if log_debugging:
        tbw_data = SummaryWriter(log_dir=os.path.join(tb_log_dir, "data"), max_queue=25)
    tbw_val = SummaryWriter(log_dir=os.path.join(tb_log_dir, "val"), flush_secs=45)
    tbw_train = SummaryWriter(log_dir=os.path.join(tb_log_dir, "train"), flush_secs=45)

    if log_debugging and model_trace_input is not None:
        tbw_data.add_graph(model, model_trace_input.to(device))  # type: ignore

    if device_str == "cuda":  # Enable onednn + cuda optimizations
        torch.jit.enable_onednn_fusion(True)
        torch.backends.cudnn.benchmark = True

        if cuda_compile:
            console.print(f"Compiling model with: [green]{cuda_compile_config}")
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 32
            model = torch.compile(model, **cuda_compile_config)  # type: ignore
            console.print(
                "[red]Warning[reset]: The first time you run this model, it will be slow! (Using torch compile)"
            )
            # if log_debugging:
            #     console.print(
            #         "[red]Warning[reset]: When using torch compile some log_debugging features may not work properly!"
            #     )
    elif cuda_compile:
        console.print(
            "[red]Warning[reset]: cuda_compile is only available for cuda devices!"
        )

    model_save_path = os.path.join(model_export_path, experiment_name)
    os.makedirs(model_save_path, exist_ok=True)
    console.print(f"Model save path: [green]{model_save_path}")

    def _lr_scheduler_step(scope):
        if lr_scheduler["enable"] and (scope == "batch") == lr_scheduler["batch_mode"]:
            lr_scheduler["scheduler"].step()

    class _GradientCentralizer:
        def __init__(self, gc_conv_only=False):
            self.gc_conv_only = gc_conv_only
            self._compiled_gc = torch.compile(
                partial(po.centralize_gradient, gc_conv_only=gc_conv_only),
                dynamic=True,
                fullgraph=True,
            ) if device_str == "cuda" else partial(po.centralize_gradient, gc_conv_only=gc_conv_only)

        @torch.no_grad()
        def apply(self, model):
            for param in model.parameters():
                if param.grad is not None:
                    self._compiled_gc(param.grad)

    if grad_centralization:
        grad_cent = _GradientCentralizer()

    early_stopping = EarlyStopping(
        monitor_name=early_stopping_cnf["monitor"],
        mode=early_stopping_cnf["mode"],
        patience=early_stopping_cnf["patience"],
        min_delta=early_stopping_cnf["min_delta"],
        verbose=True,
    )

    mpt_scaler = GradScaler(device=device_str, enabled=mixed_precision)
    metrics_hist = {"train": [], "eval": []}
    train_total_fp = 0

    try:
        for epoch in range(1, max_epochs):
            epoch_start_time = time.time()

            console.print(
                f"\n[bold bright_white]Epoch [green]{epoch}[bold]/[cyan]{max_epochs} [yellow]-->"
            )

            with console.status("[bold grey42]Preparing..."):
                test_dataloader.update_value(epoch=epoch)
                train_dataloader.update_value(epoch=epoch)
                gradient_accumulation_steps.update_value(epoch=epoch)

                if log_debugging:
                    tbw_data.add_image(
                        "Train-Dataloader",
                        make_grid(
                            torch.stack(
                                dl_retrieve_samples(
                                    train_dataloader,
                                    num_samples=9,
                                    selection_method="random",
                                    seed=42,
                                )
                            ),
                            nrow=3,
                            padding=2,
                            normalize=True,
                            pad_value=0,
                            format="CHW",
                        ),
                        epoch - 1,
                    )

                model.train()
                loss_fn = loss_fn.to(device, non_blocking=True)
                train_dataloader_len = train_dataloader.__len__()
                train_total_batches = (
                    int(train_dataloader_len / gradient_accumulation_steps)
                    if gradient_accumulation
                    else train_dataloader_len
                )
                train_eval_data_len = round(train_total_batches * train_eval_portion)
                train_eval_data = []
                train_losses = []
                batch_idx = 0

                console.print(
                    f"Train batch size: [cyan]{train_dataloader.batch_size * (gradient_accumulation_steps if gradient_accumulation else 1)}"
                )
                console.print(f"Train eval data len: [cyan]{train_eval_data_len}")
                console.print(f"Learning rate: [cyan]{optimizer.param_groups[0]['lr']}")

                # next(iter(train_dataloader))  # Warm up the dataloader

            progress_bar = Progress(
                SpinnerColumn(finished_text="[yellow]⠿"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(show_speed=True),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            )
            training_task = progress_bar.add_task(
                "Training",
                total=train_total_batches,
            )

            progress_bar.start()

            for fp_idx, (x, y) in enumerate(train_dataloader):
                with autocast(
                    device_type=device_str,
                    enabled=mixed_precision,
                    dtype=mixed_precision_dtype,
                ):
                    # if (
                    #     cuda_compile
                    #     and device_str == "cuda"
                    #     and cuda_compile_config.get("backend", "")
                    #     in ["cudagraphs"]
                    # ):
                    #     cudagraph_mark_step_begin()
                    y_pred = model(x.to(device, non_blocking=True))
                    loss = loss_fn(y_pred, y.to(device, non_blocking=True))

                train_losses.append(loss.item())

                if gradient_accumulation:
                    loss = loss / gradient_accumulation_steps

                if mixed_precision:
                    mpt_scaler.scale(loss).backward()
                else:
                    loss.backward()

                if not gradient_accumulation or (
                    (fp_idx + 1) % gradient_accumulation_steps == 0
                ):
                    batch_idx += 1

                    if mixed_precision or grad_centralization:
                        mpt_scaler.unscale_(optimizer)

                    if grad_centralization:
                        grad_cent.apply(model)

                    if mixed_precision:
                        mpt_scaler.step(optimizer)
                        mpt_scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                    _lr_scheduler_step("batch")

                    if batch_idx >= (train_total_batches - train_eval_data_len):
                        train_eval_data.append(
                            {
                                "y_pred": y_pred.detach().to("cpu", non_blocking=True),
                                "y": y.to("cpu", non_blocking=True),
                            }
                        )

                        progress_bar.update(
                            training_task,
                            advance=1,
                            description="Training (Recording Eval Data)"
                            if batch_idx != train_total_batches
                            else "Training",
                        )
                    else:
                        progress_bar.update(training_task, advance=1)

            progress_bar.stop_task(training_task)

            loss_fn = loss_fn.cpu()

            _lr_scheduler_step("epoch")

            train_eval = calc_metrics(
                torch.cat([item["y"] for item in train_eval_data]),
                torch.cat([item["y_pred"] for item in train_eval_data]),
                loss_fn,
            )

            test_eval = eval_model(
                test_dataloader,
                model,
                device,
                loss_fn=loss_fn,
                Progressbar=progress_bar,
            )

            progress_bar.stop()

            gc.collect()
            if device_str == "cuda":
                torch.cuda.empty_cache()

            metrics_hist["train"].append(train_eval)
            metrics_hist["eval"].append(test_eval)

            eval_table = Table(box=box.ROUNDED, highlight=True)
            eval_table.add_column("Set", justify="center", style="bold green")
            for metric in test_eval:
                eval_table.add_column(metric, justify="center")
            for metric_set in [[train_eval, "train"], [test_eval, "eval"]]:
                eval_table.add_row(
                    metric_set[1],
                    *[
                        f"{metric_set[0][metric]:.5f}"
                        if isinstance(metric_set[0][metric], float)
                        else metric_set[0][metric]
                        for metric in test_eval
                    ],
                )
            console.print(eval_table)

            for metric in train_eval:
                tbw_train.add_scalar(f"Metrics/{metric}", train_eval[metric], epoch)
                tbw_val.add_scalar(f"Metrics/{metric}", test_eval[metric], epoch)
            for i, batch_loss in enumerate(train_losses, start=1):
                tbw_train.add_scalar(
                    "Metrics/Iter-Loss",
                    batch_loss,
                    train_total_fp + i,
                )
            tbw_data.add_histogram("Loss/Train", np.asarray(train_losses), epoch)
            if lr_scheduler.get("enable", False):
                tbw_train.add_scalar(
                    "Other/Train-LR",
                    lr_scheduler["scheduler"].get_last_lr()[0],
                    epoch,
                )

            if log_debugging:
                for name, param in model.named_parameters():
                    param_tag, param_type = (
                        ">".join(name.replace(".", ">").split(">")[:-1]),
                        name.replace(".", ">").split(">")[-1],
                    )
                    if param.data.numel() > 0 and not torch.all(
                        torch.isnan(param.data)
                    ):
                        tbw_data.add_histogram(
                            f"Train-Parameters|>>{param_tag}/{param_type}",
                            param.data.cpu(),
                            epoch,
                        )

            tbw_data.add_scalar(
                "Other/Epoch_time (minutes)",
                (time.time() - epoch_start_time) / 60,
                epoch,
            )

            train_total_fp += train_dataloader_len

            torch.save(model, os.path.join(model_save_path, "latest_model.pth"))

            console.print(
                f"Epoch time: [cyan]{format_seconds(time.time() - epoch_start_time)}"
            )

            early_stopping.update(
                epoch, test_eval[early_stopping_cnf["monitor"]], model
            )
            if early_stopping.should_stop:
                print("Stopping the training early...")
                break

    except KeyboardInterrupt:
        console.print(
            "\n\n[bold red]KeyboardInterrupt detected.[yellow] Stopping the training..."
        )
    except Exception:
        console.print("\n\n[bold red]An error occurred during training.")
        console.print_exception(show_locals=False)

    with suppress(Exception):
        progress_bar.stop()
        console.print("[underline]Successfully closed the progress bar.")

    with suppress(Exception):
        if epoch > min_training_epochs:
            early_stopping.load_best_model(model, raise_error=True, verbose=False)
            console.print("[underline]Successfully loaded the best model.")
            torch.save(model, os.path.join(model_save_path, "best_model.pth"))
            console.print("[underline]Successfully saved the best model.")
        else:
            console.print(
                "Training was too short, deleting the model save path... (delete it manually if no confirmation is given)"
            )
            shutil.rmtree(model_save_path)
            console.print("[underline]Successfully deleted the model save path.")

    with suppress(Exception):
        tbw_val.close()
        tbw_train.close()
        console.print("[underline]Successfully closed the tensorboard writers.")
        tbw_data.close()

    with suppress(Exception):
        if not epoch > min_training_epochs:
            console.print(
                "Tensorboard logs are too short, deleting them... (delete them manually if no confirmation is given)"
            )
            shutil.rmtree(tb_log_dir)
            console.print("[underline]Successfully deleted the short tensorboard logs.")

    if epoch > min_training_epochs:
        console.print(
            f"[yellow]Best model from epoch [green]{early_stopping.best_epoch}[yellow] metrics: "
        )
        result_table = Table(box=box.ROUNDED, highlight=True)
        result_table.add_column("Set", justify="center", style="bold green")
        for metric in metrics_hist["eval"][early_stopping.best_epoch - 1]:
            result_table.add_column(metric, justify="center")
        for metric_set in [
            [metrics_hist["train"][early_stopping.best_epoch - 1], "train"],
            [metrics_hist["eval"][early_stopping.best_epoch - 1], "eval"],
        ]:
            result_table.add_row(
                metric_set[1],
                *[
                    f"{metric_set[0][metric]:.5f}"
                    if isinstance(metric_set[0][metric], float)
                    else metric_set[0][metric]
                    for metric in test_eval
                ],
            )
        console.print(result_table)

    console.print(
        f"Training completed in: [cyan]{format_seconds(time.time() - start_time)}"
    )

    return {
        "best_model": model,
        "metrics_hist": metrics_hist,
    }
