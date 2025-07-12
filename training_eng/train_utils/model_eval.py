from typing import Callable, Dict, Optional, Tuple, Union

import torch
from rich.progress import Progress
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def loss_reduction(loss_fn, y_pred, y):
    if hasattr(loss_fn, "reduction") and loss_fn.reduction == "none":
        losses = loss_fn(y_pred, y)
        loss = losses.mean()
    else:
        loss = loss_fn(y_pred, y)

    return loss


def calc_metrics(y, y_pred, loss_fn, averaging="macro"):
    """
    Calculate various metrics for multi-class classification.

    Args:
        y (torch.Tensor): Ground truth labels, shape (batch_size, num_classes)
        y_pred (torch.Tensor): Model predictions, shape (batch_size, num_classes)
        loss_fn (callable): The loss function used during training

    Returns:
        dict: A dictionary containing various evaluation metrics
    """
    epsilon = 1e-10

    def safe_metric_calculation(metric_fn, *args, **kwargs):
        try:
            return metric_fn(*args, **kwargs)
        except Exception:
            return epsilon

    metrics_dict = {
        "Loss": float(safe_metric_calculation(loss_reduction, loss_fn, y_pred, y))
    }

    y = y.type(torch.float32, non_blocking=True).numpy()
    y_pred = y_pred.type(torch.float32, non_blocking=True).numpy()

    y_pred_labels = y_pred.argmax(axis=1)
    y_labels = y.argmax(axis=1)

    metrics_dict.update(
        {
            f"F1 Score ({averaging})": safe_metric_calculation(
                f1_score, y_labels, y_pred_labels, average=averaging
            ),
            f"Precision ({averaging})": safe_metric_calculation(
                precision_score,
                y_labels,
                y_pred_labels,
                average=averaging,
                zero_division=0,
            ),
            f"Recall ({averaging})": safe_metric_calculation(
                recall_score, y_labels, y_pred_labels, average=averaging
            ),
            "AUROC": float(
                safe_metric_calculation(roc_auc_score, y, y_pred, multi_class="ovr")
            ),
            "Accuracy": safe_metric_calculation(
                accuracy_score, y_labels, y_pred_labels
            ),
            "Cohen's Kappa": float(
                safe_metric_calculation(cohen_kappa_score, y_labels, y_pred_labels)
            ),
            "Matthews Correlation Coefficient": float(
                safe_metric_calculation(matthews_corrcoef, y_labels, y_pred_labels)
            ),
        }
    )

    return metrics_dict


def eval(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    Progressbar: Progress,
    loss_fn: torch.nn.Module = None,
    return_preds: bool = False,
    verbose: bool = True,
    **kwargs,
) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]:
    """
    Evaluates the model on the provided dataloader for multi-class classification.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing evaluation data.
        model (torch.nn.Module): The PyTorch model to evaluate.
        device (torch.device): The device to run the evaluation on.
        Progressbar (Progress): The progress bar object.
        loss_fn (Optional[Callable]): The loss function for evaluation (e.g., CrossEntropyLoss). If None, loss is not calculated.
        return_preds (bool, optional): Whether to return model predictions and original labels. Defaults to False.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.
        **kwargs: Additional keyword arguments.
            - progbar_desc (str): Custom description for the progress bar.

    Returns:
        Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]: A dictionary containing various evaluation metrics, and optionally the model predictions and original labels.
    """
    model.eval()
    all_y = []
    all_y_pred = []

    task = Progressbar.add_task(
        kwargs.get("progbar_desc", "Evaluation"), total=len(dataloader)
    )

    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x.to(device, non_blocking=True))
            all_y.append(y.detach().cpu())
            all_y_pred.append(y_pred.detach().cpu())
            Progressbar.update(task, advance=1)

    Progressbar.stop_task(task)

    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)

    metrics = calc_metrics(all_y, all_y_pred, loss_fn.cpu() if loss_fn else None)

    if return_preds:
        return metrics, all_y_pred, all_y
    else:
        return metrics
