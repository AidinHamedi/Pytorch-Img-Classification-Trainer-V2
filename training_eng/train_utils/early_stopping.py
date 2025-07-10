import glob
import os
import warnings

import torch
from rich import print

warnings.filterwarnings("ignore", category=FutureWarning)


class EarlyStopping:
    """Monitor a metric and stop training if no improvement is observed.

    Args:
        monitor_name (str): The name of the metric to monitor (e.g., "loss", "accuracy").
        mode (str, optional): The optimization direction for the monitored metric.
            Use "min" to minimize the metric (e.g., loss) or "max" to maximize it (e.g., accuracy).
            Defaults to "max".
        patience (int, optional): The number of epochs to wait for an improvement in the monitored metric before stopping training. Defaults to 10.
        min_delta (float, optional): The minimum change in the monitored metric to be considered as an improvement. Defaults to 0.
        cache_dir (str, optional): The directory to save the best model during training. Defaults to "./cache/early_stopping".
        verbose (bool, optional): Whether to print messages about early stopping events. Defaults to True.
    """

    def __init__(
        self,
        monitor_name: str,
        mode: str = "max",
        patience: int = 10,
        min_delta: float = 0,
        cache_dir: str = "./cache/early_stopping",
        verbose: bool = True,
    ):
        # Initialize instance variables
        self.monitor_name = monitor_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.best_epoch = 0
        self.best_monitor = float("inf") if mode == "min" else -float("inf")
        self.should_stop = False
        self.model_save_name = "best_model.pth"
        self.print_indicator = "[[yellow]Early stopping[reset]]"

        # Make the cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            # Clear the cache directory
            [os.remove(file) for file in glob.glob(f"{cache_dir}/*.pth")]

    def _improve_function(self, new: float, old: float) -> bool:
        """Determine if the new metric value is better than the old one based on the mode and minimum delta.

        Args:
            new (float): The new metric value.
            old (float): The old metric value.

        Returns:
            bool: True if the new value is better, False otherwise.
        """
        return (
            new >= old + self.min_delta
            if not self.mode == "min"
            else new <= old - self.min_delta
        )

    def update(
        self, epoch: int, monitor: float, model: torch.nn.Module, verbose: bool = None
    ):
        """Update the early stopping state based on the current epoch's monitored metric value.

        Args:
            epoch (int): The current epoch number.
            monitor (float): The current value of the monitored metric.
            model (torch.nn.Module): The model to be saved if it has the best monitored metric value.
            verbose (bool, optional): Whether to print messages about early stopping events. Defaults to None (uses the class's verbose attribute).
        """
        if verbose is None:
            verbose = self.verbose

        if self._improve_function(monitor, self.best_monitor):
            self.best_epoch = epoch
            self.best_monitor = monitor
            self.should_stop = False
            if verbose:
                print(
                    f"{self.print_indicator} [green]New best {self.monitor_name}[reset] ({self.mode}): {self.best_monitor:.4f} at epoch {self.best_epoch}"
                )
            # Save the model
            torch.save(
                model.state_dict(), os.path.join(self.cache_dir, self.model_save_name)
            )
        elif epoch - self.best_epoch >= self.patience:
            self.should_stop = True
            if verbose:
                print(
                    f"{self.print_indicator} [red]Triggered! [reset]Best results: Epoch {self.best_epoch} with {self.monitor_name} of {self.best_monitor:.4f}"
                )

    def load_best_model(
        self, model: torch.nn.Module, raise_error: bool = False, verbose: bool = None
    ):
        """Load the best model from the cache directory.

        Args:
            model (torch.nn.Module): The model to load the best state into.
            raise_error (bool, optional): Whether to raise an error if the best model is not found. Defaults to False.
            verbose (bool, optional): Whether to print a message about loading the best model. Defaults to None (uses the instance's verbose value).
        """
        if verbose is None:
            verbose = self.verbose

        if os.path.exists(os.path.join(self.cache_dir, self.model_save_name)):
            model.load_state_dict(
                torch.load(os.path.join(self.cache_dir, self.model_save_name))
            )
            if verbose:
                print(
                    f"{self.print_indicator} [green]Loaded the best model from {self.cache_dir}"
                )
        else:
            if verbose:
                print(f"{self.print_indicator} [red]No model found in {self.cache_dir}")
            if raise_error:
                raise FileNotFoundError(
                    f"No model found in {self.cache_dir}. Please check the cache directory."
                )
