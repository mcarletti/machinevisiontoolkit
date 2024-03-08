from torch.utils.tensorboard import SummaryWriter


class EarlyStopping():

    def __init__(self, patience: int=50) -> None:
        """
        Early stopping callback.

        Parameters
        ----------
        `patience` (int): number of epochs to wait for improvement.
        """
        assert patience >= 1, "Invalid patience value. Must be >= 1"
        self.best_result = 0.0
        self.best_epoch = 0
        self.patience = patience
        self.stop = False

    def __call__(self, epoch: int, result: float) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        `epoch` (int): current epoch.
        `result` (float): validation result.

        Returns
        -------
        `stop` (bool): stop training or not.
        """

        if result >= self.best_result:
            self.best_epoch = epoch
            self.best_result = result

        delta = epoch - self.best_epoch
        self.stop = delta >= self.patience

        return self.stop


class Logger():

    def __init__(self, logdir: str) -> None:
        """
        Logger callback.

        Parameters
        ----------
        `logdir` (str): directory to save logs.
        """
        self.writer = SummaryWriter(logdir)

    def __call__(self, updates: dict, global_step: int) -> None:
        """
        Log updates.

        Parameters
        ----------
        `updates` (dict): dictionary of updates.
        `global_step` (int): global step number (eg. epoch number).
        """
        for title, value in updates.items():
            self.writer.add_scalar(title, value, global_step)
        self.writer.flush()
