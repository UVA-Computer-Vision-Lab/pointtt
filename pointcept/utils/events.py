"""
Events Utils

Modified from Detectron2
Modified for PoinTTT: only the parts the trainer uses are kept (EventStorage with running
averages, ExceptionWriter); the event writers, HistoryBuffer and the storage stack are removed.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
import traceback
import sys

from collections import defaultdict

__all__ = [
    "EventStorage",
    "ExceptionWriter",
]


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.
    """

    def __init__(self):
        self._history = defaultdict(AverageMeter)

    def put_scalar(self, name, value, n=1):
        """
        Add a scalar `value` to the `AverageMeter` associated with `name`.
        """
        self._history[name].update(value, n)

    def history(self, name):
        """
        Returns:
            AverageMeter: the history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def reset_histories(self):
        for name in self._history.keys():
            self._history[name].reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count


class ExceptionWriter:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            tb = traceback.format_exception(exc_type, exc_val, exc_tb)
            formatted_tb_str = "".join(tb)
            self.logger.error(formatted_tb_str)
            sys.exit(1)  # This prevents double logging the error to the console
