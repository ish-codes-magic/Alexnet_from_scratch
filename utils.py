import json
import os
import shutil
from enum import Enum
from typing import Optional

import torch
from torch import nn, optim

__all__ = [
    "accuracy", "load_class_label", "load_state_dict", "load_pretrained_state_dict", "load_resume_state_dict",
    "make_directory", "make_divisible", "save_checkpoint", "Summary", "AverageMeter", "ProgressMeter"
]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def load_state_dict(
        model: nn.Module,
        state_dict: dict,
) -> nn.Module:
    model_state_dict = model.state_dict()

    # Traverse the model parameters and load the parameters in the pre-trained model into the current model
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model

def load_pretrained_state_dict(
        model: nn.Module,
        model_weights_path: str,
) -> nn.Module:
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model = load_state_dict(model, checkpoint["state_dict"])

    return model


def load_resume_state_dict(
        model: nn.Module,
        model_weights_path: str,
        ema_model: nn.Module or None,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
) -> tuple[nn.Module, nn.Module, int, float, optim.Optimizer, optim.lr_scheduler]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    # 加载训练节点参数
    start_epoch = checkpoint["epoch"]

    model = load_state_dict(model, checkpoint["state_dict"])
    ema_model = load_state_dict(ema_model, checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return model, ema_model, start_epoch, optimizer, scheduler


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)
        
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"