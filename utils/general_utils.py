import os
import csv
import torch
import random
import numpy as np
import shutil, errno


def mkdir_recursive(new_dir):
    os.makedirs(new_dir, exist_ok=True)


def write_all_csv(results, iter_name, column_name, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(([iter_name, column_name]))
        writer.writerows(results)


def write_csv(lists, iter_name, colmun_name, file_name):
    write_all_csv([(i, item) for i, item in enumerate(lists)], iter_name, colmun_name, file_name)


def write_csv_rows(file_name, column_list):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(column_list)


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def convert_relu_to_softplus(model, beta=10):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.Softplus(beta=beta))
        else:
            convert_relu_to_softplus(child, beta)


def convert_relu_to_silu(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.SiLU(inplace=True))
        else:
            convert_relu_to_silu(child)


def model_no_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def model_with_grad(model):
    for param in model.parameters():
        param.requires_grad = True


def model_clear_grad(model):
    for param in model.parameters():
        param.grad = None


def save_checkpoint(
    state, is_best, result_dir, filename="checkpoint.pth.tar"
):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )