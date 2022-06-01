from __future__ import print_function
import os
import argparse
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import time

from model_zoo import *
from datasets import *
from attack import PGD
from utils.general_utils import write_csv_rows


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--device', default="cuda:0", help="The name of the device you want to use (default: None)")
parser.add_argument('--time_stamp', default="00000000",
                    help="The time stamp that helps identify different trails.")
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--epsilon', default=8. / 255,
                    help='perturbation')
parser.add_argument('--num_steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step_size', default=2. / 255,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=400, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_prefix', default='results/checkpoints/',
                    help='File folders where you want to store your checkpoints (default: results/checkpoints/)')
parser.add_argument('--csv_prefix', default='results/accuracy/',
                    help='File folders where you want to put your results (default: results/accruacy)')
parser.add_argument('--pretrained_model', default=None, help="The path of pretrained model")
parser.add_argument('--pretrained_epochs', default=0, type=int)
args = parser.parse_args()

# settings
model_prefix = args.model_prefix
csv_prefix = args.csv_prefix
model_name = f'CIFAR10_TRADES_{args.time_stamp}'
model_path = model_prefix + model_name + '.pth'
csv_path = csv_prefix + model_name + '.csv'

device = args.device

if device == "cuda:1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(args.seed)


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss_pgd
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss_pgd
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def train(args, model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss_pgd
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr:{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), scheduler.get_last_lr()))


def adv_eval(model, test_dl, attack_eps, attack_steps, attack_lr, device=None):
    total = 0
    robust_total = 0
    correct_total = 0
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    attacker = PGD(model=model,
                   eps=attack_eps,
                   alpha=attack_lr,
                   steps=attack_steps,
                   random_start=True,
                   average=True)

    for ii, (data, labels) in enumerate(test_dl):
        data = data.to(device)
        labels = labels.to(device)
        real_batch = data.shape[0]
        total += real_batch

        predictions = model(data)
        correct = torch.argmax(predictions, 1) == labels
        correct_total += correct.sum().cpu().item()

        with torch.enable_grad():
            if not torch.cuda.is_available() or device == "cpu":
                perturbed_data = attacker(data, labels)
            else:
                perturbed_data = attacker(data, labels).cuda(device=device)
        predictions = model(perturbed_data)
        robust = torch.argmax(predictions, 1) == labels
        robust_total += robust.sum().cpu().item()

    return correct_total, robust_total, total


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train_loader, _, test_loader, norm_layer = cifar10_dataloader(args.batch_size, val_ratio=0)
    model = WRN_16_8().to(device)
    model.normalize = norm_layer
    if args.pretrained_model:
        model.load(args.pretrained_model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(lr_steps * 0.5),
                                                                 int(lr_steps * 0.75)],
                                                     gamma=0.1)

    epoch_num_list = ['Epoch Number']
    clean_accuracy_list = ['Clean Accuracy']
    robust_accuracy_type_one_list = ['Robust Accuracy']

    best_acc = 0.0

    for epoch in range(args.pretrained_epochs + 1, args.epochs + 1):
        csv_row_list = []
        epoch_num_list.append(epoch)
        csv_row_list.append(epoch_num_list)

        # adjust learning rate for SGD
        # adjust_learning_rate(optimizer, epoch)

        # adversarial training
        time_start = time.time()
        model.train()
        train(args, model, device, train_loader, optimizer, epoch, scheduler)
        time_end = time.time()
        print(f"Time for a single epoch is {time_end - time_start}")

        model.eval()
        correct_total, robust_total, total = adv_eval(model, test_loader, attack_eps=8. / 255, attack_lr=2. / 255,
                                                      attack_steps=20, device=device)

        natural_acc = correct_total / total
        robust_acc = robust_total / total
        clean_accuracy_list.append(100. * natural_acc)
        robust_accuracy_type_one_list.append(100. * robust_acc)
        csv_row_list.append(clean_accuracy_list)
        csv_row_list.append(robust_accuracy_type_one_list)

        print(f'For the epoch {epoch} the accuracy is {natural_acc}')
        print(f'For the epoch {epoch} the robust accuracy is {robust_acc}')

        model.save(model_path)
        write_csv_rows(csv_path, csv_row_list)

        if robust_acc > best_acc:
            best_acc = robust_acc
            model.save(model_prefix + model_name + '_best.pth')


if __name__ == '__main__':
    main()
