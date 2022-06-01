import argparse

from tqdm import tqdm as tqdm

from attack import AutoAttack
from attack.regpgd import PGDRegAttack
from datasets import *
from model_zoo import *
from utils.general_utils import write_csv_rows

parser = argparse.ArgumentParser(description='Evaluation for Cifar10 Dataset')
parser.add_argument('--model-path', required=True)
parser.add_argument('--model-normalize', default=True, type=bool)
parser.add_argument("--batch-size", default=200, type=int,
                    help="Batch size used in the training and validation loop.")
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--dataset', default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
parser.add_argument('--model-type', default='BigConv', choices=['WideResNet', 'ResNet', 'PreActResNet', 'BigConv'])
parser.add_argument('--depth', default=18, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
parser.add_argument('--act-fn', default="relu", choices=["relu", "softplus", "swish"],
                    help="choose the activation function for your model")

parser.add_argument('--victim-model_path', default=None)
parser.add_argument('--victim-model_type', default='ResNet', choices=['WideResNet', 'ResNet'])
parser.add_argument('--victim-depth', default=50, type=int, help="Number of layers of victim_model.")
parser.add_argument("--victim-dropout", default=0.1, type=float, help="Dropout rate.")

parser.add_argument('--attack-step', default=20, type=int,
                    help='attack steps for training (default: 20)')
parser.add_argument('--attack-lr', '--attack-learning-rate', default=2. / 255, type=float,
                    help='initial attack learning rate (default: 2./255)')
parser.add_argument('--attack-method', default='PGD', choices=['PGD', 'AutoAttack'])
parser.add_argument('--attack-constraint', default='linf', choices=['linf', 'l2'])
parser.add_argument('--pgd-no-sign', default=False, action="store_true")
parser.add_argument('--val-dl', default=False, action="store_true")
args = parser.parse_args()


def evaluation(model_path):
    device = args.device
    attack_method = args.attack_method
    attack_step = args.attack_step
    attack_lr = args.attack_lr
    dataset = args.dataset
    attack_constraint = args.attack_constraint

    print(model_path)

    ########################## dataset and model ##########################
    if args.dataset == "CIFAR10":
        train_dl, val_dl, test_dl, norm_layer = cifar10_dataloader(batch_size=args.batch_size)
        num_classes = 10
        conv1_size = 3
    elif args.dataset == "CIFAR100":
        train_dl, val_dl, test_dl, norm_layer = cifar100_dataloader(batch_size=args.batch_size)
        num_classes = 100
        conv1_size = 3
    elif args.dataset == "TINY_IMAGE_NET":
        train_dl, val_dl, test_dl, norm_layer = tiny_imagenet_dataloader(batch_size=args.batch_size)
        num_classes = 200
        conv1_size = 3
    elif args.dataset == "SVHN":
        train_dl, val_dl, test_dl, norm_layer = svhn_dataloader(batch_size=args.batch_size)
        num_classes = 10
        conv1_size = 3
    elif args.dataset == "STL10":
        train_dl, val_dl, test_dl, norm_layer = stl10_dataloader(batch_size=args.batch_size)
        num_classes = 10
        conv1_size = 3
    else:
        raise NotImplementedError("Invalid Dataset")

    if args.val_dl:
        eval_dl = val_dl
    else:
        eval_dl = test_dl

    if args.act_fn == "relu":
        activation_fn = nn.ReLU
    elif args.act_fn == "softplus":
        activation_fn = nn.Softplus
    elif args.act_fn == "swish":
        activation_fn = Swish
    else:
        raise NotImplementedError("Unsupported activation function!")

    if args.model_type == "BigConv":
        model = BigConv(num_classes=num_classes)
    elif args.model_type == "WideResNet":
        if args.depth == 16:
            model = WRN_16_8(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                             activation_fn=activation_fn)
        elif args.depth == 28:
            model = WRN_28_10(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                              activation_fn=activation_fn)
        elif args.depth == 34:
            model = WRN_34_10(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                              activation_fn=activation_fn)
        elif args.depth == 70:
            model = WRN_70_16(num_classes=num_classes, conv1_size=conv1_size, dropout=args.dropout,
                              activation_fn=activation_fn)
        else:
            raise NotImplementedError("Unsupported WideResNet!")
    elif args.model_type == "PreActResNet":
        if args.depth == 18:
            model = PreActResNet18(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        elif args.depth == 34:
            model = PreActResNet34(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        else:
            model = PreActResNet50(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
    elif args.model_type == "ResNet":
        if args.depth == 18:
            model = ResNet18(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        elif args.depth == 34:
            model = ResNet34(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
        else:
            model = ResNet50(num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
    else:
        raise NotImplementedError("Unsupported Model Type!")
    model.normalize = norm_layer

    print("=> loading checkpoint '{}'".format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location=device)
    best_prec1 = checkpoint["best_prec1"]
    print("=> Best Acc in record is '{}'".format(best_prec1))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model_name = model_path.split('/')[-1].split('.')[0]

    if args.victim_model_path:
        if args.victim_model_type == "WideResNet":
            if args.victim_depth == 16:
                victim_model = WRN_16_8(dropout=args.dropout, num_classes=num_classes)
            elif args.victim_depth == 28:
                victim_model = WRN_28_10(dropout=args.dropout, num_classes=num_classes)
            elif args.victim_depth == 34:
                victim_model = WRN_34_10(dropout=args.dropout, num_classes=num_classes)
            elif args.victim_depth == 70:
                victim_model = WRN_70_16(dropout=args.dropout, num_classes=num_classes)
            else:
                raise NameError("Unsupported WideResNet!")
        else:
            if args.victim_depth == 18:
                victim_model = ResNet18(activation_fn=nn.ReLU(), num_classes=num_classes)
            elif args.depth == 34:
                victim_model = ResNet34(activation_fn=nn.ReLU(), num_classes=num_classes)
            else:
                victim_model = ResNet50(activation_fn=nn.ReLU(), num_classes=num_classes)

        if args.model_normalize:
            victim_model.normalize = norm_layer

        victim_model.load_state_dict(torch.load(args.victim_model_path, map_location=torch.device(device)))
        victim_model = victim_model.to(device)
        victim_model_name = args.victim_model_path.split('/')[-1].split('.')[0]
    else:
        victim_model = model

    if attack_method == 'PGD':
        if attack_constraint == 'linf':
            epsilon = [2./255, 8. / 255, 10. / 255, 12. / 255, 15. / 255]
        else:
            epsilon = [0.5, 1.0, 1.5, 2.0]
    elif attack_method == 'AutoAttack':
        if attack_constraint == 'linf':
            epsilon = [8. / 255, 10. / 255, 12. / 255, 15. / 255]
        else:
            epsilon = [128. / 255]
    else:
        raise (NameError, "Unsupported Attack!")

    if not args.victim_model_path:
        file_path = f'results/accuracy/evaluation/Evaluation_{dataset}_{attack_method}_attack_{model_name}.csv'
    else:
        file_path = f'results/accuracy/evaluation/Evaluation_Transfer_{dataset}_{attack_method}_attack_{model_name}_{victim_model_name}.csv'

    result_list = []
    csv_row_list = [epsilon]

    model.eval()
    victim_model.eval()
    correct = 0
    total = 0
    for ii, (images, labels) in tqdm(enumerate(eval_dl)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Natural accuracy: %.2f %%' % (100. * (correct / total).cpu().item()))

    for eps in epsilon:
        print(attack_method, eps, attack_step, attack_lr)

        if attack_method == 'PGD':
            if attack_constraint == 'linf':
                attacker = PGDRegAttack(
                    victim_model, loss_fn=torch.nn.CrossEntropyLoss(), eps=eps, steps=attack_step,
                    eps_lr=attack_lr, ord=np.inf,
                    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, sign=not args.pgd_no_sign
                )
            else:
                attacker = PGDRegAttack(
                    victim_model, loss_fn=torch.nn.CrossEntropyLoss(), eps=eps, steps=attack_step,
                    eps_lr=attack_lr, ord=2,
                    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, sign=not args.pgd_no_sign
                )

            attack_total = 0
            attack_correct = 0
            for ii, (data, label) in tqdm(enumerate(eval_dl)):
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                perturbed_data = attacker.perturb(data, label)

                score = model(perturbed_data)
                _, predicted = torch.max(score, 1)
                attack_total += label.cpu().size(0)
                attack_correct += (predicted == label).sum()
        elif attack_method == "AutoAttack":
            if attack_constraint == 'linf':
                attacker = AutoAttack(victim_model, norm='Linf', eps=eps)
            else:
                attacker = AutoAttack(victim_model, norm='L2', eps=eps)
            attack_total = 0
            attack_correct = 0
            for ii, (data, label) in tqdm(enumerate(eval_dl)):
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                if device != 'cpu':
                    perturbed_data = attacker(data, label).cuda(device=device)
                else:
                    perturbed_data = attacker(data, label)

                score = model(perturbed_data)
                _, predicted = torch.max(score, 1)
                attack_total += label.cpu().size(0)
                attack_correct += (predicted == label).sum()
        else:
            raise NameError("Unsupported Attack Method!")

        print(f'The robust accuracy against epsilon {eps} is {attack_correct / attack_total * 100}')
        result_list.append(attack_correct.cpu().item() / 100.)

    csv_row_list.append(result_list)
    write_csv_rows(file_path, csv_row_list)


if __name__ == '__main__':
    model_path_list = list(args.model_path.split(","))
    for model_path in model_path_list:
        evaluation(model_path)
