import argparse
import logging
import os

from datasets import *
from model_zoo import *
from trainer import Trainer
from utils.general_utils import setup_seed, mkdir_recursive, save_checkpoint
from utils.loading_bar import Log
from utils.math_utils import smooth_crossentropy, dlr_loss
from utils.step_lr import *

from pylab import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bi-level Adversarial Training")

    ########################## basic setting ##########################
    parser.add_argument('--device', default="cuda:0", help="The name of the device you want to use (default: None)")
    parser.add_argument('--time-stamp', default="debug",
                        help="The time stamp that helps identify different trails.")
    parser.add_argument('--dataset', default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100", "SVHN", "STL10", "TINY_IMAGE_NET"])
    parser.add_argument('--dataset-val-ratio', default=0.0, type=float)
    parser.add_argument('--mode', default='at', type=str, choices=["at"],
                        help="at : pgd-at, bat: bi-level at")

    parser.add_argument('--model-prefix', default='results/checkpoints/',
                        help='File folders where you want to store your checkpoints (default: results/checkpoints/)')
    parser.add_argument('--csv-prefix', default='results/accuracy/',
                        help='File folders where you want to put your results (default: results/accruacy)')
    parser.add_argument('--random-seed', default=37, type=int,
                        help='Random seed (default: 37)')
    parser.add_argument('--resume', default=None, help="The path of resumed model")
    parser.add_argument('--checkpoint-moe', '--p', default=None, help="pretrained model for moe")

    ########################## training setting ##########################
    parser.add_argument("--batch-size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")

    parser.add_argument("--optimizer", default="SGD", choices=['SGD', 'Adam'])
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight-decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--scheduler-decay-rate", default=0.1, type=float)
    parser.add_argument("--cyclic-milestone", default=10, type=int)

    parser.add_argument('--lr-scheduler', default='multistep',
                        choices=['cyclic', 'multistep', 'multilinear', 'cyclic_lin_qua', 'cyclic_qua_qua', 'cosine',
                                 'cosine_wr'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)

    parser.add_argument('--train-loss', default="ce", choices=["ce", "sce", "n_dlr"],
                        help="ce for cross entropy, sce for label-smoothed ce, n_dlr for negative dlr loss")

    parser.add_argument('--act-fn', default="relu", choices=["relu", "softplus", "swish"],
                        help="choose the activation function for your model")

    ########################## model setting ##########################
    parser.add_argument("--model-type", default="BigConv",
                        choices=['ResNet', 'PreActResNet', 'WideResNet', 'BigConv', 'BigConvMoE'])
    parser.add_argument("--width-factor", default=0, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--depth", default=18, type=int, help="Number of layers.")

    ########################## attack setting ##########################
    parser.add_argument('--attack-step', default=3, type=int,
                        help='attack steps for training (default: 3)')
    parser.add_argument('--attack-step-test', default=20, type=int,
                        help='attack steps for evaluation (default: 10)')
    parser.add_argument('--attack-eps', default=2., type=float,
                        help='attack constraint for training (default: 8/255)')
    parser.add_argument('--attack-eps-test', default=2., type=float,
                        help='attack constraint for evaluation (default: 8/255)')
    parser.add_argument('--attack-lr', default=2.5, type=float,
                        help='initial attack learning rate (default: 2.5/255)')
    parser.add_argument('--attack-lr-test', default=2.5, type=float,
                        help='initial attack learning rate for evaluation (default: 2.5/255)')
    parser.add_argument('--constraint-type', default='linf',
                        choices=["linf", "l2"])

    ############################### other options ###################################
    parser.add_argument("--pgd-no-sign", default=False, action="store_true",
                        help="Do you want to use no-sign versioned pgd")

    args = parser.parse_args()
    device = args.device

    print(args)

    if device == "cuda:1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    setup_seed(seed=args.random_seed)
    training_type = args.mode.upper()
    model_name = f"{args.dataset}_{training_type}_Epoch{args.epochs}_Scheduler{args.lr_scheduler}_Eps{args.attack_eps}_Arck{args.model_type}_Attacklr{args.attack_lr}_{args.time_stamp}"

    mkdir_recursive(args.csv_prefix)
    model_path = os.path.join(args.model_prefix, model_name)
    mkdir_recursive(model_path)
    csv_path = os.path.join(args.csv_prefix, model_name + '.csv')
    figure_name = os.path.join(args.csv_prefix, model_name + '.png')

    ########################## dataset and model ##########################
    if args.dataset == "CIFAR10":
        train_dl, val_dl, test_dl, norm_layer = cifar10_dataloader(batch_size=args.batch_size,
                                                                   val_ratio=args.dataset_val_ratio)
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
    elif args.model_type == "BigConvMoE":
        model = MoEBigConv(num_classes=num_classes)
        # assert args.checkpoint_moe is not None
        if args.checkpoint_moe is not None:
            if os.path.isfile(args.checkpoint_moe):
                print("=> loading checkpoint '{}'".format(args.checkpoint_moe))
                checkpoint = torch.load(args.checkpoint_moe, map_location=device)
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded pretrained model {}".format(args.checkpoint_moe))
            else:
                print("=> no checkpoint found at '{}'".format(args.BigConvMoE))
                raise ValueError("=> no checkpoint found at '{}'".format(args.BigConvMoE))

        model.duplicate()
        model.to_device(device)
        if args.checkpoint_moe is not None:
            model.fix_model()
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
    model = model.to(device)

    ########################## optimizer and scheduler ##########################
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    lr_steps = args.epochs * len(train_dl)
    if args.lr_scheduler == "cyclic":
        milestone_epoch_num = args.cyclic_milestone
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.lr_min,
                                                      max_lr=args.lr_max,
                                                      step_size_up=int(milestone_epoch_num * len(train_dl)),
                                                      step_size_down=int(
                                                          (args.epochs - milestone_epoch_num) * len(train_dl)))
    elif args.lr_scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(len(train_dl) * i * args.epochs) for i in [0.5, 0.75]],
                                                         gamma=args.scheduler_decay_rate)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_steps)
    elif args.lr_scheduler == "cosine_wr":
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=int(0.5 * lr_steps),
                                                  cycle_mult=1.0,
                                                  max_lr=args.lr_max,
                                                  min_lr=0.001,
                                                  warmup_steps=int(lr_steps * 0.1),
                                                  gamma=0.5)
    elif args.lr_scheduler == "multilinear":
        x1 = 5 * len(train_dl)
        x2 = 10 * len(train_dl)
        x3 = args.epochs * len(train_dl)
        scheduler = MultiLinearScheduler(optimizer, lr_max=0.2, lr_middle=0.1, lr_min=0, x1=x1, x2=x2, x3=x3)
    elif args.lr_scheduler == "cyclic_lin_qua":
        mile_stone_epoch_num = 5
        scheduler = CyclicLinQuaStepLR(optimizer,
                                       lr_max=args.lr_max,
                                       step_size_up=mile_stone_epoch_num * len(train_dl),
                                       step_size_down=args.epochs * len(train_dl))
    elif args.lr_scheduler == "cyclic_lin_bi_qua":
        mile_stone_epoch_num = 5
        scheduler = CyclicLinBiQuaStepLR(optimizer,
                                         lr_max=args.lr_max,
                                         step_size_up=mile_stone_epoch_num * len(train_dl),
                                         step_size_down=args.epochs * len(train_dl))
    elif args.lr_scheduler == "cyclic_qua_qua":
        mile_stone_epoch_num = 5
        scheduler = CyclicQuadraticStepLR(optimizer,
                                          lr_max=args.lr_max,
                                          step_size_up=mile_stone_epoch_num * len(train_dl),
                                          step_size_down=args.epochs * len(train_dl))
    else:
        raise NotImplementedError("Unsupported Scheduler!")

    if args.train_loss == "sce":
        loss_func = smooth_crossentropy
    elif args.train_loss == "ce":
        loss_func = torch.nn.CrossEntropyLoss()
    elif args.train_loss == "n_dlr":
        def n_dlr(predictions, labels):
            return -dlr_loss(predictions, labels)


        loss_func = n_dlr
    else:
        raise NotImplementedError("Unsupported Loss Function!")

    ########################## resume ##########################
    start_epoch = 0
    if args.resume is not None:
        # model.load(args.pretrained_model, map_location=device)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    ############################## Logger #################################
    os.makedirs("log", exist_ok=True)
    log = Log(log_each=2, initial_epoch=start_epoch)
    logging.basicConfig(filename=f'log/{model_name}.log', level=logging.INFO)
    logger = logging.getLogger("CIFAR10 BAT Training")

    ############################ BAT Trainer ###################################
    trainer = Trainer(args=args,
                      attack_steps=args.attack_step,
                      attack_eps=args.attack_eps / 255,
                      attack_lr=args.attack_lr / 255,
                      log=log)

    epoch_num_list = ['Epoch Number']
    standard_test_accuracy = ['Standard Test Accuracy']
    robust_test_accuracy = ['Robust Test Accuracy']
    standard_train_accuracy = ['Standard Train Accuracy']
    robust_train_accuracy = ['Robust Train Accuracy']

    best_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n========================Here Comes a New Epoch : {epoch}========================")
        model.train()
        csv_row_list = []
        epoch_num_list.append(epoch)
        csv_row_list.append(epoch_num_list)

        log.train(len_dataset=len(train_dl))

        model, standard_train_acc, robust_train_acc = trainer.train(model=model,
                                                                    train_dl=train_dl,
                                                                    opt=optimizer,
                                                                    loss_func=loss_func,
                                                                    scheduler=scheduler,
                                                                    device=device)

        model.eval()
        log.eval(len_dataset=len(test_dl))

        correct_total, robust_total, total = trainer.eval(model=model,
                                                          test_dl=test_dl,
                                                          attack_eps=args.attack_eps_test / 255,
                                                          attack_steps=args.attack_step_test,
                                                          attack_lr=args.attack_lr_test / 255,
                                                          device=device)

        standard_test_acc = correct_total / total
        robust_test_acc = robust_total / total

        logger.info(f'\nFor the epoch {epoch} the standard train accuracy is {standard_train_acc}')
        logger.info(f'\nFor the epoch {epoch} the robust train accuracy is {robust_train_acc}')
        logger.info(f'\nFor the epoch {epoch} the standard test accuracy is {standard_test_acc}')
        logger.info(f'\nFor the epoch {epoch} the robust test accuracy is {robust_test_acc}')
        standard_test_accuracy.append(100. * standard_test_acc)
        robust_test_accuracy.append(100. * robust_test_acc)
        standard_train_accuracy.append(100. * standard_train_acc)
        robust_train_accuracy.append(100. * robust_train_acc)

        if robust_test_acc > best_acc:
            best_acc = robust_test_acc
            is_best = True
        else:
            is_best = False

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": best_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            },
            is_best,
            result_dir=model_path,
        )

    standard_train_acc_plot = np.array(standard_train_accuracy[1:])
    standard_test_acc_plot = np.array(standard_test_accuracy[1:])
    robust_train_acc_plot = np.array(robust_train_accuracy[1:])
    robust_test_acc_plot = np.array(robust_test_accuracy[1:])
    epoch = np.arange(len(robust_test_acc_plot))

    width = 14
    height = 12
    plt.figure(figsize=(width, height))
    fontsize = 46
    x_label = "Epoch Number"
    y_label = "Accuracy"
    marker_size = 30
    line_width = 3

    import seaborn as sns

    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    plt.plot(epoch, standard_train_acc_plot, label="Standard Train Accuracy", color="green", linewidth=line_width,
             linestyle="--")
    plt.plot(epoch, robust_train_acc_plot, label="Robust Train Accuracy", color="orange", linewidth=line_width,
             linestyle="--")
    plt.plot(epoch, standard_test_acc_plot, label="Standard Test Accuracy", color="green", linewidth=line_width,
             linestyle="-")
    plt.plot(epoch, robust_test_acc_plot, label="Robust Test Accuracy", color="orange", linewidth=line_width,
             linestyle="-")

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=3, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(figure_name)

    log.flush()
    print('Training Over')
