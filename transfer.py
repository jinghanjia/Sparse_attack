import os
import sys
import logging
import argparse

from datasets import *
from model_zoo import *
from utils.loading_bar import Log
from utils.general_utils import write_csv_rows, setup_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bi-level Adversarial Transfer Learning")
    ########################## basic setting ##########################
    parser.add_argument('--device', default="cuda:0", help="The name of the device you want to use (default: None)")
    parser.add_argument('--time_stamp', default="00000000",
                        help="The time stamp that helps identify different trails.")
    parser.add_argument('--random_seed', default=37, type=int,
                        help='Random seed (default: 37)')
    parser.add_argument('--source_dataset', default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100", "SVHN", "STL10", "TINY_IMAGE_NET"])
    parser.add_argument('--target_dataset', default="CIFAR100",
                        choices=["CIFAR10", "CIFAR100", "SVHN", "STL10", "TINY_IMAGE_NET"])
    parser.add_argument('--model_prefix', default='results/checkpoints/',
                        help='File folders where you want to store your checkpoints (default: results/checkpoints/)')
    parser.add_argument('--csv_prefix', default='results/accuracy/',
                        help='File folders where you want to put your results (default: results/accruacy)')

    ########################## training setting ##########################
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=30, type=int, help="Total number of epochs.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--optimizer", default="SGD", choices=['SGD', 'Adam'])
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=0.1, type=float)

    ######################### transfer learning ###########################
    parser.add_argument('--fixed_feature', action="store_true")

    ########################## model setting ##########################
    parser.add_argument("--model_type", default="ResNet", choices=['ResNet', 'WideResNet'])
    parser.add_argument("--width_factor", default=0, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--depth", default=18, type=int, help="Number of layers.")
    parser.add_argument('--model_path', required=True, type=str)

    args = parser.parse_args()

    device = args.device

    if device == "cuda:1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    setup_seed(seed=args.random_seed)

    model_name = f"Transfer_{args.source_dataset}_{args.target_dataset}_{args.time_stamp}"
    model_path = args.model_prefix + model_name + '.pth'
    csv_path = args.csv_prefix + model_name + '.csv'

    ############################## Logger #################################
    log = Log(log_each=10)
    logging.basicConfig(filename=f'log/{model_name}.log', level=logging.INFO)
    logger = logging.getLogger("CIFAR10 BAT Training")

    ########################## dataset and model ##########################
    logger.info("Loading Dataset")
    if args.source_dataset == "CIFAR10":
        source_num_classes = 10
    elif args.source_dataset == "CIFAR100":
        source_num_classes = 100
    else:
        raise NotImplementedError("Invalid Dataset")
    if args.target_dataset == "CIFAR10":
        train_dl, val_dl, test_dl, norm_layer = cifar10_dataloader(batch_size=args.batch_size)
        target_num_classes = 10
    elif args.target_dataset == "CIFAR100":
        train_dl, val_dl, test_dl, norm_layer = cifar100_dataloader(batch_size=args.batch_size)
        target_num_classes = 100
    else:
        raise NotImplementedError("Invalid Dataset")

    logger.info("Preparing Model")
    if args.model_type == "WideResNet":
        if args.depth == 16:
            model = WRN_16_8(num_classes=source_num_classes, dropout=args.dropout)
        elif args.depth == 28:
            model = WRN_28_10(num_classes=source_num_classes, dropout=args.dropout)
        elif args.depth == 34:
            model = WRN_34_10(num_classes=source_num_classes, dropout=args.dropout)
        elif args.depth == 70:
            model = WRN_70_16(num_classes=source_num_classes, dropout=args.dropout)
        else:
            raise NotImplementedError("Unsupported WideResNet!")
    else:
        if args.depth == 18:
            model = ResNet18(num_classes=source_num_classes)
        elif args.depth == 34:
            model = ResNet34(num_classes=source_num_classes)
        else:
            model = ResNet50(num_classes=source_num_classes)
    model.normalize = norm_layer
    model = model.to(device)
    try:
        model.load(args.model_path, map_location=device)
    except IOError:
        logger.error("Loading model error")
        sys.exit()

    num_ftrs = model.linear.in_features
    model.linear = torch.nn.Linear(num_ftrs, target_num_classes).to(device)

    ########################## optimizer and scheduler ##########################
    if args.fixed_feature:
        optimizer = torch.optim.SGD(model.linear.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[
                                                         int(args.epochs * 0.5),
                                                         int(args.epochs * 0.75)],
                                                     gamma=0.1)
    loss_func = torch.nn.CrossEntropyLoss()

    ########################## other tools ######################################
    epoch_num_list = ['Epoch Number']
    clean_accuracy_list = ['Standard Accuracy']

    ########################## training #########################################
    for epoch in range(args.epochs):
        logger.info(f"\n========================Here Comes a New Epoch : {epoch}========================")
        model.train()
        csv_row_list = []
        epoch_num_list.append(epoch)
        csv_row_list.append(epoch_num_list)

        log.train(len_dataset=len(train_dl))

        for i, (data, labels) in enumerate(train_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]

            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_func(predictions, labels)

            correct = torch.argmax(predictions.data, 1) == labels
            log(model,
                loss=loss.cpu(),
                accuracy=correct.cpu(),
                learning_rate=scheduler.get_last_lr()[0],
                batch_size=real_batch)

            loss.backward()
            optimizer.step()

        scheduler.step()

        ########################## evaluation ###################################
        log.eval(len_dataset=len(val_dl))
        total = 0
        correct_total = 0

        for ii, (data, labels) in enumerate(test_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            total += real_batch

            predictions = model(data)
            correct = torch.argmax(predictions, 1) == labels
            correct_total += correct.sum().cpu().item()

            log(model=model,
                accuracy=correct.cpu(),
                robustness=correct.cpu(),
                batch_size=real_batch)

        natural_acc = correct_total / total
        logger.info(f'\nFor the epoch {epoch} the standard accuracy is {natural_acc}')
        clean_accuracy_list.append(100. * natural_acc)
        csv_row_list.append(clean_accuracy_list)

        model.save(model_path)
        write_csv_rows(csv_path, csv_row_list)

    log.flush()
    print('Training Over')
