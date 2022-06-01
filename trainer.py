from attack.pgd_attack import PgdAttack
from utils.context import ctx_noparamgrad
from utils.math_utils import *


class Trainer:
    def __init__(self, args, attack_steps, attack_eps, attack_lr, log):
        self.args = args
        self.steps = attack_steps
        self.eps = attack_eps
        self.attack_lr = attack_lr
        self.constraint_type = np.inf if args.constraint_type == "linf" else 2
        self.log = log
        self.mode = args.mode
        self.sign_pgd = not args.pgd_no_sign

    def train(self, model, train_dl, opt, loss_func, scheduler=None, device=None):
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        adversary_train = PgdAttack(
            model, loss_fn=torch.nn.CrossEntropyLoss(), eps=self.eps, steps=self.steps,
            eps_lr=self.attack_lr, ord=self.constraint_type,
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False, sign=self.sign_pgd
        )

        model.train()

        clean_total = 0
        robust_total = 0
        total = 0

        for i, (data, labels) in enumerate(train_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            total += real_batch

            clean_output = model(data)
            clean_correct = (torch.argmax(clean_output.data, 1) == labels).sum()
            clean_total += clean_correct.detach().cpu().item()

            if self.steps == 0:
                delta_star = torch.zeros_like(data).to(data)
            else:
                with ctx_noparamgrad(model):
                    delta_star = adversary_train.perturb(data, labels) - data

            delta_star.requires_grad = False

            if self.mode == "at":
                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                loss = loss_func(predictions, labels)
                loss.backward()
                opt.step()

                correct = torch.argmax(predictions.data, 1) == labels
                robust_total += correct.sum().detach().cpu().item()

            else:
                raise NotImplementedError()
            if self.log:
                with torch.no_grad():
                    self.log(model,
                             loss=loss.cpu(),
                             accuracy=correct.cpu(),
                             learning_rate=scheduler.get_last_lr()[0],
                             batch_size=real_batch)
            if scheduler:
                scheduler.step()
        return model, (clean_total / total), (robust_total / total)

    def _attack_loss(self, predictions, labels):
        return -torch.nn.CrossEntropyLoss(reduction='sum')(predictions, labels)

    def eval(self, model, test_dl, attack_eps, attack_steps, attack_lr, device=None):
        total = 0
        robust_total = 0
        correct_total = 0
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        adversary_eval = PgdAttack(
            model, loss_fn=torch.nn.CrossEntropyLoss(), eps=attack_eps, steps=attack_steps,
            eps_lr=attack_lr, ord=self.constraint_type, regular=0.0,
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

        for ii, (data, labels) in enumerate(test_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            total += real_batch

            with ctx_noparamgrad(model):
                perturbed_data = adversary_eval.perturb(data, labels)

            if attack_steps == 0:
                perturbed_data = data

            predictions = model(data)
            correct = torch.argmax(predictions, 1) == labels
            correct_total += correct.sum().cpu().item()

            predictions = model(perturbed_data)
            robust = torch.argmax(predictions, 1) == labels
            robust_total += robust.sum().cpu().item()
            if self.log:
                self.log(model=model,
                         accuracy=correct.cpu(),
                         robustness=robust.cpu(),
                         batch_size=real_batch)

        return correct_total, robust_total, total


def norm(x):
    return torch.sqrt(torch.sum(x * x))
