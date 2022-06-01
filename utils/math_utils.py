import torch
import numpy as np
import torch.nn.functional as F


def hessian_vec_prod(func, x, vec, grads=None):
    """
    求解 \nabal_{xx} f(x) * vecs
    :param func: objective function
    :param x: variable，如果传了grads，请一定保证grads确实是x的grads，x传进来的时候千万不要detach，否则grads就和x没有关系了，二阶导还要
    用到二者之间的梯度关系。
    :param vec: 向量，和内循环变量y同size
    :param grads: 每次预设的一阶梯度，减少计算量。注意，这里传进来的grads应保持retain_graph=True！！！因为还要往后求二阶导。
    :return: Hessian矩阵和向量的乘积
    """
    x.requires_grad = True
    # 这里grads对每次可以重复利用
    if grads is None:
        inner_res = func(x)
        grads = torch.autograd.grad(inner_res, x, create_graph=True, retain_graph=True)[0].view(-1)
    prod = grads.dot(vec.double())
    so_grad = torch.autograd.grad(prod, x, retain_graph=True)[0].view(-1)
    return so_grad


def hessian_vec_prod_complex(func, x, vecs):
    """
    计算Hessian和向量的乘积，这里把Hessian硬算出来后再和向量相乘，作为以上优化算法的校验算法。
    :param func: 目标函数
    :param x: Hessian求解时针对的变量
    :param vecs: 向量
    :return: Hessian矩阵和向量的乘积
    """

    def get_second_order_grad(_func, x):
        """
        硬算hessian矩阵
        :param _func: 目标函数
        :param x: 变量
        :return: Hessian矩阵
        """
        x.requires_grad = True

        inner_res = _func(x)
        grads = torch.autograd.grad(inner_res, x, create_graph=True, retain_graph=True)

        grads2 = torch.tensor([])
        for anygrad in grads[0]:
            grads2 = torch.cat((grads2, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
        return grads2.view(x.size()[0], -1)

    x.requires_grad = True
    grads_so = get_second_order_grad(func, x)
    prod = grads_so @ vecs.double()
    return prod


def hessian_vec_prod_diff(inner_func, x, y, vecs, r=1e-7):
    """
    在bi-level optimization中，用一阶差分估计二阶梯度矩阵Cross-Term 和 vector的内积
    \nabla_{xy}^{2}g(vec, x^*(vec))
    \omega^* = [(\nabla_x g(vec, x^* + \delta*r) - \nabla_x g(vec, x^* - \delta*r))/(2\delta)]
    一阶导函数上 二阶偏导对象的差分函数值作为 二阶偏导的估计。
    :param inner_func: f(y, x)，其中对y一阶偏导，对x二阶偏导；y是内循环参数，x是外循环参数。
    :param x: 外循环参数
    :param y: 内循环参数
    :param vecs: 作内积的向量
    :param r: 估计用小量
    :return:
    """
    x.requires_grad = True
    y.requires_grad = True

    # 一阶对外循环变量的偏导

    def get_grad(func, outer_var, inner_var):
        inner_res = func(outer_var, inner_var)
        grads_dy = torch.autograd.grad(inner_res
                                       , outer_var
                                       , create_graph=True
                                       , retain_graph=True
                                       )[0].view(-1)
        return grads_dy

    # 差分对象应该为内循环变量
    def add(inner_var, vec, omega):
        return inner_var + omega * vec

    y_right = add(y, vecs, r)
    y_left = add(y, vecs, -r)

    g_lefts = get_grad(inner_func, x, y_left)
    g_rights = get_grad(inner_func, x, y_right)

    return (g_rights - g_lefts) / (2 * r)


def hessian_cross_calculation(inner_func, x, y):
    """
    硬算cross二阶导数，作为以上算法的校验算法，先对y求导，再对x求导。
    :param inner_func:
    :param x:
    :param y:
    :return:
    """
    x.requires_grad = True
    y.requires_grad = True
    res = inner_func(x, y)
    grads = torch.autograd.grad(res, y, create_graph=True, retain_graph=True)

    grads2 = torch.tensor([])
    for anygrad in grads[0]:
        grads2 = torch.cat((grads2, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
    return grads2.view(x.size()[0], -1)


def hessian_gaussian_estimation(func, x, mu=1e-2, sample=100, lam=1e-3):
    """
    randomized smoothing 对 hessian 矩阵进行估计
    :param func: 目标函数
    :param x: 目标变量
    :param lam: 保证Hessian正定的补偿
    :param mu: 差分间距
    :param sample: 采样次数，与x纬度有关
    :return: 估计的Hessian矩阵
    """
    x = x.view(-1)
    d = x.size()[0]
    hessian = torch.zeros((x.size()[0], x.size()[0]))

    for i in range(sample):
        u = torch.randn(d)
        res = (func(x + mu * u) + func(x - mu * u) - 2 * func(x)) / (2 * mu ** 2)
        u = u.unsqueeze(-1)
        hessian = hessian + res * (u.matmul(u.t()) - torch.eye(d))

    hessian /= sample

    return hessian + lam * torch.eye(d)


def batch_dot(v1, v2):
    """
    求解两个batch向量对应的内积
    :param v1: shape[batch_size, n]
    :param v2: shape[batch_size, n]
    :return: [shape_size, 1]
    """
    assert v1.shape == v2.shape
    batch_size = v1.shape[0]
    return v1.view(batch_size, 1, -1).bmm(v2.view(batch_size, -1, 1)).squeeze(-1)


def batch_cg_solver(fx, b, iter_num=1000, residual_tol=1e-7, x_init=None, verbose=False):
    """
    与cg_solver不同的是，该函数成批求解A^{-1}b，其中A和b都多了一个batch_size维度，需保证A中每个方阵都是正定的。
    :param fx: 但变量函数，输入b，返回A.bmm(b)计算结果
    :param b: 一个batch的列向量
    :param iter_num: 最高迭代次数
    :param residual_tol: 一个batch的平均拟合精度
    :param x_init: 初始值，如不指定则随机
    :return: A^{-1}b的估计结果。
    """
    x = torch.zeros(b.shape).float().to(b) if x_init is None else x_init

    r = b - fx(x)
    p = r.clone()

    for i in range(iter_num):
        rdotr = batch_dot(r, r)
        Ap = fx(p)
        alpha = rdotr / (batch_dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = batch_dot(r, r)
        if verbose:
            print(f"BG iteration {i}, {newrdotr.mean()}")

        if newrdotr.mean() < residual_tol:
            if verbose:
                print(f"Early CG termination at iteration {i}")
            break

        beta = newrdotr / (rdotr + 1e-12)
        p = r + beta * p

    return x


def cg_solver(fx, b, iter_num=10, residual_tol=1e-10, x_init=None):
    """
    求解A^{-1}b或者解线性方程Ax=b，要求A是对称正定阵。原理是最小化 f(x) = 1/2 x^T A x - x^T b
    :param fx: 这是一个单变量函数，传入向量x，计算出Ax，A定义在fx内部。A应是对称正定矩阵，否则结果可能会发散。
    :param b: A^{-1}b中的b
    :param iter_num: 最大迭代次数
    :param residual_tol: cg迭代终止条件，当本次迭代residual小于次预设值或达到迭代最大次数退出迭代。
    :param x_init: 初始值，不预定则随机
    :return:
    """

    # 初始化
    x = torch.zeros(b.shape[0]).double() if x_init is None else x_init
    if b.dtype == torch.float16:
        x = x.half()
    r = (b - fx(x))
    p = r.clone()

    for i in range(iter_num):
        # 用于结果展示，但是由于这里还要再调用f_AX,相当于重新做Hessian-vec-production计算，因此就注释掉了。
        # obj_fn = 0.5 * vec.dot(fx(vec)) - 0.5 * b.dot(vec)
        # norm_x = torch.norm(vec) if type(vec) == torch.Tensor else np.linalg.norm(vec)
        # fmtstr = "%10i %10.3g %10.3g %10.3g"
        # print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = fx(p)
        alpha = rdotr / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr / rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print(f"Early CG termination at iteration {i}")
            break
    return x


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    return ((x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)).mean()


def smooth_crossentropy(pred, gold, smoothing=0.3):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()


if __name__ == "__main__":
    AA = []
    for i in range(4):
        A = torch.rand((5, 5))
        AA.append(A.matmul(A.t()).unsqueeze(0))

    AA = torch.cat(AA, dim=0)

    b = torch.rand((4, 5))

    def __for_batch_cg(x):
        return AA.bmm(x.unsqueeze(-1)).squeeze(-1)
    res = batch_cg_solver(__for_batch_cg, b)

    res_2 = torch.inverse(AA).bmm(b.unsqueeze(-1)).squeeze(-1)
    print(res - res_2)


