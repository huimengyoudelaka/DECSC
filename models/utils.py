import torch
from torch.nn import functional


def conv_power_method(D, image_size, num_iters=100, stride=1, bands=31):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [int(((image_size[0] - D.shape[-2])/stride)+1), int(((image_size[1] - D.shape[-1])/stride)+1)]
    x = torch.randn(bands, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = functional.conv_transpose2d(x, D, stride=stride)
        x = functional.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))


def calc_pad_sizes(I: torch.Tensor, kernel_size: int, stride: int):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad

def list2vec(z1_list):
    """Convert list of tensors to a vector"""
    bsz = z1_list[0].size(0)
    return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)


def vec2list(z1, cutoffs):
    """Convert a vector back to a list, via the cutoffs specified"""
    z1_list = []
    start_idx, end_idx = 0, torch.cumprod(torch.tensor(cutoffs[0][1:]),dim = 0)[-1]
    for i in range(len(cutoffs)):
        z1_list.append(z1[:,start_idx:end_idx].view(*cutoffs[i]))
        if i < len(cutoffs)-1:
            start_idx = end_idx.clone()
            end_idx += torch.cumprod(torch.tensor(cutoffs[i+1][1:]),dim = 0)[-1]
    return z1_list

class Meter(object):
    """Computes and stores the min, max, avg, and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)


class SplittingMethodStats(object):
    def __init__(self):
        self.fwd_iters = Meter()
        self.bkwd_iters = Meter()
        self.fwd_time = Meter()
        self.bkwd_time = Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_iters.reset()
        self.bkwd_time.reset()

    def report(self):
        print('Fwd iters: {:.2f}\tFwd Time: {:.4f}\tBkwd Iters: {:.2f}\tBkwd Time: {:.4f}\n'.format(
                self.fwd_iters.avg, self.fwd_time.avg,
                self.bkwd_iters.avg, self.bkwd_time.avg))

def compute_eigval(lin_module, method="power", compute_smallest=False, largest=None):
    with torch.no_grad():
        if method == "direct":
            W = lin_module.W.weight
            eigvals = torch.symeig(W + W.T)[0]
            return eigvals.detach().cpu().numpy()[-1] / 2

        elif method == "power":
            z0 = tuple(torch.randn(*shp).to(lin_module.U.weight.device) for shp in lin_module.z_shape(1))
            lam = power_iteration(lin_module, z0, 100,
                                  compute_smallest=compute_smallest,
                                  largest=largest)
            return lam

def power_iteration(linear_module, z, T,  compute_smallest=False, largest=None):
    n = len(z)
    for i in range(T):
        za = linear_module.multiply(*z)
        zb = linear_module.multiply_transpose(*z)
        if compute_smallest:
            zn = tuple(-2*largest*a + 0.5*b + 0.5*c for a,b,c in zip(z, za, zb))
        else:
            zn = tuple(0.5*a + 0.5*b for a,b in zip(za, zb))
        x = sum((zn[i]*z[i]).sum().item() for i in range(n))
        y = sum((z[i]*z[i]).sum().item() for i in range(n))
        lam = x/y
        z = tuple(zn[i]/np.sqrt(y) for i in range(n))
    return lam +2*largest if compute_smallest else lam

def get_splitting_stats(dataLoader, model):
    model = cuda(model)
    model.train()
    model.mon.save_abs_err = True
    for batch in dataLoader:
        data, target = cuda(batch[0]), cuda(batch[1])
        model(data)
        return model.mon.errs