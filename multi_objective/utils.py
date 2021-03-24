import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections

from datetime import datetime

from loaders import adult_loader, compas_loader, multi_mnist_loader, celeba_loader, credit_loader
from models import FullyConnected, MultiLeNet, EfficientNet, ResNet


def dataset_from_name(dataset, **kwargs):
    if dataset == 'adult':
        return adult_loader.ADULT(**kwargs)
    elif dataset == 'credit':
        return credit_loader.Credit(**kwargs)
    elif dataset == 'compass':
        return compas_loader.Compas(**kwargs)
    elif dataset == 'multi_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='mnist', **kwargs)
    elif dataset == 'multi_fashion':
        return multi_mnist_loader.MultiMNIST(dataset='fashion', **kwargs)
    elif dataset == 'multi_fashion_mnist':
        return multi_mnist_loader.MultiMNIST(dataset='fashion_and_mnist', **kwargs)
    elif dataset == 'celeba':
        return celeba_loader.CelebA(**kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def model_from_dataset(dataset, **kwargs):
    if dataset == 'adult':
        return FullyConnected(**kwargs)
    elif dataset == 'credit':
        return FullyConnected(**kwargs)
    elif dataset == 'compass':
        return FullyConnected(**kwargs)
    elif dataset == 'multi_mnist' or dataset == 'multi_fashion_mnist' or dataset == 'multi_fashion':
        return MultiLeNet(**kwargs)
    elif dataset == 'celeba':
        if 'efficientnet' in kwargs['model_name']:
            return EfficientNet.from_pretrained(**kwargs)
        elif kwargs['model_name'] == 'resnet18':
            return ResNet.from_name(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset))


def circle_points(n, min_angle=0.1, max_angle=np.pi / 2 - 0.1, dim=2):
    # generate evenly distributed preference vector
    assert dim > 1
    if dim == 2:
        ang0 = 1e-6 if min_angle is None else min_angle
        ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
        angles = np.linspace(ang0, ang1, n, endpoint=True)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.c_[x, y]
    elif dim == 3:
        # Fibonacci sphere algorithm
        # https://stackoverflow.com/a/26127012
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        n = n*8   # we are only looking at the positive part
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            if x >=0 and y>=0 and z>=0:
                points.append((x, y, z))
        return np.array(points)
    else:
        # this is an unsolved problem for more than 3 dimensions
        # we just generate random points
        points = np.random.rand(n, dim)
        points /= points.sum(axis=1).reshape(n, 1)
        return points


def num_parameters(params):
    if isinstance(params, torch.nn.Module):
        params = params.parameters()
    model_parameters = filter(lambda p: p.requires_grad, params)
    return int(sum([np.prod(p.size()) for p in model_parameters]))


def angle(grads):
    grads = flatten_grads(grads)
    return torch.cosine_similarity(grads[0], grads[1], dim=0)


def flatten_grads(grads):
    result = []
    for grad in grads:
        flatten = torch.cat(([torch.flatten(g) for g in grad.values()]))
        result.append(flatten)
    return result


def reset_weights(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()


def dict_to_cuda(d):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_runname(settings):
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ and 'hpo' not in settings['logdir'] else None
    if slurm_job_id:
        runname = f"{slurm_job_id}"
        if 'ablation' in settings['logdir']:
            runname += f"_{settings['lamda']}_{settings['alpha']}"
    else:
        runname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if 'task_id' in settings:
        runname += f"_{settings['task_id']:03d}"
    return runname


def calc_gradients(batch, model, objectives):
    # store gradients and objective values
    gradients = []
    obj_values = []
    for i, objective in enumerate(objectives):
        # zero grad
        model.zero_grad()
        
        logits = model(batch)
        batch.update(logits)

        output = objective(**batch)
        output.backward()
        
        obj_values.append(output.item())
        gradients.append({})
        
        private_params = model.private_params() if hasattr(model, 'private_params') else []
        for name, param in model.named_parameters():
            not_private = all([p not in name for p in private_params])
            if not_private and param.requires_grad and param.grad is not None:
                gradients[i][name] = param.grad.data.detach().clone()
    
    return gradients, obj_values


class RunningMean():

    def __init__(self, len=100) -> None:
        super().__init__()
        self.queue = collections.deque(maxlen=len)

    def __call__(self, x):
        self.queue.append(x)
        return np.mean(list(self.queue)).item()


class ParetoFront():


    def __init__(self, labels, logdir='tmp', prefix=""):
        self.labels = labels
        self.logdir = os.path.join(logdir, 'pf')
        self.prefix = prefix
        self.points = np.array([])

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)


    def append(self, point):
        point = np.array(point)
        if not len(self.points):
            self.points = point
        else:
            self.points = np.vstack((self.points, point))
    
    
    def plot(self):
        p = self.points
        plt.plot(p[:, 0], p[:, 1], 'o')
        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])
        plt.savefig(os.path.join(self.logdir, "x_{}.png".format(self.prefix)))
        plt.close()
