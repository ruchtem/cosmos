import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from utils import num_parameters, circle_points
from ..base import BaseMethod

from .models import PHNHyper, PHNTarget
from .solvers import LinearScalarizationSolver, EPOSolver


class LeNetPHNHyper(PHNHyper):
    pass


class LeNetPHNTargetWrapper(PHNTarget):

    def forward(self, x, weights=None):
        logits = super().forward(x, weights)
        return dict(logits_l=logits[0], logits_r=logits[1])


# fully connected hyper version
# this is unfortunately not published, therefore I implemented it myself
class FCPHNHyper(nn.Module):
    
    def __init__(self, dim, ray_hidden_dim=100, n_tasks=2):
        super().__init__()

        self.feature_dim = dim[0]

        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.fc_0_weights = nn.Linear(ray_hidden_dim, 60*self.feature_dim)
        self.fc_0_bias = nn.Linear(ray_hidden_dim, 60)
        self.fc_1_weights = nn.Linear(ray_hidden_dim, 25*60)
        self.fc_1_bias = nn.Linear(ray_hidden_dim, 25)
        self.fc_2_weights = nn.Linear(ray_hidden_dim, 1*25)
        self.fc_2_bias = nn.Linear(ray_hidden_dim, 1)

    def forward(self, ray):
        x = self.ray_mlp(ray)
        out_dict = {
            'fc0.weights': self.fc_0_weights(x).reshape(60, self.feature_dim),
            'fc0.bias': self.fc_0_bias(x),
            'fc1.weights': self.fc_1_weights(x).reshape(25, 60),
            'fc1.bias': self.fc_1_bias(x),
            'fc2.weights': self.fc_2_weights(x).reshape(1, 25),
            'fc2.bias': self.fc_2_bias(x),
        }
        return out_dict


class FCPHNTarget(nn.Module):

    def forward(self, x, weights):
        x = F.linear(
            x,
            weight=weights['fc0.weights'],
            bias=weights['fc0.bias']
        )
        x = F.relu(x)
        x = F.linear(
            x,
            weight=weights['fc1.weights'],
            bias=weights['fc1.bias']
        )
        x = F.relu(x)
        x = F.linear(
            x,
            weight=weights['fc2.weights'],
            bias=weights['fc2.bias']
        )
        return {'logits': x}




class HypernetMethod(BaseMethod):

    def __init__(self, objectives, dim, n_test_rays, alpha, internal_solver, **kwargs):
        self.objectives = objectives
        self.n_test_rays = n_test_rays
        self.alpha = alpha
        self.K = len(objectives)

        if len(dim) == 1:
            # tabular
            hnet = FCPHNHyper(dim, ray_hidden_dim=100)
            net = FCPHNTarget()
        elif len(dim) ==3:
            # image
            hnet: nn.Module = LeNetPHNHyper([9, 5], ray_hidden_dim=100)
            net: nn.Module = LeNetPHNTargetWrapper([9, 5])
        else:
            raise ValueError(f"Unkown dim {dim}, expected len 1 or len 3")

        print("Number of parameters: {}".format(num_parameters(hnet)))

        self.model = hnet.cuda()
        self.net = net.cuda()

        if internal_solver == 'linear':
            self.solver = LinearScalarizationSolver(n_tasks=len(objectives))
        elif internal_solver == 'epo':
            self.solver = EPOSolver(n_tasks=len(objectives), n_params=num_parameters(hnet))


    def step(self, batch):
        if self.alpha > 0:
            ray = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(len(self.objectives))], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            alpha = torch.empty(1, ).uniform_(0., 1.)
            ray = torch.tensor([alpha.item(), 1 - alpha.item()]).cuda()

        img = batch['data']

        weights = self.model(ray)
        batch.update(self.net(img, weights))

        losses = torch.stack([o(**batch) for o in self.objectives])

        ray = ray.squeeze(0)
        loss = self.solver(losses, ray, list(self.model.parameters()))
        loss.backward()

        return loss.item()
    
    def eval_step(self, batch):
        self.model.eval()

        test_rays = circle_points(self.n_test_rays, dim=self.K)

        logits = []
        for ray in test_rays:
            ray = torch.from_numpy(ray.astype(np.float32)).cuda()
            ray /= ray.sum()

            weights = self.model(ray)
            logits.append(self.net(batch['data'], weights))
        return logits




