import torch
import torch.nn as nn
import numpy as np

from utils import num_parameters, model_from_dataset, circle_points
from ..base import BaseMethod


class Upsampler(nn.Module):


    def __init__(self, K, child_model, input_dim):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            self.tabular = False
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(input_dim[-2:])
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions, got {dim}")

        self.child_model = child_model


    def forward(self, batch):
        x = batch['data']
        
        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1)
        
        if not self.tabular:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.transposed_cnn(a)
        
        x = torch.cat((x, a), dim=1)
        return self.child_model(dict(data=x))

    
    def private_params(self):
        if hasattr(self.child_model, 'private_params'):
            return self.child_model.private_params()
        else:
            return []



class COSMOSMethod(BaseMethod):

    def __init__(self, objectives, alpha, lamda, dim, n_test_rays, **kwargs):
        """
        Instanciate the cosmos solver.

        Args:
            objectives: A list of objectives
            alpha: Dirichlet sampling parameter (list or float)
            lamda: Cosine similarity penalty
            dim: Dimensions of the data
            n_test_rays: The number of test rays used for evaluation.
        """
        self.objectives = objectives
        self.K = len(objectives)
        self.alpha = alpha
        self.n_test_rays = n_test_rays
        self.lamda = lamda

        dim = list(dim)
        dim[0] = dim[0] + self.K

        model = model_from_dataset(method='cosmos', dim=dim, **kwargs)
        self.model = Upsampler(self.K, model, dim).cuda()

        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))


    def step(self, batch):
        # step 1: sample alphas
        if isinstance(self.alpha, list):
            batch['alpha']  = torch.from_numpy(
                np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()
            ).cuda()
        elif self.alpha > 0:
            batch['alpha']  = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float.")


        # step 2: calculate loss
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        task_losses = []
        for a, objective in zip(batch['alpha'], self.objectives):
            task_loss = objective(**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss)
        
        cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses), batch['alpha'], dim=0)
        loss_total -= self.lamda * cossim
            
        loss_total.backward()
        return loss_total.item(), cossim.item()


    def eval_step(self, batch, test_rays=None):
        self.model.eval()
        logits = []
        with torch.no_grad():
            if test_rays is None:
                test_rays = circle_points(self.n_test_rays, dim=self.K)

            for ray in test_rays:
                ray = torch.from_numpy(ray.astype(np.float32)).cuda()
                ray /= ray.sum()

                batch['alpha'] = ray
                logits.append(self.model(batch))
        return logits

