import torch

from utils import model_from_dataset
from .base import BaseMethod


class UniformScalingMethod(BaseMethod):

    def __init__(self, objectives, **kwargs):
        self.objectives = objectives
        self.J = len(objectives)
        self.model = model_from_dataset(method='uniform_scaling', **kwargs).cuda()


    def model_params(self):
        return list(self.model.parameters())

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):
        batch.update(self.model(batch))
        loss = sum([1/self.J * o(**batch) for o in self.objectives])
        loss.backward()
        return loss.item()


    def eval_step(self, batch, test_rays=None):
        self.model.eval()
        with torch.no_grad():
            return[self.model(batch)]