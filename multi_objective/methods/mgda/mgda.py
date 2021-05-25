# code from https://github.com/intel-isl/MultiObjectiveOptimization/blob/master/multi_task/train_multi_task.py
# and adapted

import torch
from torch.autograd import Variable

from ..base import BaseMethod
from min_norm_solvers import MinNormSolver, gradient_normalizers
from utils import model_from_dataset, calc_gradients


class MGDAMethod(BaseMethod):

    def __init__(self, objectives, approximate_norm_solution, normalization_type, **kwargs) -> None:
        super().__init__()
        self.objectives = objectives
        self.approximate_norm_solution = approximate_norm_solution
        self.normalization_type = normalization_type
        self.task_ids = kwargs['task_ids']

        self.model =  model_from_dataset(method='mdga', **kwargs).cuda()

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):

        # Scaling the loss functions based on the algorithm choice
        # loss_data = {}
        # grads = {}
        # scale = {}
        # mask = None
        # masks = {}
        
        # Will use our MGDA_UB if approximate_norm_solution is True. Otherwise, will use MGDA
        if self.approximate_norm_solution:
            self.model.zero_grad()
            # First compute representations (z)
            with torch.no_grad():
                # images_volatile = Variable(images.data, volatile=True)
                # rep, mask = model['rep'](images_volatile, mask)
                rep = self.model.forward_feature_extraction(batch)

            # we require gradients wrt to (z)
            rep = Variable(rep, requires_grad=True)

            # Compute gradients of each loss function wrt z
            grads = {t: {} for t in self.task_ids}
            obj_values = {t: None for t in self.task_ids}
            for t, objective in zip(self.task_ids, self.objectives):
                # zero grad
                self.model.zero_grad()
                
                logits = self.model.forward_linear(rep, t)
                batch.update(logits)

                output = objective(**batch)
                output.backward()
                
                obj_values[t] = output.item()

                grads[t]['input'] = rep.grad.data.detach().clone()
                rep.grad.data.zero_()
        else:
            # This is plain MGDA
            grads, obj_values = calc_gradients(batch, self.model, self.objectives)

        # Normalize all gradients, this is optional and not included in the paper.

        gn = gradient_normalizers(grads, obj_values, self.normalization_type)
        for t in self.task_ids:
            for gr_i in grads[t]:
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Min norm solver by Sener and Koltun
        # They don't use their FW solver in their code either.
        # We can also use the scipy implementation by me, does not matter.
        grads = [[v for v  in d.values()] for d in grads.values()]
        sol, min_norm = MinNormSolver.find_min_norm_element(grads)

        # Scaled back-propagation
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        for a, objective in zip(sol, self.objectives):
            task_loss = objective(**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            
        loss_total.backward()
        return loss_total.item(), 0

        # rep, _ = model['rep'](images, mask)
        # for i, t in enumerate(tasks):
        #     out_t, _ = model[t](rep, masks[t])
        #     loss_t = loss_fn[t](out_t, labels[t])
        #     loss_data[t] = loss_t.data[0]
        #     if i > 0:
        #         loss = loss + scale[t]*loss_t
        #     else:
        #         loss = scale[t]*loss_t
        # loss.backward()

    def eval_step(self, batch):
        self.model.eval()
        return [self.model(batch)]