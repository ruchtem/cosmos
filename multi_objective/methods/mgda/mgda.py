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

            # As an approximate solution we only need gradients for input
            # if isinstance(rep, list):
            #     # This is a hack to handle psp-net
            #     rep = rep[0]
            #     rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
            #     list_rep = True
            # else:
            #     rep_variable = Variable(rep.data.clone(), requires_grad=True)
            #     list_rep = False

            # Compute gradients of each loss function wrt z
            

            gradients = []
            obj_values = []
            for i, objective in enumerate(self.objectives):
                # zero grad
                self.model.zero_grad()
                
                logits = self.model.forward_linear(rep, i)
                batch.update(logits)

                output = objective(**batch)
                output.backward()
                
                obj_values.append(output.item())
                gradients.append({})
                
                private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
                for name, param in self.model.named_parameters():
                    not_private = all([p not in name for p in private_params])
                    if not_private and param.requires_grad and param.grad is not None:
                        gradients[i][name] = param.grad.data.detach().clone()
                        param.grad = None
                self.model.zero_grad()
            
            grads = gradients


            # for t in tasks:
            #     self.model.zero_grad()
            #     out_t, masks[t] = model[t](rep_variable, None)
            #     loss = loss_fn[t](out_t, labels[t])
            #     loss_data[t] = loss.data[0]
            #     loss.backward()
            #     grads[t] = []
            #     if list_rep:
            #         grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
            #         rep_variable[0].grad.data.zero_()
            #     else:
            #         grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
            #         rep_variable.grad.data.zero_()

        else:
            # This is MGDA
            grads, obj_values = calc_gradients(batch, self.model, self.objectives)

            # for t in tasks:
            #     # Comptue gradients of each loss function wrt parameters
            #     self.model.zero_grad()
            #     rep, mask = model['rep'](images, mask)
            #     out_t, masks[t] = model[t](rep, None)
            #     loss = loss_fn[t](out_t, labels[t])
            #     loss_data[t] = loss.data[0]
            #     loss.backward()
            #     grads[t] = []
            #     for param in self.model['rep'].parameters():
            #         if param.grad is not None:
            #             grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

        # Normalize all gradients, this is optional and not included in the paper.

        gn = gradient_normalizers(grads, obj_values, self.normalization_type)
        for t in range(len(self.objectives)):
            for gr_i in grads[t]:
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        grads = [[v for v  in d.values()] for d in grads]
        sol, min_norm = MinNormSolver.find_min_norm_element(grads)
        # for i, t in enumerate(range(len(self.objectives))):
        #     scale[t] = float(sol[i])

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