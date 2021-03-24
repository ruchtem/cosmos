import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from min_norm_solvers import MinNormSolver

from utils import calc_gradients, circle_points, model_from_dataset, reset_weights
from ..base import BaseMethod


def get_d_paretomtl_init(grads, losses, preference_vectors, pref_idx):
    """ 
    calculate the gradient direction for ParetoMTL initialization 

    Args:
        grads: flattened gradients for each task
        losses: values of the losses for each task
        preference_vectors: all preference vectors u
        pref_idx: which index of u we are currently using
    
    Returns:
        flag: is a feasible initial solution found?
        weight: 
    """
    
    flag = False
    nobj = losses.shape
   
    # check active constraints, Equation 7
    current_pref = preference_vectors[pref_idx]         # u_k
    w = preference_vectors - current_pref               # (u_j - u_k) \forall j = 1, ..., K
    gx =  torch.matmul(w,losses/torch.norm(losses))     # In the paper they do not normalize the loss
    idx = gx >  0                                       # I(\theta), i.e the indexes of the active constraints
    
    active_constraints = w[idx]     # constrains which are violated, i.e. gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        # Equation 9
        # w[idx] = set of active constraints, i.e. where the solution is closer to another preference vector than the one desired.
        gx_gradient = torch.matmul(active_constraints, grads)    # We need to take the derivatives of G_j which is w.dot(grads)
        sol, nd = MinNormSolver.find_min_norm_element([[gx_gradient[t]] for t in range(len(gx_gradient))])
        sol = torch.Tensor(sol).cuda()
    
    # from MinNormSolver we get the weights (alpha) for each gradient. But we need the weights for the losses?
    weight = torch.matmul(sol, active_constraints)

    return flag, weight


def get_d_paretomtl(grads, losses, preference_vectors, pref_idx):
    """
    calculate the gradient direction for ParetoMTL 
    
    Args:
        grads: flattened gradients for each task
        losses: values of the losses for each task
        preference_vectors: all preference vectors u
        pref_idx: which index of u we are currently using
    """
    
    # check active constraints
    current_weight = preference_vectors[pref_idx]
    rest_weights = preference_vectors 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,losses/torch.norm(losses))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        # here there are no active constrains in gx
        sol, nd = MinNormSolver.find_min_norm_element_FW([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()
    else:
        # we have active constraints, i.e. we have move too far away from out preference vector
        #print('optim idx', idx)
        vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        sol = torch.Tensor(sol).cuda()

        # FIX: handle more than just 2 objectives
        n = preference_vectors.shape[1]
        weights = []
        for i in range(n):
            weight_i =  sol[i] + torch.sum(torch.stack([sol[j] * w[idx][j - n , i] for j in torch.arange(n, n + torch.sum(idx))]))
            weights.append(weight_i)
        # weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        # weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        # weight = torch.stack([weight0,weight1])

        weight = torch.stack(weights)
        
        
        return weight


class ParetoMTLMethod(BaseMethod):

    def __init__(self, objectives, num_starts, **kwargs):
        assert len(objectives) <= 2
        self.objectives = objectives
        self.num_pareto_points = num_starts
        self.init_solution_found = False

        self.model = model_from_dataset(method='paretoMTL', **kwargs).cuda()
        self.pref_idx = -1
        # the original ref_vec can be obtained by circle_points(self.num_pareto_points, min_angle=0.0, max_angle=0.5 * np.pi)
        # we use the same min angle / max angle as for the other methods for comparison.
        self.ref_vec = torch.Tensor(circle_points(self.num_pareto_points)).cuda().float()


    def new_epoch(self, e):
        if e == 0:
            # we're restarting
            self.pref_idx += 1
            reset_weights(self.model)
            self.init_solution_found = False
        self.e = e
        self.model.train()


    def log(self):
        return {"train_ray": self.ref_vec[self.pref_idx].cpu().numpy().tolist()}


    def _find_initial_solution(self, batch):

        grads = {}
        losses_vec = []
        
        # obtain and store the gradient value
        for i in range(len(self.objectives)):
            self.model.zero_grad()
            batch.update(self.model(batch))
            task_loss = self.objectives[i](**batch) 
            losses_vec.append(task_loss.data)
            
            task_loss.backward()
            
            grads[i] = []
            
            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment
            private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
            for name, param in self.model.named_parameters():
                if name not in private_params and param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

        
        grads_list = [torch.cat([g for g in grads[i]]) for i in range(len(grads))]
        grads = torch.stack(grads_list)
        
        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        self.init_solution_found, weight_vec = get_d_paretomtl_init(grads, losses_vec, self.ref_vec, self.pref_idx)
        
        if self.init_solution_found:
            print("Initial solution found")
        
        # optimization step
        self.model.zero_grad()
        for i in range(len(self.objectives)):
            batch.update(self.model(batch))
            task_loss = self.objectives[i](**batch) 
            if i == 0:
                loss_total = weight_vec[i] * task_loss
            else:
                loss_total = loss_total + weight_vec[i] * task_loss
        
        loss_total.backward()
        return loss_total.item()


    def step(self, batch):

        if self.e < 2 and not self.init_solution_found:
            # run at most 2 epochs to find the initial solution
            # stop early once a feasible solution is found 
            # usually can be found with a few steps
            return self._find_initial_solution(batch)
        else:
            # run normal update
            gradients, obj_values = calc_gradients(batch, self.model, self.objectives)
            
            grads = [torch.cat([torch.flatten(v) for k, v in sorted(grads.items())]) for grads in gradients]
            grads = torch.stack(grads)
            
            # calculate the weights
            losses_vec = torch.Tensor(obj_values).cuda()
            weight_vec = get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)
            
            normalize_coeff = len(self.objectives) / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            loss_total = None
            for a, objective in zip(weight_vec, self.objectives):
                logits = self.model(batch)
                batch.update(logits)
                task_loss = objective(**batch)

                loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            
            loss_total.backward()
            return loss_total.item()
            
            # private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
            # for name, param in self.model.named_parameters():
            #     if name not in private_params:
            #         param.grad.data.zero_()
            #         param.grad = sum(weight_vec[o] * gradients[o][name] for o in range(len(self.objectives))).cuda()
    
    def eval_step(self, batch):
        self.model.eval()
        return [self.model(batch)]