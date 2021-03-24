import torch
import random
import numpy as np

# seed now to be save and overwrite later
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

import os
import pathlib
import json
import torch.utils.data as data

import utils
from main import parse_args, solver_from_name
from scores import from_objectives, mcr
from objectives import from_name
from hv import HyperVolume


def evaluate(j, e, solver, scores1, scores2, data_loader, logdir, reference_point, split, result_dict):
    """
    Do one forward pass through the dataloader and log the scores.
    """
    assert split in ['train', 'val', 'test']

    # mode = 'mcr'
    mode = 'pf'

    if mode == 'pf':
        # generate Pareto front
        assert len(scores1) == len(scores2) <= 3, "Cannot generate cirlce points for more than 3 dimensions."
        n_test_rays = 25
        test_rays = utils.circle_points(n_test_rays, dim=len(scores1))
    elif mode == 'mcr':
        # calculate the MRCs using a middle ray
        test_rays = np.ones((1, len(scores1)))
        test_rays /= test_rays.sum(axis=1).reshape(1, 1)
    else:
        raise ValueError()
    
    print(test_rays[0])

    # we wanna calculate the loss and mcr
    score_values1 = np.array([])
    score_values2 = np.array([])
    
    for k, batch in enumerate(data_loader):
        print(f'eval batch {k+1} of {len(data_loader)}')
        batch = utils.dict_to_cuda(batch)
        
        # more than one for some solvers
        s1 = []
        s2 = []
        for l in solver.eval_step(batch, test_rays):
            batch.update(l)
            s1.append([s(**batch) for s in scores1])
            s2.append([s(**batch) for s in scores2])
        if score_values1.size == 0:
            score_values1 = np.array(s1)
            score_values2 = np.array(s2)
        else:
            score_values1 += np.array(s1)       
            score_values2 += np.array(s2)  
    
    score_values1 /= len(data_loader)
    score_values2 /= len(data_loader)

    hv = HyperVolume(reference_point)

    if mode == 'pf':
        pareto_front = utils.ParetoFront([s.__class__.__name__ for s in scores1], logdir, "{}_{:03d}".format(split, e))
        pareto_front.append(score_values1)
        pareto_front.plot()
        volume = hv.compute(score_values1)
    else:
        volume = -1

    result = {
        "scores_loss": score_values1.tolist(),
        "scores_mcr": score_values2.tolist(),
        "hv": volume,
        "task": j,
        # expected by some plotting code
        "max_epoch_so_far": -1,
        "max_volume_so_far": -1,
        "training_time_so_far": -1,
    }

    result.update(solver.log())

    result_dict[f"start_{j}"][f"epoch_{e}"] = result

    with open(pathlib.Path(logdir) / f"{split}_results.json", "w") as file:
        json.dump(result_dict, file)
    
    return result_dict


# Path to the checkpoint dir. Use the method folder.
CHECKPOINT_DIR = 'path/to/checkpoints'


def eval(settings):
    """
    The full evaluation loop. Generate scores for all checkpoints found in the directory specified above.

    Uses the same ArgumentParser as main.py to determine the method and dataset.
    """

    settings['batch_size'] = 2048

    print("start evaluation with settings", settings)

    # create the experiment folders
    logdir = os.path.join(settings['logdir'], settings['method'], settings['dataset'], utils.get_runname(settings))
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)


    # prepare
    train_set = utils.dataset_from_name(split='train', **settings)
    val_set = utils.dataset_from_name(split='val', **settings)
    test_set = utils.dataset_from_name(split='test', **settings)

    train_loader = data.DataLoader(train_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    val_loader = data.DataLoader(val_set, settings['batch_size'], shuffle=True,num_workers=settings['num_workers'])
    test_loader = data.DataLoader(test_set, settings['batch_size'], settings['num_workers'])

    objectives = from_name(settings.pop('objectives'), val_set.task_names())
    scores1 = from_objectives(objectives)
    scores2 = [mcr(o.label_name, o.logits_name) for o in objectives]

    solver = solver_from_name(objectives=objectives, **settings)

    train_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    val_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))
    test_results = dict(settings=settings, num_parameters=utils.num_parameters(solver.model_params()))

    
    task_ids = settings['task_ids'] if settings['method'] == 'SingleTask' else [0]
    for j in task_ids:
        if settings['method'] == 'SingleTask':
            # we ran it in parallel
            checkpoints = pathlib.Path(CHECKPOINT_DIR).glob(f'**/*_{j:03d}/*/c_*.pth')
        else:
            checkpoints = pathlib.Path(CHECKPOINT_DIR).glob('**/c_*.pth')
        
        train_results[f"start_{j}"] = {}
        val_results[f"start_{j}"] = {}
        test_results[f"start_{j}"] = {}

        for c in sorted(checkpoints):

            #c = list(sorted(checkpoints))[-1]
            print("checkpoint", c)
            _, e = c.stem.replace('c_', '').split('-')

            j = int(j)
            e = int(e)
            
            solver.model.load_state_dict(torch.load(c))

            # Validation results
            val_results = evaluate(j, e, solver, scores1, scores2, val_loader, logdir, 
                reference_point=settings['reference_point'],
                split='val',
                result_dict=val_results)

            # Test results
            test_results = evaluate(j, e, solver, scores1, scores2, test_loader, logdir, 
                reference_point=settings['reference_point'],
                split='test',
                result_dict=test_results)

            # Train results
            # train_results = evaluate(j, e, solver, scores1, scores2, train_loader, logdir, 
            #     reference_point=settings['reference_point'],
            #     split='train',
            #     result_dict=train_results)


if __name__ == "__main__":

    settings = parse_args()
    eval(settings)