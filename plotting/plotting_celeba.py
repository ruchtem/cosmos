import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multi_objective.hv import HyperVolume

from plotting import (
    font_size,
    markers,
    colors,
    dirname,
    methods,
    titles,
    ax_lables,
    load_files,
)

from plotting_convergence import (
    adjust_lightness
)

dirname = "../results_plot/results_celeba/"

scenario = 'mcr_10'

if scenario == 'easy':
    methods = ['cosmos_2_pf', 'SingleTask', ]
    task = [16, 22]
elif scenario == 'hard':
    methods = ['cosmos_2_pf', 'SingleTask', ]
    task = [25, 27]
elif scenario == '3tasks_pf':
    methods = ['cosmos_3_pf', ]
    task = [16, 22, 24]
elif scenario == 'mcr_3':
    methods = ['cosmos_3', ]
    task = [16, 22, 24]
elif scenario == 'mcr_4':
    methods = ['cosmos_4', 'SingleTask_mcr', ]
    task = [16, 22, 24, 26]
elif scenario == 'mcr_10':
    methods = ['cosmos_10', 'SingleTask_mcr', ]
    task = [8, 36, 4, 16, 7, 31, 28, 30, 24, 13]

print(scenario)

# Load the data
p = Path(dirname)
hv = HyperVolume([2 for _ in range(len(task))])
results = {}

for method in methods:
    results[method] = {}

    val_files = list(sorted(p.glob(f'**/{method}/**/val*.json')))
    test_files = list(sorted(p.glob(f'**/{method}/**/test*.json')))
    train_files = list(sorted(p.glob(f'**/{method}/**/train*.json')))

    if len(val_files) == len(test_files) == 0:
        continue

    assert len(val_files) == len(test_files)
    results[method] = {}
    if method == 'SingleTask' or method == 'SingleTask_mcr':
        idx = val_files[0].parts[4].find('_')
        val_file = list(filter(lambda x: task[0] == int(x.parts[4][idx+1:]), val_files))
        test_file = list(filter(lambda x: task[0] == int(x.parts[4][idx+1:]), test_files))
            
        data_val = load_files(val_file)[0]
        data_test = load_files(test_file)[0]

        
        for e in range(100):


            if f'epoch_{e}' in data_val[f'start_{task[0]}']:

                val_scores_loss = []
                val_scores_mcr = []
                test_scores_loss = []
                test_scores_mcr = []
                for i, task_id in enumerate(task):
                    val_scores_loss.append(data_val[f'start_{task_id}'][f'epoch_{e}']['scores_loss'][0][i])
                    val_scores_mcr.append(data_val[f'start_{task_id}'][f'epoch_{e}']['scores_mcr'][0][i])
                    test_scores_loss.append(data_test[f'start_{task_id}'][f'epoch_{e}']['scores_loss'][0][i])
                    test_scores_mcr.append(data_test[f'start_{task_id}'][f'epoch_{e}']['scores_mcr'][0][i])

                r = {
                    'val_scores_loss': [val_scores_loss],
                    'val_scores_mcr': [val_scores_mcr],
                    'test_scores_loss': [test_scores_loss],
                    'test_scores_mcr': [test_scores_mcr],
                }
                results[method][e] = r

    else:
        data_val = load_files(val_files)
        data_test = load_files(test_files)

        data_val = list(filter(lambda x: x['settings']['task_ids'] == task, data_val))
        data_test = list(filter(lambda x: x['settings']['task_ids'] == task, data_test))

        assert len(data_test) == len(data_val) == 1

        data_val = data_val[0]
        data_test = data_test[0]

        for e in range(40):       #100):
            if f'epoch_{e}' in data_val[f'start_0']:
                v = data_val[f'start_0'][f'epoch_{e}']
                t = data_test[f'start_0'][f'epoch_{e}']

                if v['hv'] == -1:
                    v['hv'] = hv.compute(v['scores_loss'])
                if t['hv'] == -1:
                    t['hv'] = hv.compute(t['scores_loss'])

                r = {
                    'val_scores_loss': v['scores_loss'],
                    'val_scores_mcr': v['scores_mcr'],
                    'test_scores_loss': t['scores_loss'],
                    'test_scores_mcr': t['scores_mcr'],
                    'val_hv': v['hv'],
                    'test_hv': t['hv'],
                }
                results[method][e] = r


# determine best epoch
early_stop= {}
for method in methods:
    if 'cosmos' in method:

        best_hv = 0
        best_e = -1

        for e, d in results[method].items():
            if d['val_hv'] > best_hv:
                best_hv = d['val_hv']
                best_e = e
        early_stop[method] = {
            'e': best_e,
            'hv': best_hv,
        }
    elif method == 'SingleTask' or method == 'SingleTask_mcr':

        r = {}

        for i, task_id in enumerate(task):

            best_score = 10000
            best_e = -1

            for e, d in results[method].items():
                if d['val_scores_loss'][0][i] < best_score:
                    best_score = d['val_scores_loss'][0][i]
                    best_e = e
            
            r[task_id] = {
                'e': best_e,
                'score': best_score,
            }
        
        early_stop[method] = r



with open('results_celeba.json', 'w') as outfile:
    json.dump(results, outfile)


# Generate the plots and tables
plt.rcParams.update({'font.size': font_size})

limits = {
    # dataset: [left, right, bottom, top]
    'multi_mnist': [.2, .7, .25, .7], 
    'multi_fashion': [.45, 1, .45, 1], 
    'multi_fashion_mnist': [.1, 1, .35, .6],
}

ax_lables = {
    'adult': ('Loss', 'DEO'),
    'compas': ('Loss', 'DEO'),
    'credit': ('Loss', 'DEO'), 
    'multi_mnist': ('Loss Task TL', 'Loss Task BR'), 
    'multi_fashion': ('Loss Task TL', 'Loss Task BR'), 
    'multi_fashion_mnist': ('Loss Task TL', 'Loss Task BR'), 
}

titles = {
    16: 'CelebA Easy Tasks',
    25: 'CelebA Hard Tasks'
}

ax_lables = {
    16: ('Loss “Goatee“', 'Loss “Mustache”', 'Loss “No Beard”'),
    25: ('Loss “Oval Face“', 'Loss “Pointy Nose”')
}


def plot_2d(epochs):

    fig, ax = plt.subplots(1)


    # single_task
    method = 'SingleTask'
    e1 = early_stop['SingleTask'][task[0]]['e']
    e2 = early_stop['SingleTask'][task[1]]['e']
    s0 = np.array(results['SingleTask'][e1]['test_scores_loss'])
    s1 = np.array(results['SingleTask'][e2]['test_scores_loss'])

    s0 = np.squeeze(s0)
    s1 = np.squeeze(s1)
    ax.axvline(x=s0[0], color=colors[method], linestyle='-.')
    ax.axhline(y=s1[1], color=colors[method], linestyle='-.', label="Single Task")
    
    # cosmos
    method = 'cosmos_2_pf'
    # convergence
    color_shades = np.linspace(1.8, 1.6, len(epochs)).tolist()
    for i, e in enumerate(epochs): 
        s = np.array(results[method][e]['test_scores_loss'])

        ax.plot(
            s[:, 0], 
            s[:, 1], 
            color=adjust_lightness(colors[method], amount=color_shades[i]),
            marker='.',
            linestyle='--' if method != 'ParetoMTL' else ' ',
            label="COSMOS epoch {}".format(e+1)
        )
    
    # final
    final_e = early_stop[method]['e']
    s = np.array(results[method][final_e]['test_scores_loss'])

    ax.plot(
        s[:, 0], 
        s[:, 1], 
        color=adjust_lightness(colors[method], amount=1),
        marker=markers[method],
        linestyle='--' if method != 'ParetoMTL' else ' ',
        label="COSMOS converged"
    )

    ax.set_title(titles[task[0]])
    ax.set_xlabel(ax_lables[task[0]][0])
    ax.set_ylabel(ax_lables[task[0]][1])
    ax.legend(loc='upper right')


    plt.savefig(f'celeba' + '-'.join([str(t) for t in task]) + '.pdf')
    plt.close()


def plot_3d():
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    
    method = 'cosmos_ln'
    e = early_stop['cosmos_3_pf']['e']
    s = np.array(results['cosmos_3_pf'][e]['test_scores_loss'])

    ax.plot_trisurf(s[:, 0], s[:, 1], s[:, 2])

    ax.set_xlabel(ax_lables[task[0]][0])
    ax.set_ylabel(ax_lables[task[0]][1])
    ax.set_zlabel(ax_lables[task[0]][2])
    
    ax.view_init(elev=22, azim=52)
    ax.set_title("CelebA three objectives, perspective 1", y=1)
    plt.savefig('3_obj_1.pdf')

    ax.set_title("CelebA three objectives, perspective 2", y=1)
    ax.view_init(elev=15, azim=123)
    plt.savefig('3_obj_2.pdf')


print(early_stop)
if scenario == 'easy' or scenario == 'hard':
    plot_2d([0, 2, 4])

    e1 = early_stop['SingleTask'][task[0]]['e']
    e2 = early_stop['SingleTask'][task[1]]['e']
    final_e = early_stop['cosmos_2_pf']['e']

    s0 = results['SingleTask'][e1]['test_scores_mcr'][0][0]
    s1 = results['SingleTask'][e2]['test_scores_mcr'][0][1]


    print('single task final mcr (averaged):', sum([s0, s1])/2)
    print('cosmos mcr (middle ray, averaged)', sum(results['cosmos_2_pf'][final_e]['test_scores_mcr'][12])/2)

    hv = HyperVolume([2, 2])
    print('hv single task', hv.compute(results['SingleTask'][e1]['test_scores_loss']))
    print('hv single task', results['cosmos_2_pf'][final_e]['test_hv'])

    print("single task mcr, 2 tasks", [s0, s1])
    print("cosmos mcr, 2 tasks", results['cosmos_2_pf'][final_e]['test_scores_mcr'][12])

if scenario == '3tasks_pf':
    plot_3d()

if scenario == 'mcr_3':
    final_e = early_stop['cosmos_3']['e']
    print("cosmos mcr, 3 tasks", results['cosmos_3'][final_e]['test_scores_mcr'])


if scenario == 'mcr_4':
    e1 = early_stop['SingleTask_mcr'][task[0]]['e']
    e2 = early_stop['SingleTask_mcr'][task[1]]['e']
    e3 = early_stop['SingleTask_mcr'][task[2]]['e']
    e4 = early_stop['SingleTask_mcr'][task[3]]['e']
    final_e = early_stop['cosmos_4']['e']

    s1 = results['SingleTask_mcr'][e1]['test_scores_mcr'][0][0]
    s2 = results['SingleTask_mcr'][e2]['test_scores_mcr'][0][1]
    s3 = results['SingleTask_mcr'][e3]['test_scores_mcr'][0][2]
    s4 = results['SingleTask_mcr'][e3]['test_scores_mcr'][0][3]

    print("single task mcr, 3 tasks", [s1, s2, s3, s4])
    print("cosmos mcr, 3 tasks", results['cosmos_4'][final_e]['test_scores_mcr'])


if scenario == 'mcr_10':
    
    for i, task_id in enumerate(task):
        e = early_stop['SingleTask_mcr'][task_id]['e']
        s = results['SingleTask_mcr'][e]['test_scores_mcr'][0][i]
        print(f"single task id {task_id}, result: {s:.2%}")
    
    final_e = early_stop['cosmos_10']['e']
    for n in results['cosmos_10'][final_e]['test_scores_mcr'][0]:
        print(f"cosmos {n:.2%}")
    




























# def plot_gain():

#     st = np.array(results['SingleTask']['test_scores'])
#     co = np.array(results['cosmos_ln']['even']['test_scores'])

#     gain = st/co

#     fig,ax = plt.subplots(1)

#     ax.hist(gain.ravel(), bins=50, histtype='stepfilled', alpha=.7)
    
#     ylim = 7
#     ax.set_ylim(top=ylim)
#     mean = np.mean(gain)
#     std = np.std(gain)
#     height = ylim * .9
#     ax.vlines(mean, 0, ylim)
#     ax.arrow(mean, height, -std, 0, length_includes_head=True, head_length=0.01, head_width=0.1)
#     ax.arrow(mean, height, +std, 0, length_includes_head=True, head_length=0.01, head_width=0.1)
#     ax.text(mean-std/2, height + .1, "std", horizontalalignment='center')
#     ax.text(mean+std/2, height + .1, "std", horizontalalignment='center')

#     ax.set_xlabel(r"$MCR^{(COSMOS)} / MCR^{(Single Task)}$")
#     ax.set_ylabel("Frequencies")
#     #ax.set_title("MRC COSMOS / MRC Single Task = ")

#     #plt.show()
#     plt.savefig('hist')
#     plt.close()


# plot_gain()


# def plot_comparison():
#     even = np.array(results['cosmos_ln']['even']['train_scores'])

#     eye = np.array(results['cosmos_ln']['eye_soften']['train_scores'])

#     gain = even / np.expand_dims(np.diagonal(eye), 0)

#     print()


# plot_comparison()