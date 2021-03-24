import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pathlib import Path
import re
from multi_objective.hv import HyperVolume


#
# Helper functions
#

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def get_early_stop(epoch_data, key='hv'):
    assert key in ['hv', 'score', 'last']
    if key == 'hv':
        last_epoch = epoch_data[natural_sort(epoch_data.keys())[-1]]
        return last_epoch['max_epoch_so_far']
    elif key == 'score':
        min_score = 1e15
        min_epoch = -1
        for e in natural_sort(epoch_data.keys()):
            if 'scores' in epoch_data[e]:
                s = epoch_data[e]['scores'][0]
                s = s[epoch_data[e]['task']]

                if s < min_score:
                    min_score = s
                    min_epoch = e

        return int(min_epoch.replace('epoch_', ''))
    elif key == 'last':
        last_epoch = natural_sort(epoch_data.keys())[-1]
        return int(last_epoch.replace('epoch_', ''))


def fix_scores_dim(scores):
    scores = np.array(scores)
    if scores.ndim == 1:
        return np.expand_dims(scores, axis=0).tolist()
    if scores.ndim == 2:
        return scores.tolist()
    if scores.ndim == 3:
        return np.squeeze(scores).tolist()
    raise ValueError()


def lists_to_tuples(dict):
    for k, v in dict.items():
        if isinstance(v, list):
            dict[k] = tuple(v)
    return dict


def compare_settings(data):
    sets = [set(lists_to_tuples(d['settings']).items()) for d in data]
    diff = set.difference(*sets)
    
    assert len(diff) == 1, f"Runs or not similar apart from seed! {diff}"
    assert 'seed' in dict(diff)


def load_files(paths):
    contents = []
    for p in paths:
        with p.open(mode='r') as json_file:
            contents.append(json.load(json_file))
    return contents


def mean_and_std(values):
    return (
        np.array(values).mean(axis=0).tolist(),
        np.array(values).std(axis=0).tolist()
    )


def process_non_pareto_front(data_val, data_test):
    result_i = {}
    # we need to aggregate results from different runs
    result_i['val_scores'] = []
    result_i['test_scores'] = []
    result_i['early_stop_epoch'] = []
    for s in natural_sort(data_val.keys()):
        if 'start_' in s:
            e = "epoch_{}".format(get_early_stop(data_val[s], key=stop_key[method]))
            val_results = data_val[s][e]
            test_results = data_test[s][e]

            # the last training time is the correct one, so just override
            result_i['training_time'] = test_results['training_time_so_far']

            result_i['early_stop_epoch'].append(int(e.replace('epoch_', '')))

            if method == 'SingleTask':
                # we have the task id for the score
                val_score = val_results['scores'][0][val_results['task']]
                result_i['val_scores'].append(val_score)
                test_score = test_results['scores'][0][test_results['task']]
                result_i['test_scores'].append(test_score)
            else:
                # we have no task id
                result_i['val_scores'].append(val_results['scores'])
                result_i['test_scores'].append(test_results['scores'])

    result_i['val_scores'] = fix_scores_dim(result_i['val_scores'])
    result_i['test_scores'] = fix_scores_dim(result_i['test_scores'])

    # compute hypervolume
    hv = HyperVolume(reference_points[dataset])
    result_i['val_hv'] = hv.compute(result_i['val_scores'])
    result_i['test_hv'] = hv.compute(result_i['test_scores'])
    return result_i



#
# Plotting params
#
font_size = 12
figsize=(14, 3.5)

dirname = '../results_plot/results_final_paper'

datasets = ['adult', 'compas', 'credit', 'multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
methods = ['SingleTask', 'hyper_epo', 'hyper_ln', 'ParetoMTL',  'cosmos_ln']

generating_pareto_front = ['cosmos_ln', 'hyper_ln', 'hyper_epo']

stop_key = {
    'SingleTask': 'score', 
    'hyper_epo': 'hv', 
    'hyper_ln': 'hv', 
    'cosmos_ln': 'hv', 
    'ParetoMTL': 'hv', 
}

early_stop = ['hyper_ln', 'hyper_epo', 'SingleTask', 'ParetoMTL']

reference_points = {
    'adult': [2, 2], 'compas': [2, 2], 'credit': [2, 2], 
    'multi_mnist': [2, 2], 'multi_fashion': [2, 2], 'multi_fashion_mnist': [2, 2],
}

plt.rcParams.update({'font.size': font_size})
plt.tight_layout()

markers = {
    'hyper_epo': '.', 
    'hyper_ln': 'x', 
    'cosmos_ln': 'd', 
    'cosmos_2_pf': 'd', 
    'ParetoMTL': '*'
}

colors = {
    'SingleTask': '#1f77b4', 
    'hyper_epo': '#ff7f0e', 
    'hyper_ln': '#2ca02c',
    'cosmos_ln': '#d62728',
    'cosmos_2_pf': '#d62728',
    'ParetoMTL': '#9467bd', 
}

titles = {
    'adult': 'Adult',
    'compas': 'Compass',
    'credit': 'Default', 
    'multi_mnist': "Multi-MNIST", 
    'multi_fashion': 'Multi-Fashion',
    'multi_fashion_mnist': 'Multi-Fashion+MNIST'
}

ax_lables = {
    'adult': ('Binary Cross-Entropy Loss', 'DEO'),
    'compas': ('Binary Cross-Entropy Loss', 'DEO'),
    'credit': ('Binary Cross-Entropy Loss', 'DEO'), 
    'multi_mnist': ('Cross-Entropy Loss Task TL', 'Cross-Entropy Loss Task BR'), 
    'multi_fashion': ('Cross-Entropy Loss Task TL', 'Cross-Entropy Loss Task BR'), 
    'multi_fashion_mnist': ('Cross-Entropy Loss Task TL', 'Cross-Entropy Loss Task BR'), 
}

method_names = {
    'SingleTask': 'Single Task', 
    'hyper_epo': 'PHN-EPO', 
    'hyper_ln': 'PHN-LS',
    'cosmos_ln': 'COSMOS',
    'ParetoMTL': 'ParetoMTL', 
}

limits_baselines = {
    # dataset: [left, right, bottom, top]
    'adult': [.3, .6, -0.01, .14],
    'compas': [0, 1.5, -.01, .35],
    'credit': [.42, .65, -0.001, .017],
    'multi_mnist': [.24, .5, .3, .5], 
    'multi_fashion': [.45, .75, .47, .75], 
    'multi_fashion_mnist': [.18, .6, .4, .6],
}


#
# Load the data
#

p = Path(dirname)
results = {}

for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        # ignore folders that start with underscore
        val_file = list(sorted(p.glob(f'**/{method}/{dataset}/**/val*.json')))
        test_file = list(sorted(p.glob(f'**/{method}/{dataset}/**/test*.json')))
        train_file = list(sorted(p.glob(f'**/{method}/{dataset}/**/train*.json')))

        assert len(val_file) == len(test_file)

        data_val = load_files(val_file)
        data_test = load_files(test_file)

        if len(val_file) == 0:
            continue
        elif len(val_file) == 1:
            data_val = data_val[0]
            data_test = data_test[0]
        elif len(val_file) > 1:
            compare_settings(data_val)
            compare_settings(data_test)

        result_i = {}
        if method in generating_pareto_front:

            
            s = 'start_0'
            if isinstance(data_val, list):
                result_i['num_parameters'] = data_val[0]['num_parameters']
                # we have multiple runs of the same method
                early_stop_epoch = []
                val_scores = []
                test_scores = []
                val_hv = []
                test_hv = []
                training_time = []
                for val_run, test_run in zip(data_val, data_test):
                    e = "epoch_{}".format(get_early_stop(val_run[s], key=stop_key[method]))
                    val_results = val_run[s][e]
                    test_results = test_run[s][e]

                    early_stop_epoch.append(int(e.replace('epoch_', '')))
                    val_scores.append(val_results['scores'])
                    test_scores.append(test_results['scores'])
                    val_hv.append(val_results['hv'])
                    test_hv.append(test_results['hv'])
                    training_time.append(test_results['training_time_so_far'])
                
                result_i['early_stop_epoch'] = mean_and_std(early_stop_epoch)
                result_i['val_scores'] = mean_and_std(val_scores)
                result_i['test_scores'] = mean_and_std(test_scores)
                result_i['val_hv'] = mean_and_std(val_hv)
                result_i['test_hv'] = mean_and_std(test_hv)
                result_i['training_time'] = mean_and_std(training_time)
            else:
                # we have just a single run of the method
                assert len([True for k in data_val.keys() if 'start_' in k]) == 1
                result_i['num_parameters'] = data_val['num_parameters']
                e = "epoch_{}".format(get_early_stop(data_val[s], key=stop_key[method]))
                val_results = data_val[s][e]
                test_results = data_test[s][e]

                result_i['early_stop_epoch'] = int(e.replace('epoch_', ''))
                result_i['val_scores'] = val_results['scores']
                result_i['test_scores'] = test_results['scores']
                result_i['val_hv'] = val_results['hv']
                result_i['test_hv'] = test_results['hv']
                result_i['training_time'] = test_results['training_time_so_far']

        else:

            if isinstance(data_val, list):
                early_stop_epoch = []
                val_scores = []
                test_scores = []
                val_hv = []
                test_hv = []
                training_time = []
                for val_run, test_run in zip(data_val, data_test):
                    result_i = process_non_pareto_front(val_run, test_run)
                    early_stop_epoch.append(result_i['early_stop_epoch'])
                    val_scores.append(result_i['val_scores'])
                    test_scores.append(result_i['test_scores'])
                    val_hv.append(result_i['val_hv'])
                    test_hv.append(result_i['test_hv'])
                    training_time.append(result_i['training_time'])
                
                result_i['num_parameters'] = data_val[0]['num_parameters']
                result_i['early_stop_epoch'] = mean_and_std(early_stop_epoch)
                result_i['val_scores'] = mean_and_std(val_scores)
                result_i['test_scores'] = mean_and_std(test_scores)
                result_i['val_hv'] = mean_and_std(val_hv)
                result_i['test_hv'] = mean_and_std(test_hv)
                result_i['training_time'] = mean_and_std(training_time)
            else:
                result_i['num_parameters'] = data_val['num_parameters']
                result_i = process_non_pareto_front(data_val, data_test)



        results[dataset][method] = result_i


with open('results.json', 'w') as outfile:
    json.dump(results, outfile)

#
# Generate the plots and tables
#

def plot_row(datasets, methods, limits, prefix):
    assert len(datasets) == 3
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for j, dataset in enumerate(datasets):
        if dataset not in results:
            continue
        ax = axes[j]
        for method in methods:
            if method not in results[dataset]:
                continue
            r = results[dataset][method]
            # we take the mean only
            s = np.array(r['test_scores'][0]) if isinstance(r['test_scores'], tuple) else np.array(r['test_scores'])
            if method == 'SingleTask':
                s = np.squeeze(s)
                ax.axvline(x=s[0], color=colors[method], linestyle='-.')
                ax.axhline(y=s[1], color=colors[method], linestyle='-.', label="{}".format(method_names[method]))
            else:
                ax.plot(
                    s[:, 0], 
                    s[:, 1], 
                    color=colors[method],
                    marker=markers[method],
                    linestyle='--' if method != 'ParetoMTL' else ' ',
                    label="{}".format(method_names[method])
                )
                
                if dataset == 'multi_fashion' and method == 'cosmos_ln' and prefix == 'cosmos':
                    axins = zoomed_inset_axes(ax, 7, loc='upper right')
                    axins.plot(
                        s[:, 0], 
                        s[:, 1], 
                        color=colors[method],
                        marker=markers[method],
                        linestyle='--' if method != 'ParetoMTL' else '',
                        label="{}".format(method_names[method])
                    )
                    axins.set_xlim(.4658, .492)
                    axins.set_ylim(.488, .513)
                    axins.set_yticklabels([])
                    axins.set_xticklabels([])
                    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        lim = limits[dataset]
        ax.set_xlim(left=lim[0], right=lim[1])
        ax.set_ylim(bottom=lim[2], top=lim[3])
        ax.set_title(titles[dataset])
        ax.set_xlabel(ax_lables[dataset][0])
        if j==0:
            ax.set_ylabel(ax_lables[dataset][1])


        if j==2:
            ax.legend(loc='upper right')
    plt.subplots_adjust(wspace=.25)
    fig.savefig(prefix + '_' + '_'.join(datasets) + '.pdf', bbox_inches='tight')
    plt.close(fig)


datasets1 = ['adult', 'compas', 'credit']
methods1 = ['hyper_epo', 'hyper_ln', 'ParetoMTL', 'cosmos_ln']
datasets2 = ['multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
methods2 = ['SingleTask', 'cosmos_ln']

plot_row(datasets1, methods1, limits_baselines, prefix='baselines')
plot_row(datasets2, methods1, limits_baselines, prefix='baselines')

plot_row(datasets1, methods2, limits_baselines, prefix='cosmos')
plot_row(datasets2, methods2, limits_baselines, prefix='cosmos')


#
# generating the tables
#

def generate_table(datasets, methods, name):
    text = f"""
\\toprule
                & Hyper Vol. & Time (Sec) & \\# Params. \\\\ \\midrule"""
    for dataset in datasets:
        text += f"""
                & \\multicolumn{{3}}{{c}}{{\\bf {titles[dataset]}}} \\\\ \cmidrule{{2-4}}"""
        for method in methods:
            r = results[dataset][method]
            text += f"""
{method_names[method]}    & {r['test_hv'][0]:.2f} $\pm$ {r['test_hv'][1]:.2f}        & {r['training_time'][0]:,.0f}          &  {r['num_parameters']//1000:,d}k \\\\ """

    text += f"""
\\bottomrule"""
    
    with open(f'results_{name}.txt', 'w') as f:
        f.writelines(text)




datasets1 = ['adult', 'compas', 'credit']
generate_table(datasets1, methods, name='fairness')

datasets2 = ['multi_mnist', 'multi_fashion', 'multi_fashion_mnist']
generate_table(datasets2, methods, name='mnist')
