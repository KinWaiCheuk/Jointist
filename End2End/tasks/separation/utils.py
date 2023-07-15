import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np

def calculate_sdr(ref, est):
    assert ref.dim()==est.dim(), f"ref {ref.shape} has a different size than est {est.shape}"
    
    s_true = ref
    s_artif = est - ref

    sdr = 10. * (
        torch.log10(torch.clip(torch.mean(s_true ** 2, 1), 1e-8, torch.inf)) \
        - torch.log10(torch.clip(torch.mean(s_artif ** 2, 1), 1e-8, torch.inf)))
    return sdr



def _append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
        
        
def barplot(stat_mean, title="Untitles", figsize=(4,24)):
#     stat_mean = collections.OrderedDict(sorted(stat_mean.items()))    
    stat_mean = {k: v for k, v in sorted(stat_mean.items(), key=lambda item: item[1])}
    fig, ax = plt.subplots(1,1, figsize=figsize)
    xlabels = list(stat_mean.keys())
    values = list(stat_mean.values())
    ax.barh(xlabels, values, color='cyan')
    global_mean = sum(stat_mean.values())/len(stat_mean.values())
    ax.vlines(global_mean, 0, len(stat_mean), 'r')    
    ax.tick_params(labeltop=True, labelright=False)
    ax.set_ylim([-1,len(xlabels)])
    ax.set_title(title)
    ax.grid(axis='x')
    ax.grid(b=True, which='minor', linestyle='--')    
    
    # move the left boundary to origin
    ax.spines['left'].set_position('zero')
    # turn off the RHS boundary
    ax.spines['right'].set_color('none')
    
    fig.savefig(f'{title}.png', bbox_inches='tight')

    return global_mean, fig        