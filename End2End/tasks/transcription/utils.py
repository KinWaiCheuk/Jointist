import pathlib
import pickle
import os
import numpy as np
import mir_eval
import time
import tqdm
import matplotlib.pyplot as plt
import collections
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def evaluate_F1(pred_path, label_path, ignore_drums=True):
    """
    Calcuate piece-wise F1 score for both note and note with offset metrics.
    notewise_dict will be a nested dictionary with the following keys:
    {'note':
        {filename: 
            {inst_name:
                {'precision': int
                 'recall': int
                 'f1': int
                 }
            },
            .
            .
            .
        .
        .
        .
        },
     'note_w_off':
        {filename: 
            {inst_name:
                {'precision': int
                 'recall': int
                 'f1': int
                 },
            },
            .
            .
            .
        .
        .
        .
        },
    }
    
    """
    pred_pkl_files = pathlib.Path(pred_path)
    pred_pkl_files = pred_pkl_files.glob('*.pkl')
    start = time.time()
    notewise_dict = {}
    notewise_dict['note'] = {}
    notewise_dict['note_w_off'] = {}
    for pkl_file in tqdm.tqdm(pred_pkl_files):
        note_events = pickle.load(open(os.path.join(label_path, pkl_file.name), 'rb'))
        transcribed_dict = pickle.load(open(pkl_file,'rb'))
        unique_plugin_names = sorted(list(set([note_event['plugin_name'] for note_event in note_events])))    

        notewise_dict['note'][pkl_file.name] = {}
        notewise_dict['note_w_off'][pkl_file.name] = {}        
        for plugin_name in unique_plugin_names:           
            ref_on_off_pairs = []
            ref_pitches = []

            for note_event in note_events:
                if note_event['plugin_name'] == plugin_name:
                    ref_on_off_pairs.append([note_event['start'], note_event['end']])
                    ref_pitches.append(note_event['pitch'])

            est_on_off_pairs = []
            est_pitches = []
            
            if transcribed_dict.get(plugin_name)==None:
                est_on_off_pairs = []
                est_pitches = []
            else:
                for note_event in transcribed_dict[plugin_name]:
                    est_on_off_pairs.append([note_event['onset_time'], note_event['offset_time']])
                    est_pitches.append(note_event['midi_note'])

            # from IPython import embed; embed(using=False); os._exit(0)

            ref_on_off_pairs = np.array(ref_on_off_pairs)
            ref_pitches = np.array(ref_pitches)
            est_on_off_pairs = np.array(est_on_off_pairs)
            est_pitches = np.array(est_pitches)
            
            if est_on_off_pairs.shape[0]!=0:
                (note_precision, note_recall, note_f1, _,) = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_on_off_pairs,
                    ref_pitches=ref_pitches,
                    est_intervals=est_on_off_pairs,
                    est_pitches=est_pitches,
                    onset_tolerance=0.05,
                    offset_ratio=None,
                )
                if plugin_name!='Drums': # only calculate offset metric for non-drums
                    (note_woffset_precision, note_woffset_recall, note_woffset_f1, _,) = mir_eval.transcription.precision_recall_f1_overlap(
                        ref_intervals=ref_on_off_pairs,
                        ref_pitches=ref_pitches,
                        est_intervals=est_on_off_pairs,
                        est_pitches=est_pitches,
                        onset_tolerance=0.05,
                        offset_ratio=0.2,
                    )
            else:
                print(f"empty pianoroll")
                note_precision = 0
                note_recall = 0
                note_f1 = 0     
                note_woffset_precision = 0
                note_woffset_recall = 0
                note_woffset_f1 = 0
            notewise_dict['note'][pkl_file.name][plugin_name] = {}                
            notewise_dict['note'][pkl_file.name][plugin_name] = {
                                                        'precision': note_precision,
                                                        'recall': note_recall,
                                                        'f1': note_f1
                                                        }
            if plugin_name!='Drums': # only calculate offset metric for non-drums
                notewise_dict['note_w_off'][pkl_file.name][plugin_name] = {}
                notewise_dict['note_w_off'][pkl_file.name][plugin_name] = {
                                                            'precision': note_woffset_precision,
                                                            'recall': note_woffset_recall,
                                                            'f1': note_woffset_f1
                                                            }            
    return notewise_dict

def evaluate_flat_F1(pred_path):
    """
    Calcuate piece-wise F1 score for both note and note with offset metrics.
    notewise_dict will be a nested dictionary with the following keys:
    {'note':
        {filename: 
            {inst_name:
                {'precision': int
                 'recall': int
                 'f1': int
                 }
            },
            .
            .
            .
        .
        .
        .
        },
     'note_w_off':
        {filename: 
            {inst_name:
                {'precision': int
                 'recall': int
                 'f1': int
                 },
            },
            .
            .
            .
        .
        .
        .
        },
    }
    
    """
    pred_pkl_files = pathlib.Path(pred_path)
    pred_pkl_files = pred_pkl_files.glob('*.flat_pkl')
    notewise_dict = {}
    notewise_dict['note'] = {}
    notewise_dict['note_w_off'] = {}
    for pkl_file in tqdm.tqdm(pred_pkl_files):
        note_events = pickle.load(open(f"{os.path.splitext(pkl_file)[0]}.label_pkl", 'rb'))
        transcribed_dict = pickle.load(open(pkl_file,'rb'))
        plugin_name = list(transcribed_dict.keys())[0]
#         print(f'{plugin_name}')

        notewise_dict['note'][pkl_file.name] = {}
        notewise_dict['note_w_off'][pkl_file.name] = {}        

        ref_on_off_pairs = []
        ref_pitches = []

        for note_event in note_events[plugin_name]:
            ref_on_off_pairs.append([note_event['onset_time'], note_event['offset_time']])
            ref_pitches.append(note_event['midi_note'])

        est_on_off_pairs = []
        est_pitches = []

        for note_event in transcribed_dict[plugin_name]:
            est_on_off_pairs.append([note_event['onset_time'], note_event['offset_time']])
            est_pitches.append(note_event['midi_note'])

        # from IPython import embed; embed(using=False); os._exit(0)

        ref_on_off_pairs = np.array(ref_on_off_pairs)
        ref_pitches = np.array(ref_pitches)
        est_on_off_pairs = np.array(est_on_off_pairs)
        est_pitches = np.array(est_pitches)

        if est_on_off_pairs.shape[0]!=0:
            (note_precision, note_recall, note_f1, _,) = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_on_off_pairs,
                ref_pitches=ref_pitches,
                est_intervals=est_on_off_pairs,
                est_pitches=est_pitches,
                onset_tolerance=0.05,
                offset_ratio=None,
            )

            (note_woffset_precision, note_woffset_recall, note_woffset_f1, _,) = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_on_off_pairs,
                ref_pitches=ref_pitches,
                est_intervals=est_on_off_pairs,
                est_pitches=est_pitches,
                onset_tolerance=0.05,
                offset_ratio=0.2,
            )
        else:
            print(f"empty pianoroll")
            note_precision = 0
            note_recall = 0
            note_f1 = 0     
            note_woffset_precision = 0
            note_woffset_recall = 0
            note_woffset_f1 = 0
        notewise_dict['note'][pkl_file.name] = {
                                                    'precision': note_precision,
                                                    'recall': note_recall,
                                                    'f1': note_f1
                                                    }
        notewise_dict['note_w_off'][pkl_file.name] = {
                                                    'precision': note_woffset_precision,
                                                    'recall': note_woffset_recall,
                                                    'f1': note_woffset_f1
                                                    }            
    return notewise_dict

def get_flat_average(score_dict, key_name):
    """
    In the case of notewise scores
    score_dict: {'note':{
                    'Track01876.flat_pkl': {
                        'precision': 0.5455840455840456,
                        'recall': 0.7323135755258127,
                        'f1': 0.6253061224489797},
                    'Track01877.flat_pkl': {
                        'precision': 0.2944965948831217,
                        'recall': 0.823045267489712,
                        'f1': 0.43378066964890877}
                    }
                 'note_w_off':{
                    'Track01876.flat_pkl': {
                        'precision': 0.24465811965811965,
                        'recall': 0.3283938814531549,
                        'f1': 0.2804081632653061},
                    'Track01877.flat_pkl': {
                        'precision': 0.13178722621019695,
                        'recall': 0.3683127572016461,
                        'f1': 0.1941168496678867}
                        }                         
                 }

    In the case of framewise score
    score_dict: {'flat_framewise':{
                    'Track01876': {
                        'precision': 0.5455840455840456,
                        'recall': 0.7323135755258127,
                        'f1': 0.6253061224489797},
                    'Track01877': {
                        'precision': 0.2944965948831217,
                        'recall': 0.823045267489712,
                        'f1': 0.43378066964890877}
                    }                       
                 }            
    """

    summary = [[scores['precision'], scores['recall'], scores['f1']] for trackname, scores in score_dict[key_name].items()]
    flat_p_mean, flat_r_mean, flat_f1_mean = np.mean(summary, axis=0)
    return flat_p_mean, flat_r_mean, flat_f1_mean

def piecewise_evaluation(note_dict, metric_type='note', metric='f1'):
#     note_dict = pickle.load(open(os.path.join(dict_path, 'notewise_dict.pkl'), 'rb'))    
    piecewise_metric = []
    for name, contents in note_dict[metric_type].items():
        accumulation = []
        for inst, scores in contents.items():
            accumulation.append(scores[metric])
        piecewise_metric.append(np.mean(accumulation))

    return piecewise_metric     

def calculate_intrumentwise_statistics(notewise_dict, key):
    instrument_wise_precision = {}
    instrument_wise_recall = {}
    instrument_wise_F1 = {}
    for piece, instruments in notewise_dict[key].items():
        for i in instruments:
            if i in instrument_wise_precision:
                instrument_wise_precision[i].append(instruments[i]['precision'])
                instrument_wise_recall[i].append(instruments[i]['recall'])
                instrument_wise_F1[i].append(instruments[i]['f1'])
            else:
                instrument_wise_precision[i] = [instruments[i]['precision']]
                instrument_wise_recall[i] = [instruments[i]['recall']]
                instrument_wise_F1[i] = [instruments[i]['f1']]    
                
    return instrument_wise_precision, instrument_wise_recall, instrument_wise_F1

def calculate_mean_std(instrument_wise_scores):
    stat_mean = {}
    stat_std = {}
    for i, j in instrument_wise_scores.items():
        stat_mean[i] = np.mean(j)
        stat_std[i] = np.std(j)
        
    return stat_mean, stat_std


def barplot(stat_mean, title="Untitles", figsize=(4,24)):
#     stat_mean = collections.OrderedDict(sorted(stat_mean.items()))    
    stat_mean = {k: v for k, v in sorted(stat_mean.items(), key=lambda item: item[1])}
    fig, ax = plt.subplots(1,1, figsize=figsize)
    xlabels = list(stat_mean.keys())
    values = list(stat_mean.values())
    ax.barh(xlabels, values)
    global_mean = sum(stat_mean.values())/len(stat_mean.values())
    ax.vlines(global_mean, 0, len(stat_mean), 'r')    
    ax.tick_params(labeltop=True, labelright=False)
#     ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_ylim([-1,len(xlabels)])
    ax.set_xlim([0,1])
    ax.set_title(title)
#     text_kwargs = dict(fontsize=12, color='C1', transform=ax.transAxes)
#     ax.text(0, 1.03, f'mean={global_mean}', **text_kwargs)
    ax.grid(axis='x')
    ax.grid(b=True, which='minor', linestyle='--')    
    fig.savefig(f'{title}.png', bbox_inches='tight')
#     ax.invert_yaxis()  # labels read top-to-bottom

    return global_mean, fig
