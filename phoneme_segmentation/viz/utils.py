import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables

from Diphone.config import (
        SUBJECTS_ALL,
        MODELS_ANALYSIS,
        ROIS,
        MODEL_DIR
        )

def data_prep(return_hemi=False,
            threshold = 10E-16):
        perf_roi_all = {}
        for subj in SUBJECTS_ALL:
            for m in MODELS_ANALYSIS:
                for r in ROIS:
                    print("loading data for %s %s %s"%(subj, m, r))
                    if r == "wholeBrain":
                        perf_f = tables.open_file(f"{MODEL_DIR}/{subj}_{m}.hf5")
                        perf_tmp = perf_f.root[r].read()
                        perf_roi_all["%s_%s_%s"%(subj, m, r)] = perf_tmp

                    else:
                        tmp_left = perf_f.root[f"{r}_left"].read()
                        tmp_right = perf_f.root[f"{r}_right"].read() 
                        if return_hemi == False:
                            perf_roi_all["%s_%s_%s"%(subj, m, r)] = np.concatenate((tmp_left, tmp_right), axis = 0)
                        else:
                            perf_roi_all["%s_%s_%s_left"%(subj, m, r)] = tmp_left
                            perf_roi_all["%s_%s_%s_right"%(subj, m, r)] = tmp_right

        return perf_roi_all

def wta_prep(perf_all):
    ind_max = np.zeros(perf_all.shape[0])
    max_value = np.zeros(perf_all.shape[0])
    for n_i, n in enumerate(perf_all):
        if np.sum(np.isnan(n))== perf_all.shape[-1]:
            ind_max[n_i] = np.nan
            max_value[n_i] = -100
        else:
            ind_max[n_i] = np.argmax(np.nan_to_num(n))+1
            max_value[n_i] = np.max(np.nan_to_num(n))

    return ind_max, max_value

def dict2df(input_dict,
                model_list=["thirdOrder", "semantic"]):

    subj_df_prep = []
    model_df_prep = []
    roi_df_prep = []
    perf_df_prep = []

    for k, value in input_dict.items():

        subj, m, r = k.split("_")

        if m in model_list:
            for v in np.nan_to_num(value):
                subj_df_prep.append(subj)
                model_df_prep.append(m)
                roi_df_prep.append(r)
                perf_df_prep.append(v)

    df = pd.DataFrame({"subjects": subj_df_prep,
                   "models": model_df_prep,
                    "rois": roi_df_prep,
                   "performance":perf_df_prep})

    return df

def dict_mean2df(input_dict,
                model_list=["thirdOrder", "semantic"]):
    
    subj_df_prep = []
    model_df_prep = []
    roi_df_prep = []
    perf_df_prep = []
    
    for k, v in input_dict.items():
        
        subj, m, r = k.split("_")
        
        if m in model_list:
            subj_df_prep.append(subj)
            model_df_prep.append(m)
            roi_df_prep.append(r)
            perf_df_prep.append(np.nanmean(v))

    df = pd.DataFrame({"subjects": subj_df_prep,
                   "models": model_df_prep,
                    "rois": roi_df_prep,
                   "performance":perf_df_prep})

    return df
   

def dict_mean2df_hemi(input_dict,
                model_list=["thirdOrder", "semantic"]):

    subj_df_prep = []
    model_df_prep = []
    roi_df_prep = []
    perf_df_prep = []
    hemi_df_prep = []

    for k, v in input_dict.items():

        subj, m, r, h = k.split("_")

        if m in model_list:
            subj_df_prep.append(subj)
            model_df_prep.append(m)
            roi_df_prep.append(r)
            perf_df_prep.append(np.nanmean(v))
            hemi_df_prep.append(h)	    

    df = pd.DataFrame({"subjects": subj_df_prep,
                   "models": model_df_prep,
                    "rois": roi_df_prep,
                    "hemis": hemi_df_prep,
                   "performance":perf_df_prep})

    return df 

def dict_mean2df_wta(input_count_dict,
                     input_maxPerf_dict,
			models_dict={"1": "single",
                  "2": "diphone",
                  "3": "triphone",
                  "4": "semantic"}):
    
    
    subj_df_prep = []
    roi_df_prep = []
    count_df_prep = []
    model_df_prep = []
    perf_max_df_prep = []
    
    for k, v in input_count_dict.items():
        
        subj, r = k.split("_")
        
        for m in range(len(models_dict)):
            model_df_prep.append(models_dict["%s"%(m+1)])
            subj_df_prep.append(subj)
            roi_df_prep.append(r)
            count_df_prep.append(np.sum(v==(m+1)))
            model_perf_tmp = input_maxPerf_dict[k][np.where(v==(m+1))[0]]
            perf_max_df_prep.append(np.nanmean(model_perf_tmp))

    df = pd.DataFrame({"subjects": subj_df_prep,
                   "models": model_df_prep,
                    "rois": roi_df_prep,
                       "count": count_df_prep,
                   "performance":perf_max_df_prep})

    return df


def add_countAll(input_df,
                models=None):
    subjects = np.unique(input_df["subjects"])
    rois = np.unique([input_df["rois"]])
    if models == None:
        models = np.unique(input_df["models"])
    
    input_df["count_all"] = ""
    
    for subj in subjects:
        for r in rois:
            count_all_tmp = []
            for m in models:
                count_all_tmp.append(input_df[(input_df["subjects"] == subj) & (input_df["rois"] == r) & (input_df["models"] == m)]["count"])
            count_all_ind = np.sum(np.array(count_all_tmp))
            for m in models:
                input_df.loc[(input_df["subjects"] == subj) & (input_df["rois"] == r) & (input_df["models"] == m), "count_all"] = count_all_ind

    return input_df

def plot_prep(features, rois, subjects, df):
    metrics = []
    for f in features:
        data_subset = np.zeros((len(rois), len(subjects)))
        for r_i, r in enumerate(rois):
            for s_i, s in enumerate(subjects):
                data_tmp = np.array(df[(df["rois"]==r) &  (df["subjects"] == s) &  (df["models"]== f)]["performance"])[0]
                data_subset[r_i, s_i] = data_tmp
        metrics.append(data_subset)
        
    return metrics

def plot_prep_wta(features, rois, subjects, df):
    metrics_perf = []
    metrics_count = []
    for f in features:
        data_subset = np.zeros((len(rois), len(subjects)))
        count_subset = np.zeros((len(rois), len(subjects)))
        for r_i, r in enumerate(rois):
            for s_i, s in enumerate(subjects):
                data_tmp = np.array(df[(df["rois"]==r) &  (df["subjects"] == s) &  (df["models"]== f)]["performance"])[0]
                data_subset[r_i, s_i] = data_tmp
                
                count_tmp = np.array(df[(df["rois"]==r) &  (df["subjects"] == s) &  (df["models"]== f)]["count"])[0]
                count_subset[r_i, s_i] = count_tmp

        metrics_perf.append(data_subset)
        metrics_count.append(count_subset)
        
    return metrics_perf, metrics_count   

def draw_plot(data, median_color, fill_color, pos):
    bp = ax.boxplot(data, 0, '', positions = pos, patch_artist=True, widths=0.3)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=fill_color)

    plt.setp(bp["medians"], color=median_color)
    for whisker in bp['whiskers']:
        whisker.set_linestyle('-.')

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)     
        patch.set_alpha(0.5) 

def add_countAll(input_df):
    subjects = np.unique(input_df["subjects"])
    rois = np.unique([input_df["rois"]])
    models = np.unique(input_df["models"])
    
    input_df["count_all"] = ""
    
    for subj in subjects:
        for r in rois:
            count_all_tmp = []
            for m in models:
                count_all_tmp.append(input_df[(input_df["subjects"] == subj) & (input_df["rois"] == r) & (input_df["models"] == m)]["count"])
            count_all_ind = np.sum(np.array(count_all_tmp))
            for m in models:
                input_df.loc[(input_df["subjects"] == subj) & (input_df["rois"] == r) & (input_df["models"] == m), "count_all"] = count_all_ind

    return input_df
