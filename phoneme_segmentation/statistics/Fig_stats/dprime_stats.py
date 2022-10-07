import numpy as np
import pandas as pd


def dprime_calc(x, y):
    dprime = (np.nanmean(x)-np.nanmean(y))/np.std((x-y), ddof=1)
    return dprime

def dprime_manual_calc(x, y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    x_y_std = np.sum(np.array([np.var(x[i]-y[i]) for i in range(len(x))]))/(len(x)-1)
    
    dprime = (x_mean - y_mean)/x_y_std
    
    return dprime

def thirdSem_dprime_calc(df,
                        metrics="performance",
                        models_plot=["thirdOrder", "semantic"]):
    rois_all = np.unique(df["rois"])
    subjects_all = np.unique(df["subjects"])
    
    dprime_all = {}

    for r in rois_all:
        perf_third_tmp = []
        perf_sem_tmp = []
        for s_i, subj in enumerate(subjects_all):
            perf_third_tmp.append(df[(df["subjects"] == subj) &
                                     (df["rois"] == r) & 
                                     (df["models"] == models_plot[0])][metrics])

            perf_sem_tmp.append(df[(df["subjects"] == subj) &
                                 (df["rois"] == r) & 
                                 (df["models"] == models_plot[1])][metrics])

        print(np.array(perf_third_tmp)[:, 0].shape)
        print(np.array(perf_sem_tmp)[:, 0].shape)
        dprime_all[f"{r}"] = dprime_calc(np.array(perf_sem_tmp)[:, 0], np.array(perf_third_tmp)[:, 0])
        
    return dprime_all

def phns_dprime_calc(df,
                    metrics="performance",
                    models_plot=["single", "diphone", "triphone"]):
    rois_all = np.unique(df["rois"])
    subjects_all = np.unique(df["subjects"])
    
    dprime_all = {}

    for r in rois_all:
        perf_single_tmp = []
        perf_diphn_tmp = []
        perf_triphn_tmp = []
        for s_i, subj in enumerate(subjects_all):
            perf_single_tmp.append(df[(df["subjects"] == subj) &
                                 (df["rois"] == r) & 
                                 (df["models"] == models_plot[0])][metrics])

            perf_diphn_tmp.append(df[(df["subjects"] == subj) &
                                 (df["rois"] == r) & 
                                 (df["models"] == models_plot[1])][metrics])
            perf_triphn_tmp.append(df[(df["subjects"] == subj) &
                                 (df["rois"] == r) & 
                                 (df["models"] == models_plot[2])][metrics])

        print(np.array(perf_single_tmp)[:, 0].shape)
        print(np.array(perf_diphn_tmp)[:, 0].shape)
        print(np.array(perf_triphn_tmp)[:, 0].shape)
        dprime_all[f"{r}_singleDiphn"] = dprime_calc(np.array(perf_diphn_tmp)[:, 0], np.array(perf_single_tmp)[:, 0])
        dprime_all[f"{r}_DiphnTriphn"] = dprime_calc(np.array(perf_diphn_tmp)[:, 0], np.array(perf_triphn_tmp)[:, 0])
        dprime_all[f"{r}"] = np.mean((dprime_all[f"{r}_singleDiphn"], dprime_all[f"{r}_DiphnTriphn"]))

    return dprime_all


