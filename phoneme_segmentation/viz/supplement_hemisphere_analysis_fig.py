import numpy as np
from .utils import data_prep, dict_mean2df, dict_mean2df_wta, wta_prep, add_countAll, plot_prep, dict_mean2df_hemi, draw_plot

from Diphone.config import (
        MODEL_DIR,
        FIG_DIR
        )

import seaborn as sns
import matplotlib.pyplot as plt
import cortex


## data loading

perf_roi_all = data_prep(return_hemi=True)
perf_roi_all_mean_df = dict_mean2df_hemi(perf_roi_all)


perf_roi_all_mean_df.to_csv("{MODEL_DIR}/df_perfMean_thirdSem_hemi.csv")
perf_roi_phone_mean_df = dict_mean2df_hemi(perf_roi_all, model_list=["single", "diphone", "triphone"])

perf_roi_phone_mean_df.to_csv("{MODEL_DIR}/df_perfMean_phns_hemi.csv")

## seaborn quick plot as sanity check

## non-thresholded result

colors = ["#DB57B2", "#009716"]
thirdSemPalette = sns.set_palette(sns.color_palette(colors))

sns.catplot(x="rois", y="performance",
                hue="models", order=["ACunique", "STG", "STS", "LTCunique", "Broca", "wholeBrain"],
                 col="hemis",
            data=perf_roi_all_mean_df, kind="box",
                palette=thirdSemPalette,
                 height=4, aspect=1.5);

colors = ["#91DB57","#57D3DB", "#DBC257"]
wtaPalette_phns = sns.set_palette(sns.color_palette(colors))

sns.catplot(x="rois", y="performance",
                hue="models", order=["ACunique", "STG", "STS", "LTCunique", "Broca", "wholeBrain"],
            col="hemis",     
            data=perf_roi_phone_mean_df, kind="box", 
            hue_order = ["single", "diphone", "triphone"],
                palette=wtaPalette_phns, 
                 height=4, aspect=1.5);



## prep for custom plotting 

perf_roi_phone_mean_df_left = perf_roi_phone_mean_df[perf_roi_phone_mean_df["hemis"] == "left"]
perf_roi_phone_mean_df_right = perf_roi_phone_mean_df[perf_roi_phone_mean_df["hemis"] == "right"]

perf_roi_all_mean_df_left = perf_roi_all_mean_df[perf_roi_all_mean_df["hemis"] == "left"]
perf_roi_all_mean_df_right = perf_roi_all_mean_df[perf_roi_all_mean_df["hemis"] == "right"]

rois_lists = ["WB", "ACunique", "STG", "STS", "LTCunique", "Broca"]
roi_lists_name = ["Cortex", "PAC", "STG", "STS", "LTC", "Broca"]

## custom plots
fig, ax = plt.subplots(2, 2, figsize=(15, 10))


models_phns_plot = ["single", "diphone", "triphone"]

metrics_phn_left = plot_prep(models_phns_plot, rois_lists, subjects, perf_roi_phone_mean_df_left)
metrics_phn_right = plot_prep(models_phns_plot, rois_lists, subjects, perf_roi_phone_mean_df_right)

ticks = rois_lists
x = np.array(range(len(ticks)))
metrics = metrics_phn_left

draw_plot(metrics[0].T, 'k', '#91DB57', x*2.0+0.2, ax[1, 0])
draw_plot(metrics[1].T, 'k', '#57D3DB', x*2.0+0.2*3, ax[1, 0])
draw_plot(metrics[2].T, 'k', '#DBC257', x*2.0+0.2*5, ax[1, 0])

for m in range(len(models_phns_plot)):
    for r in range(len(rois_lists)):
        ax[1, 0].scatter(np.ones(len(subjects))*2*r+0.4*(m+1)-0.2, metrics[m][r], color = "grey", alpha = 0.3, s = 30)

for s in range(len(subjects)):
    for r in range(len(rois_lists)):
        ax[1, 0].plot([2*r+0.4-0.2, 2*r+0.4*2-0.2],[metrics[0][r][s], metrics[1][r][s]], '-o', color = "grey", alpha = 0.3)
        ax[1, 0].plot([2*r+0.4*2-0.2, 2*r+0.4*3-0.2],[metrics[1][r][s], metrics[2][r][s]], '-o', color = "grey", alpha = 0.3)

for l in list(range(2, len(ticks) * 2+2, 2)):
    ax[1, 0].vlines(l, ymin = -0.005, ymax =0.041, linestyle='--', alpha=0.3)
for r in list(np.arange(0, 0.041, 0.01)):
    ax[1, 0].hlines(r, xmin = 0, xmax= 13,linestyle='--', alpha=0.3)
ax[1, 0].set_xticks(range(1, len(ticks) * 2+1, 2), ticks)
ax[1, 0].set_xticklabels(roi_lists_name);
ax[1, 0].set_yticks(np.arange(0, 0.041, 0.01))
ax[1, 0].set_yticklabels(np.arange(0, 0.041, 0.01))


metrics = metrics_phn_right

draw_plot(metrics[0].T, 'k', '#91DB57', x*2.0+0.2, ax[1, 1])
draw_plot(metrics[1].T, 'k', '#57D3DB', x*2.0+0.2*3, ax[1, 1])
draw_plot(metrics[2].T, 'k', '#DBC257', x*2.0+0.2*5, ax[1, 1])

for m in range(len(models_phns_plot)):
    for r in range(len(rois_lists)):
        ax[1, 1].scatter(np.ones(len(subjects))*2*r+0.4*(m+1)-0.2, metrics[m][r], color = "grey", alpha = 0.3, s = 30)

for s in range(len(subjects)):
    for r in range(len(rois_lists)):
        ax[1, 1].plot([2*r+0.4-0.2, 2*r+0.4*2-0.2],[metrics[0][r][s], metrics[1][r][s]], '-o', color = "grey", alpha = 0.3)
        ax[1, 1].plot([2*r+0.4*2-0.2, 2*r+0.4*3-0.2],[metrics[1][r][s], metrics[2][r][s]], '-o', color = "grey", alpha = 0.3)

for l in list(range(2, len(ticks) * 2+2, 2)):
    ax[1, 1].vlines(l, ymin = -0.005, ymax =0.041, linestyle='--', alpha=0.3)
for r in list(np.arange(0, 0.041, 0.01)):
    ax[1, 1].hlines(r, xmin = 0, xmax= 13,linestyle='--', alpha=0.3)
ax[1, 1].set_xticks(range(1, len(ticks) * 2+1, 2), ticks)
ax[1, 1].set_xticklabels(roi_lists_name);
ax[1, 1].set_yticks(np.arange(0, 0.041, 0.01))
ax[1, 1].set_yticklabels(np.arange(0, 0.041, 0.01))

## custom plots
models_thirdSem_plot = ["thirdOrder", "semantic"]

metrics_thirdSem = plot_prep(models_thirdSem_plot, rois_lists, subjects, perf_roi_all_mean_df_left)


ticks = rois_lists
x = np.array(range(len(ticks)))
metrics = metrics_thirdSem


draw_plot(metrics[0].T, 'k', '#009716', x*2.0+0.2, ax[0, 0])
draw_plot(metrics[1].T, 'k', '#DB57B2', x*2.0+0.2*3, ax[0, 0])

                
for m in range(len(models_thirdSem_plot)):
    for r in range(len(rois_lists)):
        ax[0, 0].scatter(np.ones(len(subjects))*2*r+0.4*(m+1)-0.2, metrics[m][r], color = "grey", alpha = 0.3, s = 30)

for s in range(len(subjects)):
    for r in range(len(rois_lists)):
        ax[0, 0].plot([2*r+0.4-0.2, 2*r+0.4*2-0.2],[metrics[0][r][s], metrics[1][r][s]], '-o', color = "grey", alpha = 0.3) 

for l in list(range(2, len(ticks) * 2+2, 2)):
    ax[0, 0].vlines(l, ymin = -0.005, ymax =0.061, linestyle='--', alpha=0.3)
for r in list(np.arange(0, 0.061, 0.02)):
    ax[0, 0].hlines(r, xmin = 0, xmax= 13,linestyle='--', alpha=0.3)
ax[0, 0].set_xticks(range(1, len(ticks) * 2+1, 2), ticks)
ax[0, 0].set_xticklabels(roi_lists_name);
ax[0, 0].set_yticks(np.arange(0, 0.061, 0.02))
ax[0, 0].set_yticklabels(np.arange(0, 0.061, 0.02))

metrics_thirdSem = plot_prep(models_thirdSem_plot, rois_lists, subjects, perf_roi_all_mean_df_right)

ticks = rois_lists
x = np.array(range(len(ticks)))
metrics = metrics_thirdSem


draw_plot(metrics[0].T, 'k', '#009716', x*2.0+0.2, ax[0, 1])
draw_plot(metrics[1].T, 'k', '#DB57B2', x*2.0+0.2*3, ax[0, 1])

                
for m in range(len(models_thirdSem_plot)):
    for r in range(len(rois_lists)):
        ax[0, 1].scatter(np.ones(len(subjects))*2*r+0.4*(m+1)-0.2, metrics[m][r], color = "grey", alpha = 0.3, s = 30)

for s in range(len(subjects)):
    for r in range(len(rois_lists)):
        ax[0, 1].plot([2*r+0.4-0.2, 2*r+0.4*2-0.2],[metrics[0][r][s], metrics[1][r][s]], '-o', color = "grey", alpha = 0.3) 

for l in list(range(2, len(ticks) * 2+2, 2)):
    ax[0, 1].vlines(l, ymin = -0.005, ymax =0.061, linestyle='--', alpha=0.3)
for r in list(np.arange(0, 0.061, 0.02)):
    ax[0, 1].hlines(r, xmin = 0, xmax= 13,linestyle='--', alpha=0.3)
ax[0, 1].set_xticks(range(1, len(ticks) * 2+1, 2), ticks)
ax[0, 1].set_xticklabels(roi_lists_name);
ax[0, 1].set_yticks(np.arange(0, 0.061, 0.02))
ax[0, 1].set_yticklabels(np.arange(0, 0.061, 0.02))
   
ax[0, 0].set_ylabel("Mean Performance")
ax[1, 0].set_ylabel("Mean Performance")
ax[1, 0].set_xlabel("Region of Interest")
ax[1, 1].set_xlabel("Region of Interest")

plt.savefig("{FIG_DIR}/perf_roi_hemi_sum.png")



