from .utils import dict_mean2df, data_prep
import seaborn as sns

from Diphone.config import (
        MODEL_DIR,
        FIG_DIR
        )

## load in data
perf_roi_all = data_prep()

## convert dict to df to prep for plotting
## also save the df for R stats computation
perf_roi_all_mean_df = dict_mean2df(perf_roi_all)
perf_roi_all_mean_df.to_csv("{MODEL_DIR}/df_roi_thirdSem.csv")

## make plots using seaborn as a double check
## non-thresholded result

colors = ["#009716","#DB57B2"]
thirdSemPalette = sns.set_palette(sns.color_palette(colors))

sns.catplot(x="rois", y="performance",
                hue="models", order=["ACunique", "STG", "STS", "LTCunique", "Broca", "wholeBrain"],
                 data=perf_roi_all_mean_df, kind="box",
                palette=thirdSemPalette, 
                 height=4, aspect=1.5);

ax = sns.stripplot(x="rois", y="performance",
                hue="models", order=["ACunique", "STG", "STS", "LTCunique", "Broca", "wholeBrain"],
                 data=perf_roi_all_mean_df, 
                palette=thirdSemPalette, dodge=True)


ax.get_legend().remove()

## plot for different phn types
perf_roi_phone_mean_df = dict_mean2df(perf_roi_all, model_list=["single", "diphone", "triphoneVP"])
perf_roi_phone_mean_df.to_csv("{MODEL_DIR}/df_roi_phns.csv")


## convert df to dict to prep for custom plotting

models_thirdSem_plot = ["thirdOrder", "semantic"]
metrics_thirdSem = plot_prep(models_thirdSem_plot, rois, subjects, perf_roi_all_mean_df)
models_phns_plot = ["single", "diphone", "triphoneVP"]
metrics_phn = plot_prep(models_phns_plot, rois, subjects, perf_roi_phone_mean_df)


fig, ax = plt.subplots(figsize=(8, 6))
ticks = rois
x = np.array(range(len(ticks)))
metrics = metrics_thirdSem

draw_plot(metrics[0].T, 'k', '#009716', x*2.0+0.2)
draw_plot(metrics[1].T, 'k', '#DB57B2', x*2.0+0.2*3)

                
for m in range(len(models_thirdSem_plot)):
    for r in range(len(rois)):
        plt.scatter(np.ones(len(subjects))*2*r+0.4*(m+1)-0.2, metrics[m][r], color = "grey", alpha = 0.3, s = 30)

for s in range(len(subjects)):
    for r in range(len(rois)):
        plt.plot([2*r+0.4-0.2, 2*r+0.4*2-0.2],[metrics[0][r][s], metrics[1][r][s]], '-o', color = "grey", alpha = 0.3) 


plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.savefig("{FIG_DIR}/perf_roi_thirdSem.eps", format='eps')

fig, ax = plt.subplots(figsize=(8, 6))
ticks = rois
x = np.array(range(len(ticks)))
metrics = metrics_phn


draw_plot(metrics[0].T, 'k', '#91DB57', x*2.0+0.2)
draw_plot(metrics[1].T, 'k', '#57D3DB', x*2.0+0.2*3)
draw_plot(metrics[2].T, 'k', '#DBC257', x*2.0+0.2*5)
                
for m in range(len(models_phns_plot)):
    for r in range(len(rois)):
        plt.scatter(np.ones(len(subjects))*2*r+0.4*(m+1)-0.2, metrics[m][r], color = "grey", alpha = 0.3, s = 30)

for s in range(len(subjects)):
    for r in range(len(rois)):
        plt.plot([2*r+0.4-0.2, 2*r+0.4*2-0.2],[metrics[0][r][s], metrics[1][r][s]], '-o', color = "grey", alpha = 0.3) 
        plt.plot([2*r+0.4*2-0.2, 2*r+0.4*3-0.2],[metrics[1][r][s], metrics[2][r][s]], '-o', color = "grey", alpha = 0.3)


plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.savefig("{FIG_DIR}/perf_roi_phns.eps", format='eps')



