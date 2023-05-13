from release_movies.release.utils.example import plot_flatmap_from_mapper

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline

def flatmap_generation(subject:str,
                        perf_data,
                        mapper_path:str):

    '''
    Performance can be found in Source_data_performance.csv
    '''
    mapper_file = f"{mapper_path}/{subject}_mappers.hdf"
    plot_flatmap_from_mapper(np.nan_to_num(voxel_data), mapper_file, vmin = 0, vmax = 0.1, cmap="hot")
    plt.show()

def dataPrep_boxplots(data_path:str,
                    sheet_name:str):
    data = pd.read_excel(open(f'{data_path}/Source_data_manuscript.xlsx', 'rb'), sheet_name=sheet_name, engine='openpyxl').dropna()

    perf_meanPlot_dict = {}

    for subj in np.unique(data["subjects"]):
        for m in np.unique(data["models"]):
            for r in np.unique(data["rois"]):
                data_tmp = np.array(data[(data["subjects"] == subj) & (data["models"] == m) & (data["rois"] == r)]["performance"])
                data_tmp[data_tmp<=0]=np.nan
                perf_meanPlot_dict[f"{subj}_{m}_{r}"] = data_tmp

    df_mean = dict_mean2df(perf_meanPlot_dict,
                        model_list=list(np.unique(data["models"])))

    return df_mean


def generate_boxplots(df,
                features:list=["single", "diphone", "triphone"],
                rois:list=["cortex", "PAC", "STG", "STS", "LTC", "Broca"],
                colors:list=["#91DB57","#57D3DB", "#DBC257"]):

    '''
    Function used to generate box plots for figure 4 and 6C

    The default parameters are used for figure 4
    For figure 6c:
        colors:list=["#009716", "#DB57B2"],
        features:list=["thirdOrder", "semantic"],
        rois are the same

    df: output from dataPrep_boxplots()
    '''

    cPalette = sns.set_palette(sns.color_palette(colors))

    ax=sns.catplot(x="rois", y="performance",
                    hue="models", order=rois,
                     data=df, kind="box",
                    hue_order = features,
                    palette=cPalette,
                     height=4, aspect=1.5);

    sns.stripplot(x="rois", y="performance",
                    hue="models", order=rois,
                     hue_order = features,
                    data=df, color="gray",
                     dodge=True)


def generate_fig5(w:float=0.3,
                df_sig,
                df_mean,
                stimRatio):

    perf_meanPlot_normed, perf_meanPlot_sig = dataPrep_fig5(df_sig, df_mean) 

    ## Fig5 zoomed in panel
    plt.figure()
    fig, ax1 = plt.subplots(figsize=(8, 6))
    draw_boxplot(perf_meanPlot_sig, ax1)
    ax2 = ax1.twinx()
    barlist = ax2.bar(np.arange(1,4)+w, np.array([stimRatio["shortWords"], stimRatio["bg"], stimRatio["res"]]), alpha = 0.3,hatch="/", width = w, fc="white",edgecolor='black')

    barlist[0].set_color("r")
    barlist[1].set_color("g")
    barlist[2].set_color("b")

    ax1.set_xticks([1+w/2,2+w/2,3+w/2])
    ax2.set_yticks(np.arange(0,1.1,0.2))
    ax1.set_xticklabels(('Short words', 'Word Beginning', 'Diphon Residual'))

    fig.tight_layout()
    for ax in [ax1, ax2]:
        for p in ["top", "right", "left", "bottom"]:
            ax.spines[p].set_visible(False)

    ## Fig5 main figure
    plt.figure()
    fig, ax1 = plt.subplots(figsize=(8, 6))
    draw_boxplot(perf_meanPlot_normed, ax1)

    fig.tight_layout()
    for ax in [ax1]:
        for p in ["top", "right", "left", "bottom"]:
            ax.spines[p].set_visible(False)

    ax1.set_xticklabels(('Short words', 'Word Beginning', 'Diphon Residual'))

    for s in range(len(SUBJECTS_ALL)):
            plt.plot([1, 2],[perf_meanPlot_normed[s, 0], perf_meanPlot_normed[s, 1]], '-o', color = "grey", alpha = 0.3)
            plt.plot([2, 3],[perf_meanPlot_normed[s, 1], perf_meanPlot_normed[s, 2]], '-o', color = "grey", alpha = 0.3)


def dataPrep_fig5(df,
                SUBJECTS_ALL:list=SUBJECTS_ALL
                MODELS_DIPHN_CATE:list=MODELS_DIPHN_CATE,
                ):
    perf_meanPlot_normed = np.zeros((len(SUBJECTS_ALL), len(MODELS_DIPHN_CATE)))
    perf_meanPlot_sig = np.zeros((len(SUBJECTS_ALL), len(MODELS_DIPHN_CATE)))

    for subj_i, subj in enumerate(SUBJECTS_ALL):
        for m_i, m in enumerate(MODELS_DIPHN_CATE):
            m_tmp = m.split("_")[-1]
            subj_tmp = f"SUBJ{subj_i:02}"
            data_tmp = np.array(df[(df["subjects"] == subj_tmp) & (df["models"] == m_tmp)]["performance"])
            perf_meanPlot_sig[subj_i, m_i] = np.nanmean(data_tmp)

            perf_sig_normed_tmp = np.divide(data_tmp, stimRatio[m_tmp]) 
            perf_meanPlot_normed[subj_i, m_i] = np.nanmean(perf_sig_normed_tmp)

    return perf_meanPlot_normed, perf_meanPlot_sig

def generate_fig7C(data_path:str,
                    sulcus:str,
                    smooth_factor:int=1000,
                    hemis:list=["left", "right"]
                    models_plot:list=["thirdOrder", "semantic"]
                    c:list=["#009716", "#DB57B2"]):

    df = pd.read_excel(open(f'{data_path}/Source_data_manuscript.xlsx', 'rb'), sheet_name='Figure7C', engine='openpyxl') 

    fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
    data_all = []
    data_allSub = {}
    hemi_df_tmp = []
    model_df_tmp = []
    x_tmp = []
    for h_i, h in enumerate(hemis):
        for m_i, m in enumerate(models_plot):
            data_all = []
            for subj in SUBJECTS_ALL:
                tmp = df[(df["sulcus"] == sulcus) & (df["hemis"] == h) & (df["models"] == m) & (df["subjects"] == subj)]["performance"]
                tmp[tmp<=0] = np.nan
                data_all.append(tmp)
            data = np.array(data_all)
            data_allSub["%s_%s"%(m, h)] = data
            
            y = np.nanmean(data,0)
            data_all.extend(y)
            hemi_df_tmp.extend([h]*len(y))
            model_df_tmp.extend([m]*len(y))
            x = np.unique(df[(df["sulcus"] == sulcus) & (df["hemis"] == h)]["coordinates"])
            x_tmp.extend(x)
            
            error = scipy.stats.sem(data,0)
            y_smooth = make_interp_spline(x, y)(xnew)
            error_smooth = make_interp_spline(x, error)(xnew)

            axs[h_i].plot(xnew, y_smooth, c[m_i])
            axs[h_i].fill_between(xnew, y_smooth-error_smooth, y_smooth+error_smooth, color =c[m_i], alpha = 0.2)

            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)

    plt.show()   


def generate_fig7D(df,
                    sulcus:str,
                    smooth_factor:int=1000,
                    sharex=True): 
    
    x = np.unique(df[(df["sulcus"] == sulcus)]["coordinates"])
    xnew = np.linspace(x[0], x[-1], smooth_factor)

    if sulcus == "IFS":
        sharex = False
    fig, axs = plt.subplots(1,2, sharex=sharex, sharey=True, figsize=(8,6))

    for h_i, h in enumerate(np.unique(df["hemis"])): 
        data_all = []
        for subj_i, subj in enumerate(np.unique(df["subject"])):
            tmp = df[(df["sulcus"] == sulcus) & (df["hemis"] == h) & (df["subject"] == subj)]["curvature"]
            y = np.nan_to_num(tmp)
            y_smooth = make_interp_spline(x, y)(xnew)
            axs[h_i].plot(xnew, y_smooth, alpha=0.3)
            data_all.append(tmp)
 
        data = np.array(data_all)
        y_mean = np.nanmean(data,0)
        y_mean_smooth = make_interp_spline(x, y_mean)(xnew)
        axs[h_i].plot(xnew, y_mean_smooth, color="k")

        if ((sulcus == "IFS") & (h == "left"):
            axs[h_i].set_xlim([-30, 40])
        elif ((sulcus == "IFS") & (h == "right"):
            axs[h_i].set_xlim([-55, 5])


def draw_boxplot(data, 
                ax, 
                subjects:list=SUBJECTS_ALL):
    for n in range(3):
        ax.scatter(np.ones(len(subjects))*(n+1), data[:,n], color = "gray")
    
    bp = ax.boxplot(data, 0, '',  positions = np.arange(1,4), patch_artist=True, widths=0.3);
    colors = ["r", "g", "b"]
    for patch, whisker, color in zip(bp['boxes'], bp['whiskers'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        whisker.set_linestyle('-.')
    for element in ['medians', 'caps']:
        plt.setp(bp[element], color="k")

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

