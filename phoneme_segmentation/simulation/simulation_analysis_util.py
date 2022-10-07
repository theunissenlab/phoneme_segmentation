import  numpy as np
from scipy.interpolate import make_interp_spline


def load_data(model_input):
    '''
    model: single or diphone
    '''
    f_path = "20200924_simulation/numPhone_%s/tempLen*_itr*/rsq_perf"%(model_input)
    file_names_all = cci.glob(f_path)
    
    ## first get all the tempLen
    tempLen_uniqueAll = np.unique(np.array([int(f.split("tempLen")[-1].split("_itr")[0]) for f in file_names_all]))

    print (tempLen_uniqueAll)
    
    rsq_perf_all = dict()

    for t in tempLen_uniqueAll:
        files = cci.glob("20200924_simulation/numPhone_%s/tempLen%s_itr*/rsq_perf"%(model_input, t))
        print (t)
        print (len(files))
        rsq_perf_tmp_all = []
        for f in files:
            rsq_perf = cci.download_raw_array(f)
            rsq_perf_tmp_all.append(rsq_perf)

        rsq_perf_all["%s"%(t)] = np.array(rsq_perf_tmp_all)
        
    return rsq_perf_all

def calc_stats(rsq_perf_all, fit_model):
    
    perf_plot_mean = np.zeros((len(rsq_perf_all), 3))
    perf_plot_std = np.zeros((len(rsq_perf_all), 3))

    if fit_model == "phnid":
        fit_model_idx = 1
    elif fit_model == "phncount":
        fit_model_idx = 0
        
    for k_i, k in enumerate(rsq_perf_all.keys()):
        perf_plot_mean[k_i,:] = rsq_perf_all[k][:,fit_model_idx,:].mean(0)
        perf_plot_std[k_i,:] = np.std(rsq_perf_all[k][:,fit_model_idx,:],0)
        
    return perf_plot_mean, perf_plot_std

def mk_plot(x_input, perf_plot_mean, perf_plot_std, model_input, smooth_factor=1000, save_plt=False):
    
    x = np.array(x_input)*2/60
    
    if model_input == "single":
        c = ["purple", "green"]
        
    elif model_input == "diphone":
        c = ["purple", "cyan"]

    fig, ax = plt.subplots()
    xnew = np.linspace(x[0], x[-1], smooth_factor) 

    for m in range(2):
        y = perf_plot_mean[:, m]
        error = perf_plot_std[:, m]

        y_smooth = make_interp_spline(x, y)(xnew)
        error_smooth = make_interp_spline(x, error)(xnew)

        ax.plot(xnew, y_smooth, c[m])
        ax.fill_between(xnew, y_smooth-error_smooth, y_smooth+error_smooth, color =c[m], alpha = 0.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('Performance (R^2)')
    ax.set_xlabel('Stimulus Length (min)')
    # ax.set_xscale('log')
    ax.legend(["numPhone model", "%s model"%(model_input)], bbox_to_anchor=(1.1, 1.05))

    if save_plt:
        fig.savefig('./singlePhone_simulation_sum.eps', bbox_inches='tight', format='eps', dpi=1000)

def analysis_pipeline(model_input, fit_model):
    '''
    model_input: str: single or diphone or triphone or semantic
    fit  model: str: phnid or phncount
    '''
    ## load in data
    rsq_perf_all = load_data(model_input)
    
    ## compute stats
    perf_plot_mean, perf_plot_std = calc_stats(rsq_perf_all, fit_model)
    
    ## mkplots 
    mk_plot(tempLen_uniqueAll, perf_plot_mean, perf_plot_std, model_input)
    
    return rsq_perf_all
