import numpy as np

from himalaya.kernel_ridge import WeightedKernelRidge
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import Kernelizer
from sklearn.pipeline import make_pipeline

import cottoncandy as cc
cci = cc.get_interface("glab-xgong-speech")

import sys  
sys.path.append('/home/jlg/lilly/gbox/repository/speech/scripts/utils')
from util import make_delayed
import os

import matplotlib.pyplot as plt

import cortex
from scipy import optimize

B = np.array([[0, 0, 0, 0, 0, -1, 1], # Abc
              [0, 0, 0, 0, -1, 0, 1], # aBc
              [0, 0, 0, -1, 0, 0, 1], # abC
              [0, 0, -1, 0, 1, 1, -1], # ABc
              [0, -1, 0, 1, 0, 1, -1], # AbC
              [-1, 0, 0, 1, 1, 0, -1], # aBC
              [1, 1, 1, -1, -1, -1, 1], # ABC
             ])


## load in features
def load_features(model="thirdOrder_rerun",
                    features=["singlePhn", "diphone", "triphone"],
                    root_path="moth_listening_en",
                    delays=range(1,5)):

    Xs_train = [np.asarray(np.nan_to_num(make_delayed(cci.download_raw_array(os.path.join(root_path, "Features/%s_train"%(f))), delays))) for f in features]
    Xs_test = [np.asarray(np.nan_to_num(make_delayed(cci.download_raw_array(os.path.join(root_path, "Features/%s_test"%(f))), delays))) for f in features]

    return Xs_train, Xs_test

## load in Y_train and X_train
def load_BOLD(subj,
             root_path="moth_listening_en"):
    Y_train = np.asarray(np.nan_to_num(cci.download_raw_array("%s/BOLD_baseline_stepwised/%s_train"%(root_path, subj))))
    Y_test = np.asarray(np.nan_to_num(cci.download_raw_array("%s/BOLD_baseline_stepwised/%s_test"%(root_path, subj))))

    print(Y_train.shape)
    print(Y_test.shape)
    
    return Y_train, Y_test


def run_vp(n_feature_lists,
          deltas,
          Xs_train_arr,
           Xs_test_arr,
          Y_train,
          Y_test):
    ## prep pipeline
    print(sum(n_feature_lists))
    print(Xs_train_arr.shape)

    assert sum(n_feature_lists) == Xs_train_arr.shape[-1]
    start_and_end = np.concatenate([[0], np.cumsum(n_feature_lists)])

    print(start_and_end)

    slices = [
        slice(start, end)
        for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]

    print(slices)

    # Create a different ``Kernelizer`` for each feature space.
    from himalaya.kernel_ridge import Kernelizer
    from himalaya.kernel_ridge import ColumnKernelizer

    kernelizers = [("space %d" % ii, Kernelizer(), slice_)
                   for ii, slice_ in enumerate(slices)]
    column_kernelizer = ColumnKernelizer(kernelizers)

    print(kernelizers)
    print(column_kernelizer)

    from himalaya.kernel_ridge import WeightedKernelRidge
    # model_all_test = WeightedKernelRidge(alpha = best_alphas, deltas=deltas,
    #                                     kernels = "precomputed")

    ## after discussing with Tom
    model_all_test = WeightedKernelRidge(alpha = 1, deltas=deltas,
                                        kernels = "precomputed")

    pipe_all_test = make_pipeline(column_kernelizer, model_all_test)

    pipe_all_test.fit(Xs_train_arr, Y_train)

    test_scores_all = pipe_all_test.score(Xs_test_arr, Y_test)

    return test_scores_all


def run_vp_wrapper(feature_idx:list,
                   feature_name,
                  Xs_train:list,
                  Xs_test:list,
                  deltas,
                  Y_train,
                  Y_test,
                  res_dir:str,
                  mk_plot:bool=False,
                  pycortex_info:dict=None):

    n_feature_lists = [Xs_train[i].shape[-1] for i in feature_idx]
    deltas_tmp = deltas[feature_idx]
    Xs_train_arr = np.hstack([Xs_train[i] for i in feature_idx])
    Xs_test_arr = np.hstack([Xs_test[i] for i in feature_idx])

    print(Xs_train_arr.shape)
    print(Xs_test_arr.shape)
    print(n_feature_lists)
    print(deltas_tmp.shape)

    test_score = run_vp(n_feature_lists,
                              deltas_tmp,
                              Xs_train_arr,
                               Xs_test_arr,
                              Y_train,
                              Y_test)


    if mk_plot == True:
        vol = cortex.Volume(test_score, **pycortex_info[subj],
                            vmin=0, vmax=0.1, cmap="hot", with_curvature=True)
        cortex.quickshow(vol)
        plt.show()

    cci.upload_raw_array(f"{res_dir}/vp/test_score_{feature_name}", test_score)

    return test_score

from scipy import optimize

def correct_rsqs(b, neg_only=False, minimize="l2", verbose=True, **etc):
    maxs = B.dot(np.nan_to_num(b))
    print(maxs.shape)
    if minimize == "l2":
        minfun = lambda x: (x ** 2).sum()
    elif minimize == "l1":
        minfun = lambda x: np.abs(x).sum()
    else:
        raise ValueError(minimize)

    biases = np.zeros((maxs.shape[1], 7)) + np.nan
    for vi in range(b.shape[1]):
        if not (vi % 1000) and verbose:
            print("%d / %d" % (vi, b.shape[1]))
        
        if neg_only:
            bnds = [(None, 0)] * 7
        else:
            bnds = [(None, None)] * 7
        res = optimize.fmin_slsqp(minfun, np.zeros(7),
                                        f_ieqcons=lambda x: maxs[:,vi] - B.dot(x),
                                        bounds=bnds, iprint=0)
        biases[vi] = res
    
    # compute fixed (legal) variance explained values for each model
    fixed_b = np.array(b) - np.array(biases).T

    orig_parts = B.dot(b)
    fixed_parts = B.dot(fixed_b)
    
    return biases, orig_parts, fixed_parts
