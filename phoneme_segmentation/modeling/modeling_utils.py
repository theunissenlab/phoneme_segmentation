import numpy as np
import tables
from scipy.stats import zscore
import operator
from functools import reduce
from phoneme_segmentation.config import pycortex_info

from himalaya.backend import set_backend
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
from himalaya.kernel_ridge import predict_and_score_weighted_kernel_ridge
from himalaya.kernel_ridge import primal_weights_weighted_kernel_ridge
from himalaya.scoring import r2_score_split
from himalaya.viz import plot_alphas_diagnostic
print(__doc__)

from phoneme_segmentation.features.dsutils import (
        make_delayed,
        save_table_file
    )

from phoneme_segmentation.config import (
        BOLD_DIR,
        FEATURES_MATRIX_PATH,
        MODEL_DIR,
        MODEL_FEATURE_MATRIX
    )

backend = set_backend("cupy")


def run_himalaya(subject,  
                model, 
                delays=range(1,5),
                n_iter=1000,
                alphas=np.logspace(-10, 10, 21),
                n_targets_batch=1000,
                n_alphas_batch=20,
                return_weights="dual",
                n_targets_batch_refit=200):

    if model == "baseline":
        BOLD_f = tables.open_file(f"{BOLD_DIR}/{subject}_BOLD.hdf")
    else:
        BOLD_f = tables.open_file(f"{BOLD_DIR}/{subject}_BOLD_baseline_stepwised.hdf")

    Y_train = backend.asarray(np.nan_to_num(BOLD_f.root.zRresp.read()))
    Y_test = backend.asarray(np.nan_to_num(BOLD_f.root.zPresp.read()))

    FEATURE_f = tables.open_file(FEATURES_MATRIX_PATH)
    features = MODEL_FEATURE_MATRIX[model]
    Xs_train = [backend.asarray(np.nan_to_num(make_delayed(FEATURE_f.root[f"{f}_Rstim"], delays))) for f in features]
    Xs_test = [backend.asarray(np.nan_to_num(make_delayed(FEATURE_f.root[f"{f}_Pstim"], delays))) for f in features] 

    assert np.all(Y_test.mean(1) < 1e-5)
    assert np.all(Y_test.std(1) - 1 < 1e-5)
    assert np.all(Y_train.mean(0) < 1e-5)
    assert np.all(Y_train.std(0) - 1 < 1e-5)

    print("BOLD data size")
    print (Y_train.shape)
    print (Y_test.shape)

    print("feature data size")
    for i, f in enumerate(Xs_train):
            print (f.shape)
            print (Xs_test[i].shape)

    # Precompute the linear kernels, and cast them to float32.
    Ks_train = backend.stack([X_train @ X_train.T for X_train in Xs_train])
    Ks_train = backend.asarray(Ks_train, dtype=backend.float32)
    Y_train = backend.asarray(Y_train, dtype=backend.float32)
    Ks_test = backend.stack(
        [X_test @ X_train.T for X_train, X_test in zip(Xs_train, Xs_test)])
    Ks_test = backend.asarray(Ks_test, dtype=backend.float32)
    Y_test = backend.asarray(Y_test, dtype=backend.float32)

    results = solve_multiple_kernel_ridge_random_search(
    Ks=Ks_train,
    Y=Y_train,
    n_iter=n_iter,
    alphas=alphas,
    n_targets_batch=n_targets_batch,
    return_weights=return_weights,
    n_alphas_batch=n_alphas_batch,
    n_targets_batch_refit=n_targets_batch_refit,
    jitter_alphas=True,
    )

    deltas = backend.to_numpy(results[0])
    dual_weights = backend.to_numpy(results[1])
    cv_scores = backend.to_numpy(results[2])

    split = True
    scores = predict_and_score_weighted_kernel_ridge(
    Ks_test, dual_weights, deltas, Y_test, split=split,
    n_targets_batch=n_targets_batch, score_func=r2_score_split)
    scores = backend.to_numpy(scores)

    split = False
    scores_all = predict_and_score_weighted_kernel_ridge(
    Ks_test, dual_weights, deltas, Y_test, split=split,
    n_targets_batch=n_targets_batch, score_func=r2_score_split)
    scores_all = backend.to_numpy(scores_all)

    output = f"{MODEL_DIR}/{subject}_{model}_himalaya_modeling_res.hdf"
    
    if model == "baseline":
        primal_wts = primal_weights_weighted_kernel_ridge(dual_weights, deltas, Xs_train)
        pred_train_prep = {features[f_i]: np.dot(Xs_train[f_i], primal_wts[f_i]) for f_i in range(len(features))} 
        pred_test_prep =  {features[f_i]: np.dot(Xs_test[f_i], primal_wts[f_i]) for f_i in range(len(features))}

        pred_train = reduce(operator.add, [np.nan_to_num(pred_train_prep[i]) for i in pred_train_prep.keys()])
        pred_test = reduce(operator.add, [np.nan_to_num(pred_test_prep[i]) for i in pred_test_prep.keys()])

        save_table_file(output, dict(deltas = deltas, dual_weights = dual_weights, cv_scores = cv_scores, scores=scores, scores_all=scores_all, pred_train = pred_train, pred_test = pred_test))

    else: 
        save_table_file(output, dict(deltas = deltas, dual_weights = dual_weights, cv_scores = cv_scores, scores=scores, scores_all=scores_all)) 

    return scores_all

def calc_primal_wts_wasabi(features:list,
                            res_dir:str,
                            root_path="moth_listening_en"
                        ):

    Xs_train = [np.nan_to_num(make_delayed(cci.download_raw_array(os.path.join(root_path, f"Features/{f}_train")), delays)) for f in features]
 
    wts_dual = cci.download_raw_array(os.path.join(res_dir, "dual_weights"))
    deltas = cci.download_raw_array(os.path.join(res_dir, "deltas"))

    pred = predict_weighted_kernel_ridge(Xs_test, wts_dual, deltas)

    return pred

def stepwise_BOLD(subject:str, 
                model:str):
    BOLD_f = tables.open_file(f"{BOLD_DIR}/{subject}_BOLD.hdf")
    Y_train = backend.asarray(np.nan_to_num(BOLD_f.root.zRresp.read()))
    Y_test = backend.asarray(np.nan_to_num(BOLD_f.root.zPresp.read()))

    MODEL_f = tables.open_file(f"{MODEL_DIR}/{subject}_{model}/himalaya_modeling_res.hdf")
    pred_train = backend.asarray(np.nan_to_num(MODEL_f.root.pred_train.read()))
    pred_test = backend.asarray(np.nan_to_num(MODEL_f.root.pred_test.read()))

    Rresp_new = Y_train - pred_train
    Presp_new = Y_test - pred_test

    zRresp_new = zscore(Rresp_new, axis = 0)
    zPresp_new = zscore(Presp_new, axis = 0)

    save_table_file(f"{BOLD_DIR}/{subject}_BOLD_baseline_stepwised.hdf", dict(zRresp=zRresp_new, zPresp=zPresp_new))

def plot_flatmap(subject:str, 
                performance, 
                output_path:str):
    mask = cortex.db.get_mask(**pycortex_info[subject], type="thick")
    vol = cortex.Volume(performance, **pycortex_info[subject], mask=mask,
                            recache = True, vmin=0, vmax=0.1, cmap='inferno', with_curvature=True)
    cortex.quickshow(vol,colorbar_location="right")
    cortex.quickflat.make_png(output_path, vol, colorbar_location="right")
