import numpy as np
import random
import tables
from scipy.stats import zscore
import time

import cortex
from tikreg import models, spatial_priors as SpatialPriors, temporal_priors as TemporalPriors
from tikreg.models import *
import hrf_estimation
from tikreg.utils import columnwise_rsquared as rsq, columnwise_correlation as col_corr

from .simulation_util import mod1_generation, mod2_generation
from phoneme_segmentation.features.dsutils import save_table_file
from phoneme_segmentation.config import (
        SIMULATION_DIR,
        FEATURES_MATRIX_PATH
        )
from phoneme_segmentation.modeling.modeling_utils import run_himalaya 

start_time = time.time()

models = ["single", "diphone"]
temLen_all = list(range(30, 3500, 60))

for model in models:
    for tempLen_selection in temLen_all:
        run_simulation(model, tempLen_selection)

def run_simulation(model, tempLen_selection, iteration=10):
    
    print ("load in features")
    FEATURE_f = tables.open_file(SIMULATION_DIR)
    low_stimR = FEATURE_f.root.numPhone_Rstim.read()
    low_stimP = FEATURE_f.root.numPhone_Pstim.read()

    high_stimR = FEATURE_f.root[f"{model}_Rstim".read()
    high_stimP = FEATURE_f.root[f"{model}_Pstim".read()

    tempR_Len, feature_len = high_stimR.shape

    print ("generate fake BOLD")
    if tempLen_selection <= tempR_Len:
        if iteration < 5:
            tempR_idx_start = random.sample(range(0, tempR_Len - tempLen_selection), 1)[0]
            tempR_idx = list(range(tempR_idx_start, tempR_idx_start + tempLen_selection))
        else:
            tempR_idx = np.sort(np.array(random.sample(range(tempR_Len), tempLen_selection)))
    else:
        tempR_idx = np.sort(np.array(np.random.choice(range(tempR_Len), tempLen_selection)))

    print (tempR_idx)

    low_stimR_selected = low_stimR[tempR_idx, :]
    high_stimR_selected = high_stimR[tempR_idx, :]

    if model == "single":
        feature_selection = [5, 20, 36]
    elif model == "diphone":
        feature_selection = [200, 600, 850]

    sig1R, sig1P, bold1ModR, bold1ModP = mod1_generation(low_stimR_selected, low_stimP, 0, low_stimR)
    sig2R, sig2P, bold2ModR, bold2ModP = mod1_generation(high_stimR_selected, high_stimP, feature_selection, high_stimR)

    bold3ModR, bold3ModP = mod2_generation(sig1R, sig1P, sig2R, sig2P)

    Rresp = np.vstack((bold1ModR, bold2ModR, bold3ModR)).T
    Presp = np.vstack((bold1ModP, bold2ModP, bold3ModP)).T

    print ("zsore response")
    zRresp = zscore(Rresp)
    zPresp = zscore(Presp)

    print (zRresp.shape)
    print (zPresp.shape)

    print ("prep for ridge regression")

    trainFeatures = [low_stimR_selected, high_stimR_selected]
    testFeatures = [low_stimP, high_stimP]

    featurePriors = [SpatialPriors.SphericalPrior(low_stimR_selected, hyparams = np.logspace(-1, 6, 10)),
                    SpatialPriors.SphericalPrior(high_stimR_selected, hyparams = np.logspace(-1, 6, 10))]

    delays = range(1,5)
    temporalPrior = TemporalPriors.HRFPrior(delays)

    print ("running banded ridge regression using RSQUARED")


    fit = models.estimate_stem_wmvnp(trainFeatures, zRresp, testFeatures, zPresp, ridges = np.logspace(-1, 6, 10), temporal_prior = temporalPrior, feature_priors = featurePriors, predictions = True, performance = True, weights=True, folds=(2,5), metric = 'rsquared')

    print ("saving regression results using RSQUARED")
    output_path = f"{SIMULATION_DIR}/numPhone_{model}_tempLen{tempLen_selection}_itr{iteration}.hf5"
    save_table_file(output_path,dict(tempR_idx = np.array(tempR_idx),
                                    zRresp = zRresp,
                                    zPresp = zPresp,
                                    perf = fit["performance"],
                                    wts = fit["weights"],
                                    optima  = fit["optima"],
                                    pred = fit["predictions"]))

    print ("compute dual weights and performance for each feature space")
    optima  = fit["optima"]
    wts_core = fit["weights"]

    wts_low = dual2primal_weights_banded(wts_core, trainFeatures[0], optima[:,1]**-2, temporalPrior, False, verbose=False)
    wts_high = dual2primal_weights_banded(wts_core, trainFeatures[1], optima[:,2]**-2, temporalPrior, False, verbose=False)

    wts_low_vstack = np.vstack([w for w in wts_low])
    wts_high_vstack = np.vstack([w for w in wts_high])

    model_wts = [wts_low_vstack, wts_high_vstack]
    predictions = [np.dot(make_delayed(stim, delays), wt) for stim,wt in zip(testFeatures, model_wts)]

    rsq_perf = np.array([rsq(pred, zPresp) for pred in predictions])

    save_table_file(output_path,dict(wts_low = wts_low_vstack,
                                    wts_high = wts_high_vstack,
                                    rsq_perf = rsq_perf
                                    ))

