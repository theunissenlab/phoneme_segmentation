import numpy as np
from .fdr import *

import cortex
from scipy.stats import zscore

from tikreg.utils import columnwise_rsquared as rsq
import copy

from phoneme_segmentation.features.dsutils import save_table_file

def zscore_row(x):
    x_z = (x-x.mean(0))/x.std(0)
    return x_z

def ceiling_prep(subj, mask,  usesg, unwarp, Pstories, trim):
    response_images = load_images(subj, usesg, unwarp, Pstories)     
    ## Limit to two reps of WTS
    response_images[Pstories] = response_images[Pstories][:2]

    resps = dict()
    for story, docs in response_images.items():
        logger.info("Loading response data for story %s.."%story)
        datas = [d.get_data()[:,mask] for d in docs]
    
    print  ("zscore the validation response for %s"%(subj))
    zPresp = np.array([zscore(d[5+trim:-(5+trim)]) for d in datas])

    return zPresp[0], zPresp[1]

def calculate_ceiling_perf(data_one, data_two):
    n_vox = data_one.shape[-1]
    nt = data_one.shape[0]
    r = np.zeros(n_vox)
    for n in range(n_vox):
        r[n] = np.dot(data_one[:,n], data_two[:,n])/nt
        Rsquare = np.square(r)
    return Rsquare

def permutation_test(iterations, data, pred, blocksize, perf, threshold, output_dir):
    n_samples = int(data.shape[0]/(blocksize))
    pred_stats = pred[:blocksize*n_samples,:]

    print ("generate random index")
    rsq_stats_all = []
    for n in range(iterations):
        if n%1000 == 0:
            print (n)
        nsample_rand = np.random.permutation(blocksize*(np.arange((n_samples), dtype = int)))
        ind_rand = np.array([range(i,i+blocksize) for i in nsample_rand]).flatten()

        valid_stats = data[ind_rand, :]
        rsq_stats = rsq(pred_stats, valid_stats)
        rsq_stats_all.append(rsq_stats)

        del valid_stats 
    del pred_stats 

    rsq_stats_all_array = np.array(rsq_stats_all)

    del rsq_stats_all
    
    print ("calculate pvalues")
    pvalues = []
    for perf_i, this_perf in enumerate(np.nan_to_num(perf)):
        count = np.sum(np.nan_to_num(rsq_stats_all_array[:,perf_i]) > this_perf)
        pvalues.append(count/iterations) 
 
    pvalue_arr = np.array(pvalues)
    
    print ("FDR correction for pvalues and obtain significant voxel performance")
    pID, pN = fdr_correct(pvalue_arr, threshold) 
    #mask = pvalue_arr<=pN
    mask = pvalue_arr<=pID

    perf_fdrCorrected = copy.deepcopy(perf)
    for m_i, m in enumerate(mask):
        if m == False:
            perf_fdrCorrected[m_i] = np.nan 

    save_table_file(output_dir, dict(rsq_permutated = rsq_stats_all_array, pvalues = pvalue_arr, pc_fdr = perf_fdrCorrected)) 
    return perf_fdrCorrected

def extract_roi_hemi_perf(subj, roi, hemi, perf):
    surf_subject = dict(AH="AHfs",
                   JG="JGfs",
                   ML="MLfs",
                   WH="WHfs",
                   DS="DSfs",
                   BG="BGfs",
             NNS0="NNS0fs",
             SS="SSfs",
                    AN="ANfs",
                    TZ="TZfs",
                    SP="SPfs")[subj]

    xfms = dict(AHfs="AHfs_auto1",
               MLfs="20121210ML_auto1",
               JGfs="20110321JG_auto2",
               WHfs="20110520WH-audioloc_auto",
               DSfs="20121216DS_auto",
               BGfs="20131210BG_auto3",
               NNS0fs="20140315NNS0_auto2",
               SSfs="20150819SS_auto_reading",
                ANfs="20150722AN_auto_reading",
                TZfs="20160919TZ_auto_reading",
                SPfs="20170424SP_auto_reading")[surf_subject]
    if hemi == "left":
        hemi_code = 0
    else:
        hemi_code = 1

    roi_mask = cortex.utils.get_roi_masks(surf_subject, xfms, roi_list=[roi], gm_sampler='cortical-conservative', return_dict=True)[roi]
    wholeBrain_mask = cortex.get_cortical_mask(surf_subject, xfms, type = "thick")
    hemi_mask = cortex.utils.get_hemi_masks(surf_subject, xfms, type='nearest')[hemi_code]

    roi_hem_mask = np.zeros(np.sum(wholeBrain_mask)).astype(bool)
    h = 0
    for n_i, n in enumerate(wholeBrain_mask.flatten()):
        if n == True:
            if roi_mask.flatten()[n_i]!=0 and hemi_mask.flatten()[n_i] == True:
                roi_hem_mask[h] = True
            else:
                roi_hem_mask[h] = False
            h = h + 1

    perf_roi_hemi = perf[roi_hem_mask]

    roi_hem_mask_plot = np.full(roi_mask.shape, False)
    for x in range(roi_mask.shape[0]):
        for y in range(roi_mask.shape[1]):
            for z in range(roi_mask.shape[2]):
                if wholeBrain_mask[x,y,z]== True and hemi_mask[x,y,z]== True and roi_mask[x,y,z]!=0:
                    roi_hem_mask_plot[x,y,z] = True

    return perf_roi_hemi, roi_hem_mask_plot, roi_hem_mask

def extract_LTCunique(subj, perf, h): 

    _,STS_maskPlot, STS_maskPerf = extract_roi_hemi_perf(subj, "STS", h, perf)
    _,STG_maskPlot, STG_maskPerf = extract_roi_hemi_perf(subj, "STG", h, perf)
    _,AC_maskPlot, AC_maskPerf = extract_roi_hemi_perf(subj, "AC", h, perf)
    _,LTC_maskPlot, LTC_maskPerf = extract_roi_hemi_perf(subj, "LTC", h, perf)

    print ("obtain new mask")
    mask_union_forPlot_tmp = np.array([a or b or c for a, b, c in zip(STS_maskPlot.flatten(), STG_maskPlot.flatten(), AC_maskPlot.flatten())]).reshape((STS_maskPlot.shape))
    mask_union_forPerf_tmp = np.array([a or b or c for a, b, c in zip(STS_maskPerf, STG_maskPerf, AC_maskPerf)])

    mask_forPlot = []
    for a_i, a in enumerate(mask_union_forPlot_tmp.flatten()):
        if a == False and LTC_maskPlot.flatten()[a_i] == True:
            mask_forPlot.append(True)
        else:
            mask_forPlot.append(False)
    mask_exclusion_new = np.array(mask_forPlot)

    mask_forPerf = []
    for a_i, a in enumerate(mask_union_forPerf_tmp):
        if a == False and LTC_maskPerf[a_i] == True:
            mask_forPerf.append(True)
        else:
            mask_forPerf.append(False)
    mask_forPerf = np.array(mask_forPerf)

    perf_exclusion_new = perf[mask_forPerf]
    
    return perf_exclusion_new,  mask_exclusion_new

def extract_FCunique(subj, perf, h):
    _,Broca_maskPlot, Broca_maskPerf = extract_roi_hemi_perf(subj, "Broca", h, perf)
    _,sPMv_maskPlot, sPMv_maskPerf = extract_roi_hemi_perf(subj, "sPMv", h, perf)
    _,FC_maskPlot, FC_maskPerf = extract_roi_hemi_perf(subj, "FC", h, perf)

    print ("obtain new mask")
    mask_union_forPlot_tmp = np.array([a or b for a, b in zip(Broca_maskPlot.flatten(), sPMv_maskPlot.flatten())])
    mask_union_forPerf_tmp = np.array([a or b for a, b in zip(Broca_maskPerf, sPMv_maskPerf)])

    mask_forPlot = []
    for a_i, a in enumerate(mask_union_forPlot_tmp.flatten()):
        if a == False and FC_maskPlot.flatten()[a_i] == True:
            mask_forPlot.append(True)
        else:
            mask_forPlot.append(False)
    mask_exclusion_new = np.array(mask_forPlot)

    mask_forPerf = []
    for a_i, a in enumerate(mask_union_forPerf_tmp):
        if a == False and FC_maskPerf[a_i] == True:
            mask_forPerf.append(True)
        else:
            mask_forPerf.append(False)
    mask_forPerf = np.array(mask_forPerf)

    perf_exclusion_new = perf[mask_forPerf]

    return perf_exclusion_new,  mask_exclusion_new

def extract_ACunique(subj, perf, h):

    _,STS_maskPlot, STS_maskPerf = extract_roi_hemi_perf(subj, "STS", h, perf)
    _,STG_maskPlot, STG_maskPerf = extract_roi_hemi_perf(subj, "STG", h, perf)
    _,AC_maskPlot, AC_maskPerf = extract_roi_hemi_perf(subj, "AC", h, perf)

    print ("obtain new mask")
    mask_union_forPlot_tmp = np.array([a or b for a, b in zip(STS_maskPlot.flatten(), STG_maskPlot.flatten())]).reshape((STS_maskPlot.shape))
    mask_union_forPerf_tmp = np.array([a or b for a, b in zip(STS_maskPerf, STG_maskPerf)])

    mask_forPlot = []
    for a_i, a in enumerate(mask_union_forPlot_tmp.flatten()):
        if a == False and AC_maskPlot.flatten()[a_i] == True:
            mask_forPlot.append(True)
        else:
            mask_forPlot.append(False)
    mask_exclusion_new = np.array(mask_forPlot)

    mask_forPerf = []
    for a_i, a in enumerate(mask_union_forPerf_tmp):
        if a == False and AC_maskPerf[a_i] == True:
            mask_forPerf.append(True)
        else:
            mask_forPerf.append(False)
    mask_forPerf = np.array(mask_forPerf)

    perf_exclusion_new = perf[mask_forPerf]

    return perf_exclusion_new,  mask_exclusion_new

