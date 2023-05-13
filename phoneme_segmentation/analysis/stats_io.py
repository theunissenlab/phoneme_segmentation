import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from phoneme_segmentation.config import *

def EV_calc():
    valid_all = {subj: np.load(f"{BOLD_VALID_DIR}/{subj}_valid.npz")["wheretheressmoke"] for subj in SUBJECTS_ALL}
    for subj in SUBJECTS_ALL:
        print(subj)
        print(valid_all[subj].shape)


    EV_all = {k: explainable_variance(v) for k, v in valid_all.items()}
    print([v.shape for _, v in EV_all.items()])

    cci.dict2cloud(f"{S3_EV_DIR}", EV)
    return EV_all

def sig_vox_threshold(perf_all:dict,
                    perm_feature:str,
                    models:list):

    '''
    perf_all: output of reorg_perf_raw
    perm_feature: toSemAll(for all other models, particularly thirdOrder vs semantic comparison) or thirdOrder (for VP analysis)
    models: list of models: for toSemAll threshold (MODELS_ALL + ["powspec", "numPhns"]): baseline, powspec, numPhns, firstOrder, secondOrder, thirdOrder, toSemAll, semantic
                            for thirdOrder threshold (MODELS_VP): single, diphone, triphone, singleDiphn, singleTri, DiphoneTri, SingDiTri
    '''

    perf_sig_all = {}
    perf_sig_EVcorrected_all = {}
    EV_all = {}
    perm_res_all = {} 

    for subj in SUBJECTS_ALL:

        EV_all[subj] = cci.download_raw_array(f"{S3_EV_DIR}/{subj}")

        ## load in perm res
        perm_res_all[f"{subj}_{perm_feature}"] = np.load(f"{LOCAL_PERM_RES}/{subj}_{perm_feature}_perm.npz")["perm"]
        perf_perm_tmp = cci.download_raw_array(f"{S3_MODEL_RAW_SUMMARY_ROOT_DIR}/{subj}_{perm_feature}")


        for m in models:
            print(f"{subj}_{m}")
            perf_sig_all[f"{subj}_{m}"], perf_sig_EVcorrected_all[f"{subj}_{m}"] = calc_stats(perf_all[f"{subj}_{m}"], perf_perm_tmp, perm_res_all[f"{subj}_{perm_feature}"], EV_all[subj])
            print(perf_sig_all[f"{subj}_{m}"].shape)
            print(perf_sig_EVcorrected_all[f"{subj}_{m}"].shape)

            cci.upload_raw_array(f"{S3_MODEL_SIG_SUMMARY_ROOT_DIR}/{subj}_{m}", perf_sig_all[f"{subj}_{m}"])
            cci.upload_raw_array(f"{S3_MODEL_SIG_EVcorrected_SUMMARY_ROOT_DIR}/{subj}_{m}", perf_sig_EVcorrected_all[f"{subj}_{m}"])

    return perf_sig_all, perf_sig_EVcorrected_all


def plot_venn_diagram(perf:dict,
                    plot_idx=[0, 1, 3, 2, 4, 5, 6]):
    '''
    perf: dict: key: subj;  value: [n_features, n_vox]
    '''
    ## quickly viz venn diagram
    true_area_all = {}
    true_area_arr = np.zeros((len(SUBJECTS_ALL), len(MODELS_VP)))
    for subj_i, subj in enumerate(SUBJECTS_ALL):
        true_area = np.nan_to_num(perf[subj]).mean(1)
        true_area /= true_area.sum()
        true_area_all[subj] = true_area
        true_area_arr[subj_i, :] = true_area
        print(MODELS_VP)
        print(true_area)
        true_area_plot = [true_area[i] for i in plot_idx]
        print(true_area_plot)

        # Make the diagram
        plt.title(subj)
        venn3(subsets = [round(i, 2) for i in true_area_plot]) 
        plt.show()

def extract_roi_hemi_perf(pycortex_info
                        subj:str, 
                        roi:str, 
                        hemi:str, 
                        perf):
    if hemi == "left":
        hemi_code = 0
    else:
        hemi_code = 1

    roi_mask = cortex.utils.get_roi_masks(**pycortex_info[subj], roi_list=[roi], gm_sampler='cortical-conservative', return_dict=True)[roi]
    wholeBrain_mask = cortex.get_cortical_mask(**pycortex_info[subj], type = "thick")
    hemi_mask = cortex.utils.get_hemi_masks(**pycortex_info[subj], type='nearest')[hemi_code]

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

