from .modeling_utils import run_himalaya, stepwise_BOLD

from phoneme_segmentation.features.dsutils import save_table_file
from phoneme_segmentation.config import (
        SUBJECTS_ALL,
        MODELS_ALL,
        MODEL_VP,
        MODEL_DIR,
        VP_FEATURE_IDX,
        VP_FEATURE_NAME
)

perf_all = {}
for subject in SUBJECTS_ALL:
    for model in MODELS_ALL:
        if model == "baseline":
            _ = run_himalaya(subject, model)
            stepwise_BOLD(subject, model)
        else:
            perf_all[f"{subject}_{model}"] = run_himalaya(subject, model)

    res_dir = f"{MODEL_DIR}/{subj}_thirdOrder/"
    parameters_load_all[f"{subj}_deltas"] = np.nan_to_num(cci.download_raw_array(f"{res_dir}/deltas"))

    BOLD_all[f"{subj}_train"], BOLD_all[f"{subj}_test"] = load_BOLD(subj)

    for f_i, feature_idx_check in enumerate(VP_FEATURE_IDX): 
        test_score_vp_all[f"{subj}_{feature_names_all[f_i]}"] = run_vp_wrapper(feature_idx_check,VP_FEATURE_NAME[f_i], Xs_train_all, Xs_test_all, parameters_load_all[f"{subj}_deltas"], np.nan_to_num(BOLD_all[f"{subj}_train"]), np.nan_to_num(BOLD_all[f"{subj}_test"]), res_dir)
    save_table_file(f"{MODEL_DIR}/{subject}_perf_all.hdf")

## visualize prediction performance on the flatmap 
[plot_flatmap(k.split("_")[0], p, f"{FIG_DIR}/{k}_performance.png") for k, p in perf_all]
