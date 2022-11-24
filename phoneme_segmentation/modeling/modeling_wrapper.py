from .modeling_utils import run_himalaya, stepwise_BOLD

from phoneme_segmentation.features.dsutils import save_table_file
from phoneme_segmentation.config import (
        SUBJECTS_ALL,
        MODELS_ALL,
        MODEL_VP,
        MODEL_DIR
    
)

perf_all = {}
for subject in SUBJECTS_ALL:
    for model in MODELS_ALL:
        if model == "baseline":
            _ = run_himalaya(subject, model)
            stepwise_BOLD(subject, model)
        else:
            perf_all[f"{subject}_{model}"] = run_himalaya(subject, model)

    ## variance partition
    for model, vp_set in MODEL_VP.items():
        if model == "single":
            perf_all[f"{subject}_{model}"] = perf_all[f"{subject}_{vp_set[0]}"]
        else:
            perf_all[f"{subject}_{model}"] = perf[f"{subject}_{vp_set[1]}"] - perf_all[f"{subject}_{vp_set[0]}"]

    save_table_file(f"{MODEL_DIR}/{subject}_perf_all.hf5")

## visualize prediction performance on the flatmap 
[plot_flatmap(k.split("_")[0], p, f"{FIG_DIR}/{k}_performance.png") for k, p in perf_all]
