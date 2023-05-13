import numpy as np

from phoneme_segmentation.features.dsutils import (
        make_delayed,
    )

from phoneme_segmentation.config import ( 
        FEATURES_MATRIX_PATH,
        MODEL_DIR,
        MODEL_FEATURE_MATRIX
    )

from himalaya.backend import set_backend

def generate_perm_idx(features,
                      train_len,
                      test_len,
                     itr_num_all=1000):
    perm_idx_all = {}
    itr_per_model = len(features)
    for i_all in range(itr_num_all):
        for i in range(itr_per_model):
            train_sample_rand = np.random.permutation(range(train_len))
            test_sample_rand = np.random.permutation(range(test_len))

            perm_idx_all[f"itr{i_all}_{features[i]}_train"] = train_sample_rand
            perm_idx_all[f"itr{i_all}_{features[i]}_test"] = test_sample_rand

    return perm_idx_all

def permutation_wrapper(Xs_train_prep,
                        Xs_test_prep,
                        train_idx,
                        test_idx
                        Y_train,
                        Y_test,
                        deltas,
                        features,
                        ):

    n_feature_lists = [Xs_train[i].shape[-1] for i in range(len(features))]
    Xs_train = [backend.asarray(np.nan_to_num(make_delayed(Xs_train_prep[f_i][train_idx[f], :], delays))) for f_i, f in enumerate(features)]
    Xs_test = [backend.asarray(np.nan_to_num(make_delayed(Xs_train_prep[f_i][test_idx[f], :], delays))) for f_i, f in enumerate(features)]

    Xs_train_arr = backend.asarray(np.hstack([i for i in Xs_train]), dtype="float32")
    Xs_test_arr = backend.asarray(np.hstack([i for i in Xs_test]), dtype="float32")

    vp_running_start_time = time.time()
        test_score = run_vp(n_feature_lists,
                          deltas,
                          Xs_train_arr,
                          Xs_test_arr,
                          Y_train,
                          Y_test)

    test_score = backend.to_numpy(test_score)

    return test_score
