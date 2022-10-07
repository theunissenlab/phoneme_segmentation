import numpy as np
import tables

from .stimulus_utils import (
    load_grids_for_stories,
    load_generic_trfiles,
)

from .dsutils import (
    make_word_ds,
    make_phoneme_ds,
    make_diphone_ds,
    make_triphone_ds,
    classify_diphone, 
    extract_fs,
    histogram_phon,
    make_semantic_model,
    matrix_transform,
    save_table_file
)

from .SemanticModel import SemanticModel
from .sound_utils import wav2powspec

from phoneme_segmentation.config import (
    TEXTGRID_DIR,
    TRFILE_DIR,
    ENG1000_PATH,
    FEATURE_BASIS_PATH,
    FEATURES_DIR,
    FEATURES_MATRIX_PATH,
    TRAIN_STORIES,
    TEST_STORIES
)

ALLSTORIES = sorted(TRAIN_STORIES + TEST_STORIES)

interptype = "lanczos"
window = 3

zscore_all = lambda v: (v-v.mean())/v.std()

def load_features(allstories=ALLSTORIES, Rstories=sorted(TRAIN_STORIES), Pstories=TEST_STORIES):
    '''
    step1: load in grid recoding the time point of each phon and word
			and TRfile: recoding the time point each fMRI image was collected
    '''
    grids = load_grids_for_stories(allstories, TEXTGRID_DIR)
    trfiles = load_generic_trfiles(allstories, TRFILE_DIR)

    '''
    step2: Make word, single phoneme, diphone, and triphone data sequences
    '''
    word_seqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
    singlephone_seqs = make_phoneme_ds(grids, trfiles) # dictionary of {storyname : phoneme DataSequence}
    diphone_seqs = make_diphone_ds(word_seqs,singlephone_seqs,trfiles)
    triphone_seqs = make_triphone_ds(word_seqs,singlephone_seqs,trfiles)

    '''
    step3: project single, diphon, triphon and word stimulus onto feature space
        single phon (39)
        diphon features (858): all biphon combinations exist in the story and IPHOD 
        triphon(4841)
        semantics (985)
        sequence matrix shape: [data sequence, feature]
    '''
    phone_basis_file = tables.open_file(FEATURE_BASIS_PATH)
    singlephone_basis = np.array([i.decode() for i in phone_basis_file.root.single_phone.read()])
    diphone_basis = np.array([i.decode() for i in phone_basis_file.root.diphone.read()])
    triphone_basis = np.array([i.decode() for i in phone_basis_file.root.triphone.read()])

    eng1000 = SemanticModel.load(ENG1000_PATH)
    
    semanticseqs = dict() # dictionary to hold projected stimuli {story name : projected DataSequence}
    singlephone_histseqs = dict() # dictionary to hold single phoneme histograms
    diphone_histseqs_all = dict() # dictionary to hold biphoneme histograms
    triphone_histseqs_all = dict() # dictionary to hold triphoneme histograms

    for story in allstories:
        semanticseqs[story] = make_semantic_model(word_seqs[story], eng1000)
        singlephone_histseqs[story] = histogram_phon(singlephone_seqs[story], singlephone_basis)
        diphone_histseqs_all[story] = histogram_phon(diphone_seqs[story],diphone_basis)
        triphone_histseqs_all[story] = histogram_phon(triphone_seqs[story],triphone_basis)

    '''
    step4: classify diphon features into three categories (extract index for each category):
        singlePhonWord, diphonWord, biphon as the beginning of each word
        singlePhonWord and biphonWord could be considered as short words
        assign different index to different biphon categories
    '''

    table_singlePhone_word_all = dict()
    table_diphone_word_all = dict()
    table_diphone_bg_all = dict()

    for s in allstories:
        word_dic = word_seqs[s]
        diphone_dic = diphone_seqs[s]

        table_singlePhone_word_all[s] = classify_diphone(word_dic, diphone_dic, 1)
        table_diphone_word_all[s] = classify_diphone(word_dic, diphone_dic, 2)
        table_diphone_bg_all[s] = classify_diphone(word_dic, diphone_dic, 3)

    '''
    step5: construct count matrix of biphon histograms (True/False) that
    a. excluding short words
            i. single phone word
            ii. diphone word
    b. excluding both short words and the diphon as the beginning (BG) of long words
    sequence matrix shape: [data sequence, feature]
    If a biphon stimulus is a short word itself, its True entry for the biphon feature becomes False
    '''

    diphone_histseqs_noSinglePhonWord_all = dict()
    diphone_histseqs_noSingleDiPhonWord_all = dict()
    diphone_histseqs_noSingleDiPhonWord_BG_all = dict()

    for sto in allstories:

        diphone_orig = diphone_histseqs_all[sto]

        diphone_histseqs_noSinglePhonWord_all[sto] = extract_fs(diphone_orig, diphone_basis, table_singlePhone_word_all[sto])
        diphone_histseqs_noSingleDiPhonWord_all[sto] = extract_fs(diphone_histseqs_noSinglePhonWord_all[sto], diphone_basis, table_diphone_word_all[sto])
        diphone_histseqs_noSingleDiPhonWord_BG_all[sto] = extract_fs(diphone_histseqs_noSingleDiPhonWord_all[sto], diphone_basis, table_diphone_bg_all[sto])

    '''
    step6: Downsample stimuli so that stimuli matrix and fMRI data have same dimension in time
    '''

    semanticseqs_downsampled = dict() # dictionary to hold downsampled stimuli
    singlephone_downsampled = dict()
    diphone_downsampled_all = dict()
    triphone_downsampled_all = dict() 

    diphone_histseqs_noSinglePhonWord_ds_all = dict()
    diphone_histseqs_noSingleDiPhonWord_ds_all = dict()
    diphone_histseqs_noSingleDiPhonWord_BG_ds_all = dict()

    numWords = dict()  ## number of words in one TR
    numPhone = dict()

    for story in allstories:
        semanticseqs_downsampled[story] = semanticseqs[story].chunksums(interptype, window=window)
        singlephone_downsampled[story] = singlephone_histseqs[story].chunksums(interptype, window=window)
        diphone_downsampled_all[story] = diphone_histseqs_all[story].chunksums(interptype, window=window)
        triphone_downsampled_all[story] = triphone_histseqs_all[story].chunksums(interptype, window=window) 

        diphone_histseqs_noSinglePhonWord_ds_all[story] = diphone_histseqs_noSinglePhonWord_all[story].chunksums(interptype, window=window)
        diphone_histseqs_noSingleDiPhonWord_ds_all[story] = diphone_histseqs_noSingleDiPhonWord_all[story].chunksums(interptype, window=window)
        diphone_histseqs_noSingleDiPhonWord_BG_ds_all[story] = diphone_histseqs_noSingleDiPhonWord_BG_all[story].chunksums(interptype, window=window)

        numWords[story] = np.array([x.shape[0] for x in word_seqs[story].chunks()])
        numPhone[story] = np.array([x.shape[0] for x in singlephone_seqs[story].chunks()])


    '''
    step7: construct feature matix for all biphon stim and the biphon categories
            For biphon categories' transformation matrix:
            1. concatenate stim from all the stories togehter
            2. Use the mean and std of all biphon stim to zscore the biphon category matrix
    '''
    ## concatenate feature space across stories and zscore the whole feature matrix

    trim = 5

    semantic_Rstim = zscore_all(np.vstack([semanticseqs_downsampled[story][5+trim:-trim] for story in Rstories]))
    semantic_Pstim= zscore_all(np.vstack([semanticseqs_downsampled[story][5+trim:-trim] for story in Pstories]))

    single_Rstim = zscore_all(np.vstack([singlephone_downsampled[story][5+trim:-trim] for story in Rstories]))
    single_Pstim = zscore_all(np.vstack([singlephone_downsampled[story][5+trim:-trim] for story in Pstories]))

    diphone_Rstim = zscore_all(np.vstack([(diphone_downsampled_all[story][5+trim:-trim]) for story in Rstories]))
    diphone_Pstim = zscore_all(np.vstack([(diphone_downsampled_all[story][5+trim:-trim]) for story in Pstories]))

    triphone_Rstim = zscore_all(np.vstack([triphone_downsampled_all[story][5+trim:-trim] for story in Rstories]))
    triphone_Pstim = zscore_all(np.vstack([triphone_downsampled_all[story][5+trim:-trim] for story in Pstories]))

    numWords_Rstim = zscore_all(np.vstack([numWords[story][5+trim:-trim][:,np.newaxis] for story in Rstories]))
    numWords_Pstim = zscore_all(np.vstack([numWords[story][5+trim:-trim][:,np.newaxis] for story in Pstories]))

    numPhone_Rstim = zscore_all(np.vstack([numPhone[story][5+trim:-trim][:,np.newaxis] for story in Rstories]))
    numPhone_Pstim = zscore_all(np.vstack([numPhone[story][5+trim:-trim][:,np.newaxis] for story in Pstories]))

    ## for different categories of biphon features matrix: concatenate feature space across stories

    Rdiphone_histseqs_noSinglePhonWord_concate = np.array(np.vstack([(diphone_histseqs_noSinglePhonWord_ds_all[story][5+trim:-trim]) for story in Rstories]))
    Rdiphone_histseqs_noSingleDiPhonWord_concate = np.array(np.vstack([(diphone_histseqs_noSingleDiPhonWord_ds_all[story][5+trim:-trim]) for story in Rstories]))
    Rdiphone_histseqs_noSingleDiPhonWord_BG_concate = np.array(np.vstack([(diphone_histseqs_noSingleDiPhonWord_BG_ds_all[story][5+trim:-trim]) for story in Rstories]))

    Pdiphone_histseqs_noSinglePhonWord_concate = np.array(np.vstack([(diphone_histseqs_noSinglePhonWord_ds_all[story][5+trim:-trim]) for story in Pstories]))
    Pdiphone_histseqs_noSingleDiPhonWord_concate = np.array(np.vstack([(diphone_histseqs_noSingleDiPhonWord_ds_all[story][5+trim:-trim]) for story in Pstories]))
    Pdiphone_histseqs_noSingleDiPhonWord_BG_concate = np.array(np.vstack([(diphone_histseqs_noSingleDiPhonWord_BG_ds_all[story][5+trim:-trim]) for story in Pstories]))

    ## obtain mean and std from biphon fs orig

    Rstim_orig_mean = np.mean(np.vstack([(diphone_downsampled_all[story][5+trim:-trim]) for story in Rstories]))
    Pstim_orig_mean = np.mean(np.vstack([(diphone_downsampled_all[story][5+trim:-trim]) for story in Pstories]))

    Rstim_orig_std = np.std(np.vstack([(diphone_downsampled_all[story][5+trim:-trim]) for story in Rstories]))
    Pstim_orig_std = np.std(np.vstack([(diphone_downsampled_all[story][5+trim:-trim]) for story in Pstories]))

    ## transform the count matrix

    Rstim_noSinglePhonWord = matrix_transform(Rdiphone_histseqs_noSinglePhonWord_concate, Rstim_orig_mean, Rstim_orig_std)
    Rstim_noSingleDiPhonWord = matrix_transform(Rdiphone_histseqs_noSingleDiPhonWord_concate, Rstim_orig_mean, Rstim_orig_std)
    Rstim_noSingleDiPhonWord_BG = matrix_transform(Rdiphone_histseqs_noSingleDiPhonWord_BG_concate, Rstim_orig_mean, Rstim_orig_std)

    Pstim_noSinglePhonWord = matrix_transform(Pdiphone_histseqs_noSinglePhonWord_concate, Pstim_orig_mean, Pstim_orig_std)
    Pstim_noSingleDiPhonWord = matrix_transform(Pdiphone_histseqs_noSingleDiPhonWord_concate, Pstim_orig_mean, Pstim_orig_std)
    Pstim_noSingleDiPhonWord_BG = matrix_transform(Pdiphone_histseqs_noSingleDiPhonWord_BG_concate, Pstim_orig_mean, Pstim_orig_std)

    ## powspec features
    powspec_Rstim, powspec_Pstim = wav2powspec()

    ## save all the features 

    save_table_file(FEATURES_MATRIX_PATH, dict(
                    powspec_Rstim = powspec_Rstim,
                    powspec_Pstim = powspec_Pstim,
                    semantic_Rstim = semantic_Rstim, 
                    semantic_Pstim = semantic_Pstim,
                    single_Rstim = single_Rstim,
                    single_Pstim = single_Pstim,
                    diphone_Rstim = diphone_Rstim,
                    diphone_Pstim = diphone_Pstim,
                    triphone_Rstim = triphone_Rstim,
                    triphone_Pstim = triphone_Pstim,
                    numWords_Rstim = numWords_Rstim,
                    numWords_Pstim = numWords_Pstim,
                    numPhone_Rstim = numPhone_Rstim,
                    numPhone_Pstim = numPhone_Pstim,
                    Rstim_noSinglePhonWord = Rstim_noSinglePhonWord,
                    Rstim_noSingleDiPhonWord = Rstim_noSingleDiPhonWord,
                    Rstim_diphn_res = Rstim_noSingleDiPhonWord_BG,
                    Pstim_noSinglePhonWord = Pstim_noSinglePhonWord,
                    Pstim_noSingleDiPhonWord = Pstim_noSingleDiPhonWord,
                    Pstim_diphn_res = Pstim_noSingleDiPhonWord_BG))
