import os

LOCAL_DIR = os.path.realpath(__file__)
ROOT_DIR = f"{LOCAL_DIR.split('/phoneme_segmentation')[0]}/phoneme_segmentation"
print(ROOT_DIR)

WAV_DIR = f"{ROOT_DIR}/stimuli/"
DATA_DIR = f"{ROOT_DIR}/data/derivative/"
TEXTGRID_DIR = f"{DATA_DIR}/TextGrids"
TRFILE_DIR = f"{DATA_DIR}/TRfiles"
TRAIN_STORIES = ['alternateithicatom',
                'avatar',
                'howtodraw',
                'legacy',
                'life',
                'myfirstdaywiththeyankees',
                'naked',
                'odetostepfather',
                'souls',
                'undertheinfluence']
TEST_STORIES = ["wheretheressmoke"]


TRAIN_STORIES_5SESSIONS = sorted(['myfirstdaywiththeyankees',
                 'stagefright',
                 'thatthingonmyarm',
                 'eyespy',
                 'inamoment',
                 'legacy',
                 'hangtime',
                 'fromboyhoodtofatherhood',
                 'tildeath',
                 'alternateithicatom',
                 'haveyoumethimyet',
                 'adventuresinsayingyes',
                 'avatar',
                 'life',
                 'swimmingwithastronauts',
                 'buck',
                 'souls',
                 'naked',
                 'howtodraw',
                 'itsabox',
                 'undertheinfluence',
                 'adollshouse',
                 'exorcism',
                 'sloth',
                 'odetostepfather',
                 'theclosetthatateeverything'])

FEATURES_DIR = f"{DATA_DIR}/features/"
ENG1000_PATH = f"{FEATURES_DIR}/english1000.hf5"
FEATURE_BASIS_PATH = f"{FEATURES_DIR}/features_basis.hf5"
FEATURES_MATRIX_PATH = f"{FEATURES_DIR}/features_matrix.hf5"

BOLD_DIR = f"{DATA_DIR}/preprocessed_data/"
SUBJECTS_ALL = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"]
pycortex_info = {
    "S01": dict(subject="S01fs", xfmname="S01_auto"),
    "S02": dict(subject="S02fs", xfmname="S02_auto"),
    "S03": dict(subject="S03fs", xfmname="S03_auto"),
    "S04": dict(subject="S04fs", xfmname="S04_auto"),
    "S05": dict(subject="S05fs", xfmname="S05_auto"),
    "S06": dict(subject="S06fs", xfmname="S06_auto"),
    "S07": dict(subject="S07fs", xfmname="S07_auto"),
    "S08": dict(subject="S08fs", xfmname="S08_auto"),
    "S09": dict(subject="S09fs", xfmname="S09_auto"),
    "S10": dict(subject="S10fs", xfmname="S10_auto"),
    "S11": dict(subject="S11fs", xfmname="S11_auto") 
}

MODEL_DIR = f"{ROOT_DIR}/model"
MODELS_ALL = ["baseline", "firstOrder", "secondOrder", "thirdOrder", "toSemAll"]
MODEL_FEATURE_MATRIX = {"baseline": ["powspec", "numPhone"],
                        "firstOrder": ["single"],
                        "secondOrder": ["single", "diphone"],
                        "thirdOrder": ["single", "diphone", "triphone"],
                        "toSemAll": ["single", "diphone", "triphone","semantic"]}
MODEL_VP = {"single":["firstOrder"],
            "diphone": ["secondOrder", "firstOrder"],
            "triphone": ["thirdOrder", "secondOrder"],
            "semantic": ["toSemAll", "thirdOrder"]}
MODELS_ANALYSIS = ["thirdOrder", "single", "diphone", "triphone", "semantic"]

ROIS = ["wholeBrain", "ACunique", "STG", "STS", "LTCunique", "Broca"]
STATS_DATA = f"{DATA_DIR}/stats_data"

SIMULATION_DIR = f"{ROOT_DIR}/simulations"
SIMULATION_BOLD_PATH = f"{SIMULATION_DIR}/dataForModel.hf5"


FIG_DIR = f"{ROOT_DIR}/figures"

