# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_bsgal_config(cfg):
    _C = cfg

    _C.WITH_IMAGE_LABELS = False # Turn on co-training with classification data

    # Open-vocabulary classifier
    _C.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = False # Use fixed classifier for open-vocabulary detection
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    _C.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    _C.MODEL.ROI_BOX_HEAD.NORM_TEMP = 50.0
    _C.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    _C.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0 # >= 0: not use
    
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False # CenterNet2
    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False # Federated Loss
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/metadata/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    # Classification data configs
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = 'max_size' # max, softmax, sum
    _C.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT = 0.1
    _C.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE = 1.0
    _C.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX = False # Used for image-box loss and caption loss
    _C.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 128 # num proposals for image-labeled data
    _C.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP = False # Used for WSDDN
    _C.MODEL.ROI_BOX_HEAD.CAPTION_WEIGHT = 1.0 # Caption loss weight
    _C.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT = 0.125 # Caption loss hyper-parameter
    _C.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP = False # Used for WSDDN
    _C.MODEL.ROI_BOX_HEAD.SOFTMAX_WEAK_LOSS = False # Used when USE_SIGMOID_CE is False

    _C.MODEL.ROI_HEADS.MASK_WEIGHT = 1.0
    _C.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False # For demo only
    _C.MODEL.ROI_HEADS.SEG_IN_FEATURE = 'p3'

    _C.MODEL.REFINE_MASK = CN()
    _C.MODEL.REFINE_MASK.SEMANTIC_OUT_STRIDE = 8

    # Caption losses
    _C.MODEL.CAP_BATCH_RATIO = 4 # Ratio between detection data and caption data
    _C.MODEL.WITH_CAPTION = False
    _C.MODEL.SYNC_CAPTION_BATCH = False # synchronize across GPUs to enlarge # "classes"

    # dynamic class sampling when training with 21K classes
    _C.MODEL.DYNAMIC_CLASSIFIER = False
    _C.MODEL.NUM_SAMPLE_CATS = 50
    _C.MODEL.ACTIVE_MODE = 'paste_or_zero'
    _C.MODEL.ACTIVE_LOSS = 'cls'
    _C.MODEL.ACTIVE_LOSS_UPDATE = 'all'
    _C.MODEL.ACTIVE_SEED = 0
    _C.MODEL.ACTIVE_COMPARE = 'default'
    _C.MODEL.ACTIVE_TEST = 'select'
    _C.MODEL.ACTIVE_TEST_INS = 'one'
    _C.MODEL.ACTIVE_LR = 0.0001
    _C.MODEL.ACTIVE_OPTIMIZER = True
    _C.MODEL.ACTIVE_OPTIMIZER_MODE = 'sgd' # 'sgd', 'adam' 
    _C.MODEL.ACTIVE_PRED = False
    _C.MODEL.ACTIVE_PRED_CHOOSE = ''  # 'max', 'mean', 'all', 'random'
    _C.MODEL.ACTIVE_PRED_SUP = 'all' # 'all', 'one'
    _C.MODEL.ACTIVE_ONLY_GT_TRAIN = False
    _C.MODEL.ACTIVE_ONLY_GT_TEST = False
    _C.MODEL.ACTIVE_GRAD_COMPARE = False
    _C.MODEL.ACTIVE_GRAD_NORM = True
    _C.MODEL.ACTIVE_GRAD_SAVE = False
    _C.MODEL.ACTIVE_GRAD_UPDATE = "AVERAGE" # "AVERAGE", "MOMENTUM0.9", "MOMENTUM0.99"
    # 'only_paste_sup': cfg.MODEL.ONLY_PASTE_SUP,
    _C.MODEL.ONLY_PASTE_SUP = False
    _C.MODEL.ACTIVE_FORWARD_ONCE = False
    _C.MODEL.ACTIVE_ONCE_MODE = 'only_gt'
    _C.MODEL.ACTIVE_EVAL = False
    _C.MODEL.ACTIVE_DYNAMIC_THRES = 0.0
    _C.MODEL.ACTIVE_TEST_BATCHSIZE = 4
    # Different classifiers in testing, used in cross-dataset evaluation
    _C.MODEL.RESET_CLS_TESTS = False
    _C.MODEL.TEST_CLASSIFIERS = []
    _C.MODEL.TEST_NUM_CLASSES = []

    # Backbones
    # if _C.MODEL has no SWIN, add it
    if not hasattr(_C.MODEL, 'SWIN'):
        _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.SIZE = 'T' # 'T', 'S', 'B'
    _C.MODEL.SWIN.USE_CHECKPOINT = False
    _C.MODEL.SWIN.OUT_FEATURES = (1, 2, 3) # FPN stride 8 - 32

    _C.MODEL.TIMM = CN()
    _C.MODEL.TIMM.BASE_NAME = 'resnet50'
    _C.MODEL.TIMM.OUT_LEVELS = (3, 4, 5)
    _C.MODEL.TIMM.NORM = 'FrozenBN'
    _C.MODEL.TIMM.FREEZE_AT = 0
    _C.MODEL.DATASET_LOSS_WEIGHT = []
    
    _C.DATASETS.SEMANTIC_ROOT = ''
    _C.DATASETS.VAL = ("lvis_v1_val",)

    # Multi-dataset dataloader
    _C.DATALOADER.DATASET_RATIO = [1, 1] # sample ratio
    _C.DATALOADER.USE_RFS = [False, False]
    _C.DATALOADER.MULTI_DATASET_GROUPING = False # Always true when multi-dataset is enabled
    _C.DATALOADER.DATASET_ANN = ['box', 'box'] # Annotation type of each dataset
    _C.DATALOADER.USE_DIFF_BS_SIZE = False # Use different batchsize for each dataset
    _C.DATALOADER.DATASET_BS = [8, 32] # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SIZE = [896, 384] # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_INPUT_SCALE = [(0.1, 2.0), (0.5, 1.5)] # Used when USE_DIFF_BS_SIZE is on 
    _C.DATALOADER.DATASET_MIN_SIZES = [(640, 800), (320, 400)] # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.DATASET_MAX_SIZES = [1333, 667] # Used when USE_DIFF_BS_SIZE is on
    _C.DATALOADER.USE_TAR_DATASET = False # for ImageNet-21K, directly reading from unziped files
    _C.DATALOADER.TARFILE_PATH = 'datasets/imagenet/metadata-22k/tar_files.npy'
    _C.DATALOADER.TAR_INDEX_DIR = 'datasets/imagenet/metadata-22k/tarindex_npy'
    _C.DATALOADER.PREFETCH_FACTOR = 2
    
    _C.SOLVER.USE_CUSTOM_SOLVER = False
    _C.SOLVER.OPTIMIZER = 'SGD'
    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0 # Used in DETR
    _C.SOLVER.CUSTOM_MULTIPLIER = 1.0 # Used in DETR
    _C.SOLVER.CUSTOM_MULTIPLIER_NAME = [] # Used in DETR
    _C.SOLVER.MODEL_EMA = 0.0

    # Deformable DETR
    _C.MODEL.DETR = CN()
    _C.MODEL.DETR.NUM_CLASSES = 80
    _C.MODEL.DETR.FROZEN_WEIGHTS = '' # For Segmentation
    _C.MODEL.DETR.GIOU_WEIGHT = 2.0
    _C.MODEL.DETR.L1_WEIGHT = 5.0
    _C.MODEL.DETR.DEEP_SUPERVISION = True
    _C.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1
    _C.MODEL.DETR.CLS_WEIGHT = 2.0
    _C.MODEL.DETR.NUM_FEATURE_LEVELS = 4
    _C.MODEL.DETR.TWO_STAGE = False
    _C.MODEL.DETR.WITH_BOX_REFINE = False
    _C.MODEL.DETR.FOCAL_ALPHA = 0.25
    _C.MODEL.DETR.NHEADS = 8
    _C.MODEL.DETR.DROPOUT = 0.1
    _C.MODEL.DETR.DIM_FEEDFORWARD = 2048
    _C.MODEL.DETR.ENC_LAYERS = 6
    _C.MODEL.DETR.DEC_LAYERS = 6
    _C.MODEL.DETR.PRE_NORM = False
    _C.MODEL.DETR.HIDDEN_DIM = 256
    _C.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    _C.MODEL.DETR.USE_FED_LOSS = False
    _C.MODEL.DETR.WEAK_WEIGHT = 0.1

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 

    _C.INPUT.SCP_RFS = False
    _C.INPUT.RFS_VERSION = 0 #
    _C.INPUT.RM_BG_PROB = -1.0
    _C.INPUT.USE_SCP = True
    _C.INPUT.VIS_RESULT = False
    _C.INPUT.SCP_SRC_OBJ_SELECT = True
    _C.INPUT.SCP_NUM_SRC = 1
    _C.INPUT.LOG_SCP_PARAM = False
    _C.INPUT.BLANK_RATIO = -1.
    _C.INPUT.SCP_TYPE = '' #
    _C.INPUT.USE_COPY_METHOD = 'none'
    _C.INPUT.INST_POOL = False
    _C.INPUT.INST_POOL_FEED = False
    _C.INPUT.INST_POOL_PATH = 'datasets/lvis/inst_anno.json'
    _C.INPUT.INST_POOL_ROOT = 'datasets/lvis/inst_dir'
    _C.INPUT.INST_POOL_SAMPLE_TYPE = 'random'
    _C.INPUT.INST_POOL_AREA_STD_THRES = 0.0
    _C.INPUT.INST_POOL_AREA_CERTAINTY = 0.0
    _C.INPUT.INST_POOL_FORMAT = 'RGB'
    _C.INPUT.INST_POOL_MAX_SAMPLES = 20
    _C.INPUT.ACTIVE_SELECT = False
    _C.INPUT.ACTIVE_SELECT_TYPE = 'train'
    _C.INPUT.ROUND_RESET = True
    _C.INPUT.INIT_ROUND = False
    _C.INPUT.NUM_ROUNDS = 10
    _C.INPUT.SEPARATE_SYN = False
    _C.INPUT.SEPERATE_SUP = False
    _C.INPUT.USE_COLOR_JITTER = False
    _C.INPUT.COLOR_JITTER_FREQ_FILTER = ['r']
    _C.INPUT.COLOR_JITTER_APPLY_TYPE = 'both' # 'src' 'dst'
    _C.INPUT.COLOR_JITTER_USE_TORCHVISION = False
    _C.INPUT.LIMIT_SRC_LSJ = False
    _C.USE_LARGEST_PART = True
    _C.INPUT.CP_METHOD= ['basic']
    _C.INPUT.RANDOM_ROTATE=False
    _C.INPUT.COLOR_AUG= False
    _C.INPUT.ONLY_RC = False
    _C.INPUT.SELECT_CATS_LIST = []
    _C.INPUT.SCP_SELECT_CATS_LIST = []
    # Use instaboost; default False
    _C.INPUT.USE_INSTABOOST = False
    _C.INPUT.INSTABOOST_FREQ = ['r', 'c', 'f']
    _C.INPUT.INSTABOOST_APPLY_TYPE = 'both'
    _C.INPUT.TRANSITION_MATRIX_PATH = ''

    _C.INPUT.USE_INP_ROTATE = False
    _C.INPUT.INP_ROOT = 'datasets/lvis/lvis_rare'
    _C.INPUT.INP_ANNO = 'datasets/metadata/img_wo_rare.json'
    _C.INPUT.INP_ROTATE_ANG = 30
    _C.INPUT.INP_ROTATE_LIMIT = False
    _C.INPUT.ROTATE_SRC = False
    _C.MODEL.USE_XPASTE_BOX_LOSS=True ## for abalation
    _C.MODEL.USE_XPASTE_MASK_LOSS=True

    _C.FIND_UNUSED_PARAM = True
    _C.EVAL_PRED_AR = False
    _C.EVAL_PROPOSAL_AR = False
    _C.EVAL_CAT_SPEC_AR = False
    _C.IS_DEBUG = False
    _C.QUICK_DEBUG = False
    _C.FP16 = False
    _C.EVAL_AP_FIX = False
    _C.GEN_PSEDO_LABELS = False
    _C.SAVE_DEBUG_PATH = 'output/save_debug/'

    _C.TEST.ANALYSE = False
    _C.TEST.GEN_DATASET = False
