from yacs.config import CfgNode as CN

_C = CN()


###System settings###
_C.SYSTEM = CN()
#Number of workers
_C.SYSTEM.NUM_WORKERS = 4
#Random seed number
_C.SYSTEM.RANDOM_SEED = 3


###Model settings###
_C.MODEL = CN()
#Checkpoint directory path
_C.MODEL.CHECKPOINT_DIR = 'cub_training/'
#Saved model name
_C.MODEL.EXPERIMENT = 'network.pth'


###Dataset parameters###
_C.DATASET = CN()
#ID map path
_C.DATASET.ID_MAP_PATH = 'cub_200.csv' 
#Dataset path
_C.DATASET.DATASET_PATH = 'CUB_200_2011/train_data'
#Fraction of data for validation
_C.DATASET.VAL_SIZE = 0.2
#Image size
_C.DATASET.IMG_SIZE = (224,224)
#Image mean and std
_C.DATASET.IMG_MEAN = (0.485, 0.456, 0.406)
_C.DATASET.IMG_STD = (0.229, 0.224, 0.225)
_C.DATASET.NUM_CHANNELS = 3


###Train parameters###
_C.TRAIN = CN()
#Batch size
_C.TRAIN.BATCH_SIZE = 64
#Margin value for triplet loss
_C.TRAIN.TRIPLET_LOSS_MARGIN = 0.7
#Number of training epochs
_C.TRAIN.NUM_EPOCHS = 100
#Number of epochs after which validation to be performed
_C.TRAIN.VAL_EPOCHS = 110
#Learning rate
_C.TRAIN.LEARNING_RATE = 0.1
#Weight decay
_C.TRAIN.WEIGHT_DECAY = 1e-4
#Gradient clipping
_C.TRAIN.GRAD_CLIP = 0.1
#False negative ratio
_C.TRAIN.FNR = 0.3
#Maximum number of allowed clusters
_C.TRAIN.MAX_CLUSTERS = 10

###Inference parameters###
_C.INF = CN()
#Path to ID test dataset
_C.INF.ID_TEST_DATASET = '/mnt/gpu_storage/aishwarya/thumbails/atlas/CUB_200_2011/test_data'
#Threshold on explained variance cumsum
_C.INF.EXP_VAR_THRESHOLD = 0.95



##Biological taxonomy parameters##
_C.HIER = CN()
_C.HIER.HIERARCHY_FILE_PATH='taxonomy_data/cub_parent_child.txt'




def get_cfg_defaults():

    return _C.clone()

