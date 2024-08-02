from easydict import EasyDict as edict

__C = edict()
cfg = __C

# YOLO options
__C.YOLO = edict()

# Update the path to your custom classes file
__C.YOLO.CLASSES = "./data/compostdataset/classes.txt"

# Keep these if the anchors are suitable for your dataset
__C.YOLO.ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.XYSCALE = [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5

# Train options
__C.TRAIN = edict()

# Update the path to your training annotations file
__C.TRAIN.ANNOT_PATH = "./data/compostdataset/train.txt"

__C.TRAIN.BATCH_SIZE = 2
__C.TRAIN.INPUT_SIZE = 416
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FIRST_STAGE_EPOCHS = 20
__C.TRAIN.SECOND_STAGE_EPOCHS = 30

# Test options
__C.TEST = edict()

# Update the path to your validation/test annotations file
__C.TEST.ANNOT_PATH = "./data/compostdataset/valid.txt"

__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD = 0.25
__C.TEST.IOU_THRESHOLD = 0.5
