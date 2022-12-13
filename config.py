TRAIN_PATH = "data/coco/train2014"
VALID_PATH = "data/coco/val2014"
TEST_PATH = "data/coco/test2014"

TRAIN_SEG_PATH = "data/ADE20K/train"
VALID_SEG_PATH = "data/ADE20K/validation"
TEST_SEG_PATH = "data/ADE20K/test"
NUM_SEG_CLASSES = 151

IMG_SIZE = (128, 256)

try:
    from local_config import *
except ImportError:
    pass
