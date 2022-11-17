import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
from prj_utils.main_utils import main_train , main_validate

torch.manual_seed(0)

# main_validate("0")
main_train("0")