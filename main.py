from config import CFG
from utils import init_logger, set_seed, cfg_init
from train import ink_classifier_train_fold, segmentation_model_train_fold
import torch


if __name__ == "__main__":
    cv = 0
    ths = 0
    cfg_init(CFG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG.exp_name = "trained"
    CFG.use_pretrained = True
    CFG.model = "3D-2D"
    Logger = init_logger(log_file=f"{CFG.log_path}.txt")
    Logger.info(f"exp_id: {CFG.exp_name}")
    set_seed(CFG.seed)
    for idx, fold in enumerate(CFG.folds):
        score, th = segmentation_model_train_fold(fold, Logger, device)
        cv += score
        ths += th

    Logger.info('----------------- CV -----------------')
    Logger.info(f"cv_score: {cv / len(CFG.folds)} cv_th: {ths / len(CFG.folds)}")
