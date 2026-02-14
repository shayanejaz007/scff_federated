"""SCFF - Self-Contrastive Forward-Forward Learning"""
from .layers import Conv2d, triangle, standardnorm, L2norm, stdnorm
from .data import DualAugmentCIFAR10, DualAugmentCIFAR10_test, get_train
from .model import build_layers_from_config, get_pos_neg_batch_imgcats
from .trainer import SCFFTrainer, EvaluationConfig
from .eval import evaluate_model, train_readout, test_readout
