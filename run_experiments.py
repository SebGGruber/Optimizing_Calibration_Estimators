import pandas as pd
import numpy as np
import torch
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from scipy.special import softmax
from sklearn.metrics import pairwise
from tqdm import tqdm

from ece_bin import ECE_bin
from CV_pipeline import ECE_bin_CV
from experiments import TCE_experiments, CWCE_experiments, CCE_experiments, unpickle_probs

notion = 'tce'

settings_c10 = [
    ('Cifar10', 'LeNet-5', 'logits/probs_lenet5_c10_logits.p'),
    ('Cifar10', 'Densenet-40', 'logits/probs_densenet40_c10_logits.p'),
    ('Cifar10', 'ResNetWide-32', 'logits/probs_resnet_wide32_c10_logits.p'),
    ('Cifar10', 'Resnet-110', 'logits/probs_resnet110_c10_logits.p'),
    ('Cifar10', 'Resnet-110 SD', 'logits/probs_resnet110_SD_c10_logits.p'),
]
settings_c100 = [
    ('Cifar100', 'LeNet-5', 'logits/probs_lenet5_c100_logits.p'),
    ('Cifar100', 'Densenet-40', 'logits/probs_densenet40_c100_logits.p'),
    ('Cifar100', 'ResNetWide-32', 'logits/probs_resnet_wide32_c100_logits.p'),
    ('Cifar100', 'Resnet-110', 'logits/probs_resnet110_c100_logits.p'),
    ('Cifar100', 'Resnet-110 SD', 'logits/probs_resnet110_SD_c100_logits.p'),
]

if notion=='tce':
    settings_imgnet = [
        ('ImageNet', 'DenseNet-161', 'logits/diag_densenet161_imgnet'),
        ('ImageNet', 'Resnet-152', 'logits/diag_resnet152_imgnet'),
        ('ImageNet', 'Pnasnet-5', 'logits/diag_pnasnet5_large_imgnet'),
    ]
else:
    settings_imgnet = []


if notion=='tce':
    exp_results = TCE_experiments(
        settings_c10+settings_c100+settings_imgnet, k_folds_val=5, k_folds_test=1
    )
    filename = 'results/real_data/TCE_binning_krr_kkrr_kde.pkl'
elif notion=='cce':
    exp_results = CCE_experiments(
        settings_c10+settings_c100+settings_imgnet, k_folds_val=5, k_folds_test=1,
        estimators=['kkrr', 'krr', 'kde']
    )
    filename = 'results/real_data/CCE_krr_kkrr_kde.pkl'

with open(filename, 'wb') as file:
    pickle.dump(exp_results, file)