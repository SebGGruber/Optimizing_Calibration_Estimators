import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import torch
import pandas as pd
import pickle

from safetensors.numpy import load_file
from scipy.special import softmax
from sklearn.metrics import pairwise
from tqdm import tqdm

from ece_bin import ECE_bin
from CV_pipeline import ECE_bin_CV, ECE_krr_CV, ECE_kde_CV

# Open file with pickled variables
def unpickle_probs(file, verbose=0):
    with open(file, 'rb') as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)

    if verbose:
        print("y_preds_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_preds_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels

    return ((y_probs_val, y_val), (y_probs_test, y_test))

# top label calibration error
def TCE_experiments(settings, k_folds_val, k_folds_test, estimators=['bin', 'krr', 'kkrr', 'kde']):
    seed = 0
    exp_results = []
    for ds_name, model, logit_path in tqdm(settings):
        if logit_path[-2:]=='.p':
            ds = unpickle_probs(logit_path)
            logits = ds[1][0]
            labels = ds[1][1].squeeze(-1)
            logits_RC = None
        elif logit_path[-12:]=='.safetensors':
            ds = load_file(logit_path)
            logits = ds['logits_test']
            labels = ds['labels_test']
            logits_RC = None
        else:
            logits = np.load(logit_path + '/nll_logits.npy')
            labels = np.load(logit_path + '/nll_labels.npy').squeeze(-1)
            logits_RC = np.load(logit_path + '/nll_scores.npy')

        probs = softmax(logits, axis=1)
        top_confs = np.max(probs, axis=1)
        corrects = labels==np.argmax(probs, axis=1)
        if 'bin' in estimators:
            start_time = time.time()
            ece_est_bin = ECE_bin(Ps=top_confs, Ys=corrects)
            runtime_15b = time.time() - start_time
            start_time = time.time()
            ece_est_bin_opt, cv_bin_results = ECE_bin_CV(
                Ps=top_confs, Ys=corrects, k_folds_test=k_folds_test, k_folds_val=k_folds_val,
                seed=seed, disable_pb=True
            )
            pd_results = pd.DataFrame(cv_bin_results)
            if k_folds_test < 2:
                val_risk_15b = pd_results[pd_results['n_bins']==15]['val_risk'].mean()
                test_risk_15b = pd_results[pd_results['n_bins']==15]['test_risk'].mean()
                grp_results = pd_results.groupby(['n_bins']).mean()
                optim = grp_results['val_risk'].argmin()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            else:
                val_risk_15b = pd_results['val_risk_15b'].mean()
                val_risk = pd_results['val_risk'].mean()
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_bin,
                'Acc': corrects.mean(),
                'Type': '15 bins',
                'Val Risk': val_risk_15b,
                'Test Risk': test_risk_15b,
                'CV results': cv_bin_results,
                'Runtime': runtime_15b,
            }]
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_bin_opt,
                'Acc': corrects.mean(),
                'Type': 'Opt bins',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'CV results': cv_bin_results,
                'Runtime': time.time() - start_time,
            }]

        if 'kkrr' in estimators:
            start_time = time.time()
            ece_est_krr_opt, cv_krr_results = ECE_krr_CV(
                Ps=top_confs, Ys=corrects, k_folds_test=k_folds_test, k_folds_val=k_folds_val,
                disable_pb=True, use_kkrr=True, reg_range=10.**np.arange(-1, -10, -1), seed=seed
            )
            pd_results = pd.DataFrame(cv_krr_results)
            if k_folds_test < 2:
                grp_results = pd_results.groupby(['reg_const']).mean()
                optim = grp_results['val_risk'].argmin()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            else:
                val_risk = pd_results['val_risk'].mean()        
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_krr_opt,
                'Acc': corrects.mean(),
                'Type': 'Opt KKRR',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'CV results': cv_krr_results,
                'Runtime': time.time() - start_time,
            }]

        if 'krr' in estimators:
            start_time = time.time()
            ece_est_krr_opt, cv_krr_results = ECE_krr_CV(
                Ps=top_confs, Ys=corrects, k_folds_test=k_folds_test,
                k_folds_val=k_folds_val, disable_pb=True, seed=seed
            )
            pd_results = pd.DataFrame(cv_krr_results)
            if k_folds_test < 2:
                grp_results = pd_results.groupby(['reg_const']).mean()
                optim = grp_results['val_risk'].argmin()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            else:
                val_risk = pd_results['val_risk'].mean()        
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_krr_opt,
                'Acc': corrects.mean(),
                'Type': 'Opt KRR',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'CV results': cv_krr_results,
                'Runtime': time.time() - start_time,
            }]

        if 'kde' in estimators:
            start_time = time.time()
            ece_est_kde_opt, cv_kde_results = ECE_kde_CV(
                Ps=top_confs, Ys=corrects, k_folds_test=k_folds_test,
                k_folds_val=k_folds_val, disable_pb=True, seed=seed
            )
            pd_results = pd.DataFrame(cv_kde_results)
            if k_folds_test < 2:
                grp_results = pd_results.groupby(['bandwidth']).mean()
                optim = grp_results['val_risk'].argmin()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            else:
                val_risk = pd_results['val_risk'].mean()    
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_kde_opt,
                'Acc': corrects.mean(),
                'Type': 'Opt KDE',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'CV results': cv_kde_results,
                'Runtime': time.time() - start_time,
            }]
    return exp_results

# canonical calibration error
def CCE_experiments(settings, k_folds_val, k_folds_test, estimators=['krr', 'kkrr', 'kde']):
    seed = 0
    exp_results = []
    for ds_name, model, logit_path in tqdm(settings):
        if logit_path[-2:]=='.p':
            ds = unpickle_probs(logit_path)
            logits = ds[1][0]
            labels = ds[1][1].squeeze(-1)
            logits_RC = None
        else:
            logits = np.load(logit_path + '/nll_logits.npy')
            labels = np.load(logit_path + '/nll_labels.npy').squeeze(-1)
            logits_RC = np.load(logit_path + '/nll_scores.npy')

        probs = softmax(logits, axis=1)
        if 'kkrr' in estimators:
            ece_est_krr_opt, cv_krr_results = ECE_krr_CV(
                Ps=probs, Ys=labels, k_folds_test=k_folds_test, k_folds_val=k_folds_val,
                disable_pb=False, use_kkrr=True, reg_range=10.**np.arange(4, -5, -0.5), seed=seed
            )
            pd_results = pd.DataFrame(cv_krr_results)
            if k_folds_test < 2:
                grp_results = pd_results.groupby(['reg_const']).mean()
                optim = grp_results['val_risk'].argmin()
                nan_val = grp_results.iloc[optim]['nan_val'].item()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_krr_opt,
                'Type': 'Opt KKRR',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'NaN frac Val': nan_val,
                'CV results': cv_krr_results,
            }]

        if 'krr' in estimators:
            ece_est_krr_opt, cv_krr_results = ECE_krr_CV(
                Ps=probs, Ys=labels, k_folds_test=k_folds_test, reg_range=10.**np.arange(8, -10, -1),
                k_folds_val=k_folds_val, disable_pb=False, seed=seed
            )
            pd_results = pd.DataFrame(cv_krr_results)
            if k_folds_test < 2:
                grp_results = pd_results.groupby(['reg_const']).mean()
                optim = grp_results['val_risk'].argmin()
                nan_val = grp_results.iloc[optim]['nan_val'].item()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_krr_opt,
                'Type': 'Opt KRR',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'NaN frac Val': nan_val,
                'CV results': cv_krr_results,
            }]

        if 'kde' in estimators:
            ece_est_kde_opt, cv_kde_results = ECE_kde_CV(
                Ps=probs, Ys=labels, k_folds_test=k_folds_test,
                k_folds_val=k_folds_val, disable_pb=False, seed=seed
            )
            pd_results = pd.DataFrame(cv_kde_results)
            if k_folds_test < 2:
                grp_results = pd_results.groupby(['bandwidth']).mean()
                optim = grp_results['val_risk'].argmin()
                nan_val = grp_results.iloc[optim]['nan_val'].item()
                val_risk = grp_results.iloc[optim]['val_risk'].item()
                test_risk = grp_results.iloc[optim]['test_risk'].item()
            exp_results += [{
                'Dataset': ds_name,
                'Model': model,
                'ECE estimate': ece_est_kde_opt,
                'Type': 'Opt KDE',
                'Val Risk': val_risk,
                'Test Risk': test_risk,
                'NaN frac Val': nan_val,
                'CV results': cv_kde_results,
            }]
    return exp_results
