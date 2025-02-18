import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from sklearn.metrics import pairwise
from tqdm import tqdm

from losses import ce_risk
from ece_bin import ECE_bin_fit
from ece_kde import ECE_kde_fit
from KRR_estimators import ECE_krr_fit, ECE_kkrr_fit

def ECE_bin_CV(
    Ps, Ys, bin_range=np.arange(5, 95, 5), k_folds_val=5, k_folds_test=1, disable_pb=False, seed=0
):
    # use 20% as test set and do CV on the other 80%
    if k_folds_test < 2:
        n_all = Ps.shape[0]
        np.random.seed(seed)
        shuffle_ids = np.random.choice(n_all, n_all, replace=False)
        Ps = Ps[shuffle_ids]
        Ys = Ys[shuffle_ids]
        split = int(n_all*0.8)
        Ps_train_val = Ps[:split]
        Ys_train_val = Ys[:split]
        Ps_test = Ps[split:]
        Ys_test = Ys[split:]
        #test_ind = np.random.binomial(n=1, p=0.2, size=n_all)
        #Ps_train_val = Ps[test_ind==0]
        #Ys_train_val = Ys[test_ind==0]
        #Ps_test = Ps[test_ind==1]
        #Ys_test = Ys[test_ind==1]
        kf = KFold(n_splits=k_folds_val)
        val_results = []
        fold = 0
        for train_ind, val_ind in tqdm(kf.split(Ps_train_val), total=k_folds_val, disable=disable_pb):
            Ps_train = Ps_train_val[train_ind]
            Ys_train = Ys_train_val[train_ind]
            Ps_val = Ps_train_val[val_ind]
            Ys_val = Ys_train_val[val_ind]
            for n_bins in bin_range:
                cal_model = ECE_bin_fit(Ps_train, Ys_train, n_bins)
                cal_preds_val = cal_model(Ps_val)
                val_risk = ce_risk(Ps_val, Ys_val, cal_preds_val)
                avg_pred_val = np.diag(cal_preds_val).mean()
                cal_preds_test = cal_model(Ps_test)
                test_risk = ce_risk(Ps_test, Ys_test, cal_preds_test)
                avg_pred_test = np.diag(cal_preds_test).mean()
                val_results += [{
                    'n_bins': n_bins, 'fold': fold, 'val_risk': val_risk, 'test_risk': test_risk,
                    'avg_pred_val': avg_pred_val, 'avg_pred_test': avg_pred_test
                }]
            fold += 1
        pd_results = pd.DataFrame(val_results).groupby(['n_bins']).mean()
        optim = pd_results['val_risk'].argmin()
        ece_estimate = pd_results.iloc[optim]['avg_pred_test'].item()
        return ece_estimate, val_results

def CWCE_bin_CV(Ps, Ys, **kwargs):
    n_classes = Ps.shape[1]
    one_hot_Ys = np.zeros((Ys.shape[0], n_classes))
    one_hot_Ys[np.arange(Ys.shape[0]), Ys] = 1
    CWCE_est = 0
    for k in range(n_classes):
        ce_est, _ = ECE_bin_CV(Ps[:, k], one_hot_Ys[:, k], **kwargs)
        CWCE_est += ce_est

    return CWCE_est

def ECE_krr_CV(
    Ps, Ys, reg_range=10.**np.arange(-1, -18, -2), k_folds_val=5, k_folds_test=1,
    disable_pb=False, kernel_X=lambda x,y: pairwise.rbf_kernel(x, y, gamma=1), seed=0,
    use_kkrr=False
):
    if len(Ps.shape) > 1 and Ps.shape[1] > 1:
        is_binary = False
        Y_onehot = np.zeros((Ys.size, Ys.max() + 1))
        Y_onehot[np.arange(Ys.size), Ys] = 1
    else:
        is_binary = True

    ECE_model_fit = ECE_kkrr_fit if use_kkrr else ECE_krr_fit
    # use 20% as test data and do CV with the other 80%
    if k_folds_test < 2:
        n_all = Ps.shape[0]
        np.random.seed(seed)
        shuffle_ids = np.random.choice(n_all, n_all, replace=False)
        Ps = Ps[shuffle_ids]
        Ys = Ys[shuffle_ids]
        split = int(n_all*0.8)
        Ps_train_val = Ps[:split]
        Ys_train_val = Ys[:split]
        Ps_test = Ps[split:]
        Ys_test = Ys[split:]
        if is_binary:
            K_XX = kernel_X(Ps.reshape(-1, 1), Ps.reshape(-1, 1))
            K_YY = np.outer(Ps-Ys, Ps-Ys)
        else:
            K_XX = kernel_X(Ps, Ps)
            K_YY = (Ps-Y_onehot) @ (Ps-Y_onehot).T
        K_XX_train_val_train_val = K_XX[:split, :][:, :split]
        K_YY_train_val_train_val = K_YY[:split, :][:, :split]
        K_XX_train_val_test = K_XX[:split, :][:, split:]
        #test_ind = np.random.binomial(n=1, p=0.2, size=n_all)
        #Ps_train_val = Ps[test_ind==0]
        #Ys_train_val = Ys[test_ind==0]
        #Ps_test = Ps[test_ind==1]
        #Ys_test = Ys[test_ind==1]
        # K_XX_train_val_train_val = K_XX[test_ind==0, :][:, test_ind==0]
        # K_YY_train_val_train_val = K_YY[test_ind==0, :][:, test_ind==0]
        # K_XX_train_val_test = K_XX[test_ind==0, :][:, test_ind==1]
        kf = KFold(n_splits=k_folds_val)
        val_results = []
        fold = 0
        for train_ind, val_ind in tqdm(kf.split(Ps_train_val), total=k_folds_val, disable=disable_pb):
            Ps_train = Ps_train_val[train_ind]
            Ys_train = Ys_train_val[train_ind]
            Ps_val = Ps_train_val[val_ind]
            Ys_val = Ys_train_val[val_ind]
            K_XX_train_train = K_XX_train_val_train_val[train_ind, :][:, train_ind]
            K_YY_train_train = K_YY_train_val_train_val[train_ind, :][:, train_ind]
            K_XX_train_val = K_XX_train_val_train_val[train_ind, :][:, val_ind]
            K_XX_train_test = K_XX_train_val_test[train_ind, :]
            cal_model = ECE_model_fit(K_XX_train_train, K_YY_train_train)
            for reg_const in reg_range:
                cal_preds_val = cal_model(K_XX_train_val, reg_const)
                nan_val = np.isnan(cal_preds_val).mean()
                cal_preds_test = cal_model(K_XX_train_test, reg_const)
                val_risk = ce_risk(Ps_val, Ys_val, cal_preds_val)
                test_risk = ce_risk(Ps_test, Ys_test, cal_preds_test)
                avg_pred_val = np.diag(cal_preds_val).mean()
                avg_pred_test = np.diag(cal_preds_test).mean()
                val_results += [{
                    'reg_const': reg_const, 'fold': fold, 'val_risk': val_risk, 'test_risk': test_risk,
                    'avg_pred_val': avg_pred_val, 'avg_pred_test': avg_pred_test, 'nan_val': nan_val,
                }]
            fold += 1
        pd_results = pd.DataFrame(val_results).groupby(['reg_const']).mean()
        optim = pd_results['val_risk'].argmin()
        ece_estimate = pd_results.iloc[optim]['avg_pred_test']
        return ece_estimate, val_results

def ECE_kde_CV(
    Ps, Ys, bw_range=torch.cat((torch.logspace(start=-5, end=-1, steps=15), torch.linspace(0.2, 1, steps=5))), seed=0, disable_pb=False, k_folds_val=5, k_folds_test=1
):
    if len(Ps.shape) > 1 and Ps.shape[1] > 1:
        is_binary = False
    else:
        is_binary = True

    # use 20% as test data and do CV with the other 80%
    if k_folds_test < 2:
        n_all = Ps.shape[0]
        np.random.seed(seed)
        shuffle_ids = np.random.choice(n_all, n_all, replace=False)
        Ps = Ps[shuffle_ids]
        Ys = Ys[shuffle_ids]
        split = int(n_all*0.8)
        Ps_train_val = Ps[:split]
        Ys_train_val = Ys[:split]
        Ps_test = Ps[split:]
        Ys_test = Ys[split:]
        kf = KFold(n_splits=k_folds_val)
        val_results = []
        fold = 0
        for train_ind, val_ind in tqdm(kf.split(Ps_train_val), total=k_folds_val, disable=disable_pb):
            Ps_train = Ps_train_val[train_ind]
            Ys_train = Ys_train_val[train_ind]
            Ps_val = Ps_train_val[val_ind]
            Ys_val = Ys_train_val[val_ind]
            for bw in bw_range:
                cal_model = ECE_kde_fit(Ps_train, Ys_train, bw=bw)
                cal_preds_val = cal_model(Ps_val)
                nan_val = np.isnan(cal_preds_val).mean()
                val_risk = ce_risk(Ps_val, Ys_val, cal_preds_val, ignore_nan=True)
                avg_pred_val = np.nanmean(np.diag(cal_preds_val))
                cal_preds_test = cal_model(Ps_test)
                test_risk = ce_risk(Ps_test, Ys_test, cal_preds_test, ignore_nan=True)
                avg_pred_test = np.nanmean(np.diag(cal_preds_test))
                val_results += [{
                    'bandwidth': bw, 'fold': fold, 'val_risk': val_risk, 'test_risk': test_risk,
                    'avg_pred_val': avg_pred_val, 'avg_pred_test': avg_pred_test, 'nan_val': nan_val,
                }]
            fold += 1
        pd_results = pd.DataFrame(val_results).groupby(['bandwidth']).mean()
        optim = pd_results['val_risk'].argmin()
        ece_estimate = pd_results.iloc[optim]['avg_pred_test'].item()
        return ece_estimate, val_results

