import pandas as pd
import numpy as np
import torch
import itertools

#from pathos.pools import ProcessPool
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

from kernel_density_ratio import kde_ratio_estimator
from KRR_estimators import (
    inner_prod_MCE_split, inner_prod_MCE_full, inner_prod_QMCE_split, inner_prod_QMCE_full,
    inner_prod_QMCE_asymp, inner_prod_QMCE_CV, QMCE_fold_fit, delta_kernel, slow_CV_QMCE_fit
)
from ece_kde import get_bandwidth, get_ece_kde


def ground_truth_sim(X, U=None, temp=1):
    XU = X if U is None else np.concatenate([X, U])
    XU = XU**temp / (XU**temp).sum(axis=1, keepdims=True)
    norms = np.linalg.norm(XU, axis=1)**2
    return norms.mean()

def ground_truth_CE_sim(X, U=None, temp=1):
    XU = X if U is None else np.concatenate([X, U])
    XU_temp = XU**temp / (XU**temp).sum(axis=1, keepdims=True)
    norms = np.linalg.norm(XU - XU_temp, axis=1)**2
    return norms.mean()

def dir_noise_term(dir_alpha, n_classes):
    """ did the math before"""
    alphas = np.repeat(dir_alpha, n_classes)
    alpha_0 = alphas.sum()
    denom = alpha_0**2 * (alpha_0+1)**2
    first_moment_term = (alphas**2).sum() * (alpha_0+1)**2
    snd_moment_term = np.sum([
        ((i==j)*alphas[i] + alphas[i]*alphas[j])**2
        for i in range(n_classes) for j in range(n_classes)
    ])
    return (first_moment_term - snd_moment_term) / denom

def inner_prod_estimator_simple_simulation(seeds, data_size, p_true, estimates):
    results = []
    for seed in tqdm(seeds):
        Ys = np.random.choice(len(p_true), p=p_true, size=data_size)
        for estimate in estimates:
            loss_values = (delta_kernel(Ys) - estimate)**2
            emp_risk = (loss_values.sum() - np.diag(loss_values).sum())/(data_size**2 - data_size)
            results.append({'Seed': seed, 'Estimate': estimate, 'Data size': data_size, 'Emp. Risk': emp_risk})
    return pd.DataFrame(results)

def inner_prod_estimator_cond_simulation(seeds, dir_alpha, n_classes, data_size, T_estimates):
    results = []
    for seed in tqdm(seeds):
        Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=data_size)
        Ys = np.array([np.random.choice(n_classes, p=X) for X in Xs])
        for temp in T_estimates:
            Xs_temp = Xs**temp / (Xs**temp).sum(axis=1, keepdims=True)
            loss_values = (delta_kernel(Ys) - pairwise.linear_kernel(Xs_temp))**2
            emp_risk = (loss_values.sum() - np.diag(loss_values).sum())/(data_size**2 - data_size)
            results.append({
                'Seed': seed, 'Temp. Estimate': est, 'Data size': data_size, 'Emp. Risk': emp_risk
            })
    return pd.DataFrame(results)

def ce_const_simulation(seeds, data_size, estimates, p_true_1, p_true_2, p_1, p_2):
    results = []
    for seed in tqdm(seeds):
        Y_1s = np.random.choice(len(p_true_1), p=p_true_1, size=data_size)
        Y_2s = np.random.choice(len(p_true_2), p=p_true_2, size=data_size)
        one_hot_Y_1s = np.zeros((Y_1s.shape[0], len(p_true_1)))
        one_hot_Y_2s = np.zeros((Y_2s.shape[0], len(p_true_2)))
        one_hot_Y_1s[np.arange(Y_1s.shape[0]), Y_1s] = 1
        one_hot_Y_2s[np.arange(Y_2s.shape[0]), Y_2s] = 1
        k_target = pairwise.linear_kernel(p_1-one_hot_Y_1s, p_2-one_hot_Y_2s)
        for estimate in estimates:
            loss_values = (k_target - estimate)**2
            emp_risk = (loss_values.sum() - np.diag(loss_values).sum())/(data_size**2 - data_size)
            results.append({
                'Seed': seed, 'Estimate': estimate, 'Data size': data_size, 'Emp. Risk': emp_risk
            })
    return pd.DataFrame(results)

def ce_estimator_simulation(seeds, dir_alpha, n_classes, data_size, T_estimates, temp=2):
    results = []
    for seed in tqdm(seeds):
        Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=data_size)
        Ys = np.array([np.random.choice(n_classes, p=X) for X in Xs])
        one_hot_Ys = np.zeros((Ys.shape[0], n_classes))
        one_hot_Ys[np.arange(Ys.shape[0]), Ys] = 1
        Xs_temp = Xs**temp / (Xs**temp).sum(axis=1, keepdims=True)
        for est in T_estimates:
            Xs_est = Xs**est / (Xs**est).sum(axis=1, keepdims=True)
            loss_values = (
                pairwise.linear_kernel(Xs_temp-one_hot_Ys) - pairwise.linear_kernel(Xs_temp-Xs_est)
            )**2
            emp_risk = (loss_values.sum() - np.diag(loss_values).sum())/(data_size**2 - data_size)
            results.append({
                'Seed': seed, 'Temp. Estimate': est, 'Data size': data_size, 'Emp. Risk': emp_risk
            })
    return pd.DataFrame(results)

def ce_estimator_lin_simulation(seeds, dir_alpha, n_classes, data_size, T_estimates, temp=2):
    results = []
    for seed in tqdm(seeds):
        Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=data_size)
        Ys = np.array([np.random.choice(n_classes, p=X) for X in Xs])
        one_hot_Ys = np.zeros((Ys.shape[0], n_classes))
        one_hot_Ys[np.arange(Ys.shape[0]), Ys] = 1
        Xs_temp = Xs**temp / (Xs**temp).sum(axis=1, keepdims=True)
        for est in T_estimates:
            Xs_est = Xs**est / (Xs**est).sum(axis=1, keepdims=True)
            emp_risk = np.mean([
                (np.inner(Xs_temp[i]-one_hot_Ys[i], Xs_temp[i+1]-one_hot_Ys[i+1])
                 - np.inner(Xs_temp[i]-Xs_est[i], Xs_temp[i+1]-Xs_est[i+1]))**2
                for i in range(data_size-1)
            ])
            results.append({
                'Seed': seed, 'Temp. Estimate': est, 'Data size': data_size, 'Emp. Risk': emp_risk
            })
    return pd.DataFrame(results)

def inner_prod_estimator_simulation_iter(seed, data_size, alpha_class, reg_const, temp, kernel_X):
    dir_alpha, n_classes = alpha_class
    n_UV = n_XY = int(data_size/2)
    np.random.seed(seed)
    Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=n_XY)
    Us = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=n_UV)
    # optional temperature scaling
    Ys = np.array([np.random.choice(n_classes, p=X**temp/np.sum(X**temp)) for X in Xs])
    Vs = np.array([np.random.choice(n_classes, p=U**temp/np.sum(U**temp)) for U in Us])
    XUs = np.concatenate([Xs, Us])
    YVs = np.concatenate([Ys, Vs])
    MCE_est_split = inner_prod_MCE_split(Ys, Xs, Vs, Us, reg_const=reg_const, kernel_X=kernel_X)
    #MCE_est_alt = inner_prod_MCE_alt(YVs, XUs, reg_const=reg_const, kernel_X=kernel_X)
    QMCE_est_split = inner_prod_QMCE_split(Ys, Xs, Vs, Us, reg_const=reg_const**2, kernel_X=kernel_X)
    MCE_est_full = inner_prod_MCE_full(YVs, XUs, reg_const=reg_const, kernel_X=kernel_X)
    QMCE_est_full = inner_prod_QMCE_full(YVs, XUs, reg_const=reg_const**2, kernel_X=kernel_X)

    return MCE_est_split, QMCE_est_split, MCE_est_full, QMCE_est_full#, MCE_est_alt

def inner_prod_estimator_simulation(seeds, data_sizes, alpha_classes, reg_consts, temp=1, kernel_X=pairwise.linear_kernel):

    results_df = []
    for alpha_class in alpha_classes:
        dir_alpha, n_classes = alpha_class
        # monte carlo simulate ground truth
        samples = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=10000)
        ground_truth = ground_truth_sim(samples, temp=temp)

        for seed in tqdm(seeds):
            for reg_const in reg_consts:
                for data_size in data_sizes:
                    MCE_est_split, QMCE_est_split, MCE_est_full, QMCE_est_full = inner_prod_estimator_simulation_iter(
                        seed, data_size, alpha_class, reg_const, temp, kernel_X=kernel_X
                    )
                    results_df += [
                        {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': MCE_est_split,
                            'Temp': temp, 'Type': 'Plug-In (split)'
                        }, {
                        #     'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                        #     'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': MCE_est_alt,
                        #     'Temp': temp, 'Type': 'Plug-In (alt)'
                        # }, {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': QMCE_est_split,
                            'Temp': temp, 'Type': 'Direct (split)'
                        }, {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': MCE_est_full,
                            'Temp': temp, 'Type': 'Plug-In (full)'
                        }, {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': QMCE_est_full,
                            'Temp': temp, 'Type': 'Direct (full)'
                        }, {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': ground_truth,
                            'Temp': temp, 'Type': 'Ground Truth'
                        },
                    ]

    return pd.DataFrame(results_df)

def inner_prod_temp_simulation(seeds, data_sizes, alpha_classes, reg_const, temps, kernel_X=pairwise.linear_kernel):

    results_df = []
    for alpha_class in alpha_classes:
        for temp in tqdm(temps):
            dir_alpha, n_classes = alpha_class
            # monte carlo simulate ground truth
            samples = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=10000)
            ground_truth = ground_truth_sim(samples, temp=temp)
            for seed in seeds:
                for data_size in data_sizes:
                    _, QMCE_est_split, _, QMCE_est_full = inner_prod_estimator_simulation_iter(
                        seed, data_size, alpha_class, reg_const, temp, kernel_X=kernel_X
                    )
                    # _, QMCE_est_split_rbf, _, QMCE_est_full_rbf = inner_prod_estimator_simulation_iter(
                    #     seed, data_size, alpha_class, reg_const, temp, kernel_X=pairwise.rbf_kernel
                    # )
                    results_df += [
                        {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': QMCE_est_split,
                            'Temp': temp, 'Type': 'Direct (split)'
                        }, {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': QMCE_est_full,
                            'Temp': temp, 'Type': 'Direct (full)'
                        }, {
                        #     'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                        #     'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': QMCE_est_split_rbf,
                        #     'Temp': temp, 'Type': 'Direct (split; rbf)'
                        # }, {
                        #     'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                        #     'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': QMCE_est_full_rbf,
                        #     'Temp': temp, 'Type': 'Direct (full; rbf)'
                        # }, {
                            'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                            'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': ground_truth,
                            'Temp': temp, 'Type': 'Ground Truth'
                        },
                    ]
    return pd.DataFrame(results_df)

def inner_prod_CV_simulation(
    seeds, data_sizes, alpha_classes, reg_consts, temps, k_folds=5,
    kernel_X=lambda x,y=None: pairwise.rbf_kernel(x, y, gamma=1), kernel_Y=delta_kernel
):

    results_df = []
    for alpha_class in alpha_classes:
        for temp in temps:
            dir_alpha, n_classes = alpha_class
            # monte carlo simulate ground truth
            samples = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=10000)
            ground_truth = ground_truth_sim(samples, temp=temp)
            for seed in tqdm(seeds):
                for data_size in data_sizes:
                    dir_alpha, n_classes = alpha_class
                    np.random.seed(seed)
                    Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=data_size)
                    # optional temperature scaling
                    Ys = np.array([np.random.choice(n_classes, p=X**temp/np.sum(X**temp)) for X in Xs])
                    K_XX = kernel_X(Xs)
                    K_YY = kernel_Y(Ys)
                    kf = KFold(n_splits=k_folds)
                    est_value = []
                    loss_val = []
                    for train_ind, val_ind in kf.split(Xs):
                        K_XX_train = K_XX[train_ind, :][:, train_ind]
                        K_YY_train = K_YY[train_ind, :][:, train_ind]
                        K_XX_val = K_XX[train_ind, :][:, val_ind] # not a bug
                        K_YY_val = K_YY[val_ind, :][:, val_ind]
                        result = QMCE_fold_fit(K_XX_train, K_YY_train, K_XX_val, K_YY_val, reg_consts)
                        est_value.append(result[:, 0])
                        loss_val.append(result[:, 1])
                    est_value = np.array(est_value).mean(axis=0)
                    loss_val = np.array(loss_val).mean(axis=0)
                    # due to numerical instabilities, the loss can be negative
                    loss_val_cleaned = [value if value>0 else np.max(loss_val) for value in loss_val]
                    for reg_id, reg_const in enumerate(reg_consts):
                        results_df += [
                            {
                                'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                                'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': est_value[reg_id],
                                'Temp': temp, 'Type': 'KRR Est', 'Loss': loss_val[reg_id]
                            }, {
                                'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                                'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': ground_truth,
                                'Temp': temp, 'Type': 'Ground Truth', 'Loss': 0
                            },
                        ]
    return pd.DataFrame(results_df)

def inner_prod_CV_simulation_alt(
    seeds, data_sizes, alpha_classes, reg_consts, temps, k_folds=5,
    kernel_X=lambda x,y=None: pairwise.rbf_kernel(x, y, gamma=1), kernel_Y=delta_kernel
):

    results_df = []
    for alpha_class in alpha_classes:
        for temp in temps:
            dir_alpha, n_classes = alpha_class
            # monte carlo simulate ground truth
            samples = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=10000)
            ground_truth = ground_truth_sim(samples, temp=temp)
            for seed in tqdm(seeds):
                for data_size in data_sizes:
                    dir_alpha, n_classes = alpha_class
                    np.random.seed(seed)
                    Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=data_size)
                    # optional temperature scaling
                    Ys = np.array([np.random.choice(n_classes, p=X**temp/np.sum(X**temp)) for X in Xs])
                    K_XX = kernel_X(Xs)
                    K_YY = kernel_Y(Ys)
                    kf = KFold(n_splits=k_folds)
                    est_value = []
                    loss_val = []
                    for train_ind, val_ind in kf.split(Xs):
                        K_XX_train = K_XX[train_ind, :][:, train_ind]
                        K_YY_train = K_YY[train_ind, :][:, train_ind]
                        K_XX_val = K_XX[train_ind, :][:, val_ind] # not a bug
                        K_YY_val = K_YY[val_ind, :][:, val_ind]
                        result = slow_CV_QMCE_fit(K_XX_train, K_YY_train, K_XX_val, K_YY_val, reg_consts)
                        est_value.append(result[:, 0])
                        loss_val.append(result[:, 1])
                    est_value = np.array(est_value).mean(axis=0)
                    loss_val = np.array(loss_val).mean(axis=0)
                    # due to numerical instabilities, the loss can be negative
                    loss_val_cleaned = [value if value>0 else np.max(loss_val) for value in loss_val]
                    for reg_id, reg_const in enumerate(reg_consts):
                        results_df += [
                            {
                                'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                                'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': est_value[reg_id],
                                'Temp': temp, 'Type': 'KRR Est', 'Loss': loss_val[reg_id]
                            }, {
                                'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                                'dir_alpha': dir_alpha, 'reg_const': reg_const, 'Value': ground_truth,
                                'Temp': temp, 'Type': 'Ground Truth', 'Loss': 0
                            },
                        ]
    return pd.DataFrame(results_df)

def inner_prod_estimator_kde_simulation_iter(seed, data_size, alpha_class, temp, include_kde, include_quad, do_CV=False):
    results = {}
    dir_alpha, n_classes = alpha_class
    n_UV = n_XY = int(data_size/2)
    np.random.seed(seed)
    Xs = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=n_XY)
    Us = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=n_UV)
    # optional temperature scaling
    Ys = np.array([np.random.choice(n_classes, p=X**temp/np.sum(X**temp)) for X in Xs])
    Vs = np.array([np.random.choice(n_classes, p=U**temp/np.sum(U**temp)) for U in Us])
    XUs = np.concatenate([Xs, Us])
    YVs = np.concatenate([Ys, Vs])
    k_rbf = lambda x, y=None: pairwise.rbf_kernel(x, y, gamma=1.)
    if do_CV:
        reg_consts = [10.**i/(n_XY**.5) for i in np.arange(-10, 0, 0.5)]
        CV_results = inner_prod_QMCE_CV(YVs, XUs, kernel_X=k_rbf, reg_consts=reg_consts)
        results['QMCE_est_CV_rbf'] = CV_results[0]
        results['QMCE_RMSE_CV_rbf'] = CV_results[1]**.5
    else:
        results['QMCE_est_asymp'] = inner_prod_QMCE_asymp(YVs, XUs)
        results['QMCE_est_asymp_rbf'] = inner_prod_QMCE_asymp(YVs, XUs, kernel_X=pairwise.rbf_kernel)
    if include_quad:
        results['QMCE_est_asymp_quad'] = inner_prod_QMCE_asymp(YVs, XUs, reg_rate=1)
        results['QMCE_est_asymp_rbf_quad'] = inner_prod_QMCE_asymp(YVs, XUs, kernel_X=pairwise.rbf_kernel, reg_rate=1)
    if include_kde:
        XUs_torch = torch.from_numpy(XUs)
        YVs_torch = torch.from_numpy(YVs)
        bw = get_bandwidth(XUs_torch, device='cpu')
        results['KDE_est'] = kde_ratio_estimator(XUs_torch, YVs_torch, bandwidth=bw)

    return results

def inner_prod_estimator_kde_simulation(seeds, data_sizes, alpha_classes, temp=1, include_kde=False, include_quad=False, do_CV=False):

    results_df = []
    for alpha_class in alpha_classes:
        dir_alpha, n_classes = alpha_class
        # monte carlo simulate ground truth
        samples = np.random.dirichlet(alpha=np.repeat(dir_alpha, n_classes), size=10000)
        ground_truth = ground_truth_sim(samples, temp=temp)

        for seed in tqdm(seeds):
            for data_size in data_sizes:
                results = inner_prod_estimator_kde_simulation_iter(
                    seed, data_size, alpha_class, temp, include_kde, include_quad, do_CV
                )
                results_df += [{
                        'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                        'dir_alpha': dir_alpha, 'Value': value,
                        'Temp': temp, 'Type': key
                    } for key, value in results.items()
                ]
                results_df += [
                    {
                        'seed': seed, 'data_size': data_size, 'n_class': n_classes,
                        'dir_alpha': dir_alpha, 'Value': ground_truth,
                        'Temp': temp, 'Type': 'Ground Truth'
                    },
                ]

    return pd.DataFrame(results_df)
