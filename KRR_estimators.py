import numpy as np
import torch
import itertools

from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial.distance import jensenshannon


def delta_kernel(X, Y=None):
    if Y is None:
        Y = X
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    combinations = itertools.product(X, Y)
    left, right = zip(*combinations)
    gram_m = 1.*np.equal(left, right).reshape(n_X, n_Y)
    return gram_m

def expected_kernel(X, Y=None, kernel_Y=pairwise.rbf_kernel):
    """Compute the block kernel between X and Y.

    Parameters
    ----------
    X : ndarray of shape (n_outer, n_inner, n_features)

    Y : ndarray of shape (n_outer, n_inner, n_features)

    Returns
    -------
    kernel_block__matrix : ndarray of shape (n_outer * n_inner, n_outer * n_inner)
    """
    if Y is None:
        Y = X
    assert Y.shape == X.shape
    n_outer, n_inner, n_feat = X.shape
    # reshapes with index orders [11, 21, ..., n1, 12, 22, ...]
    # where n = n_inner
    X_flat = X.reshape(n_outer*n_inner, n_feat)
    Y_flat = Y.reshape(n_outer*n_inner, n_feat)
    kXY = kernel(X_flat, Y_flat)
    gram_m = np.array([
        [kXY[n_inner*i:n_inner*(i+1), n_inner*i:n_inner*(j+1)].mean() for i in range(n_outer)]
        for j in range(n_outer)
    ])
    return gram_m

def JS_kernel(X, Y=None, gamma=1):
    if Y is None:
        Y = X
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    combinations = itertools.product(X, Y)
    left, right = zip(*combinations)
    gram_m = np.exp(-gamma*jensenshannon(left, right, axis=1)**2).reshape(n_X, n_Y)
    return gram_m

def inner_prod_MCE_split(
    Y, X, V=None, U=None, X_prime=None,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    reg_const=1
):

    assert Y.shape[0]==X.shape[0]
    assert V.shape[0]==U.shape[0]
    n_XY = Y.shape[0]
    n_UV = V.shape[0]
    XU = np.concatenate([X, U])
    K_XUXU = kernel_X(XU)
    K_XX = K_XUXU[:n_XY, :n_XY]
    K_UU = K_XUXU[n_XY:, n_XY:]
    if X_prime is None:
        K_XUX = K_XUXU[:, :n_XY]
        K_XUU = K_XUXU[:, n_XY:]
    else:
        K_XUX = kernel_X(X_prime, X)
        K_XUU = kernel_X(X_prime, U)
    K_YV = kernel_Y(Y, V)
    W_X = np.linalg.inv(K_XX + n_XY * reg_const * np.identity(n_XY))
    W_U = np.linalg.inv(K_UU + n_UV * reg_const * np.identity(n_UV))

    return np.trace(K_XUX @ W_X @ K_YV @ W_U.T @ K_XUU.T) / K_XUX.shape[0]

def inner_prod_MCE_full(
    Y, X,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    reg_const=1
):
    assert Y.shape[0]==X.shape[0]
    n_XY = Y.shape[0]
    K_XX = kernel_X(X)
    K_YY = kernel_Y(Y)
    W_X = np.linalg.inv(K_XX + n_XY * reg_const * np.identity(n_XY))

    return np.trace(K_XX @ W_X @ K_YY @ W_X.T @ K_XX.T) / n_XY

def inner_prod_MCE_alt(
    Y, X,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    reg_const=1
):
    assert Y.shape[0]==X.shape[0]
    n_XY = Y.shape[0]
    K_XX = kernel_X(X)
    K_YY = kernel_Y(Y)
    W_X = np.linalg.inv(K_XX + n_XY * reg_const * np.identity(n_XY))

    return np.trace(K_XX @ W_X @ K_YY) / n_XY

def inner_prod_QMCE_split(
    Y, X, V=None, U=None, X_prime=None,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    reg_const=1
):

    assert Y.shape[0]==X.shape[0]
    assert V.shape[0]==U.shape[0]
    n_XY = Y.shape[0]
    n_UV = V.shape[0]
    XU = np.concatenate([X, U])
    K_XUXU = kernel_X(XU)
    K_XX = K_XUXU[:n_XY, :n_XY]
    K_UU = K_XUXU[n_XY:, n_XY:]
    if X_prime is None:
        K_XUX = K_XUXU[:, :n_XY]
        K_XUU = K_XUXU[:, n_XY:]
    else:
        K_XUX = kernel_X(X_prime, X)
        K_XUU = kernel_X(X_prime, U)
    K_YV = kernel_Y(Y, V)
    Lambda_X, Q_X = np.linalg.eigh(K_XX)
    Lambda_U, Q_U = np.linalg.eigh(K_UU)
    inv_Lambda_XU_reg = np.array([
        [1/(l_x*l_v + n_XY*n_UV*reg_const) for l_x in np.maximum(0, Lambda_X)]
        for l_v in np.maximum(0, Lambda_U)
    ])
    avg_K_XU = K_XUX.T @ K_XUU / K_XUX.shape[0]
    
    return np.trace(Q_X.T @ avg_K_XU @ Q_U @ np.multiply(inv_Lambda_XU_reg, Q_U.T @ K_YV.T @ Q_X))

def inner_prod_QMCE_full(
    Y, X,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    reg_const=1
):

    assert Y.shape[0]==X.shape[0]
    n_XY = Y.shape[0]
    K_XX = kernel_X(X)
    K_YY = kernel_Y(Y)
    Lambda_X, Q_X = np.linalg.eigh(K_XX)
    inv_Lambda_XX_reg = np.array([
        [1/(l_x*l_x + n_XY*n_XY*reg_const) for l_x in np.maximum(0, Lambda_X)]
        for l_x in np.maximum(0, Lambda_X)
    ])
    avg_K_XX = K_XX.T @ K_XX / n_XY
    
    return np.trace(Q_X.T @ avg_K_XX @ Q_X @ np.multiply(inv_Lambda_XX_reg, Q_X.T @ K_YY.T @ Q_X))

def inner_prod_QMCE_asymp(
    Y, X,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    reg_rate=.5
):
    n_XY = Y.shape[0]
    reg_const = 10.**(-14)/(n_XY**reg_rate)
    QMCE_est = inner_prod_QMCE_full(
        Y, X,
        kernel_X=kernel_X,
        kernel_Y=kernel_Y,
        reg_const=reg_const
    )
    return QMCE_est

def _QMCE_fold_fit(K_XX_train, K_YY_train, K_XX_val, K_YY_val, reg_consts):
    """."""
    n_train = K_XX_train.shape[0]
    n_val = K_XX_val.shape[1]
    # compute terms which do not need `reg_const`
    Lambda_X, Q_X = np.linalg.eigh(K_XX_train)
    sq_K_XX_val = K_XX_val @ K_XX_val.T
    Q_XXXX_val = Q_X.T @ sq_K_XX_val @ Q_X
    Q_XYYX_train = Q_X.T @ K_YY_train.T @ Q_X
    Q_XYYX_val = Q_X.T @ K_XX_val @ K_YY_val @ K_XX_val.T @ Q_X
    loss_y_term = K_YY_val.mean()

    # compute terms which need `reg_const`
    #def QMCE_fit(reg_const, Q_X=Q_X, K_XX_train=K_XX_train, K_XX_val=K_XX_val, K_YY_train=K_YY_train, sq_K_XX_val=sq_K_XX_val, K_YY_val=K_YY_val):
    def _QMCE_fit(reg_const):
        # outer product of eigen values
        inv_Lambda_XX_reg = np.outer(np.maximum(0, Lambda_X), np.maximum(0, Lambda_X))
        # add reg constant
        inv_Lambda_XX_reg += reg_const*n_train**2
        # "inverse"
        inv_Lambda_XX_reg = 1./inv_Lambda_XX_reg
        hadamard_Lambda_XYYX = np.multiply(inv_Lambda_XX_reg, Q_XYYX_train)
        loss_xy_term = np.trace(Q_XYYX_val @ hadamard_Lambda_XYYX) / n_val**2
        loss_x_term = np.trace(
            hadamard_Lambda_XYYX.T @ Q_XXXX_val @ hadamard_Lambda_XYYX @ Q_XXXX_val.T
        ) / n_val**2
        QMCE_est = np.trace(Q_XXXX_val @ hadamard_Lambda_XYYX) / n_val
        return QMCE_est, (loss_y_term - 2*loss_xy_term + loss_x_term)

    results_est_loss = [_QMCE_fit(reg_const) for reg_const in reg_consts]
    return np.array(results_est_loss)

def QMCE_fold_fit(K_XX_train, K_YY_train, K_XX_val, K_YY_val, reg_consts):
    """."""
    n_train = K_XX_train.shape[0]
    n_val = K_XX_val.shape[1]
    # compute terms which do not need `reg_const`
    Lambda_X, Q_X = np.linalg.eigh(K_XX_train)
    Q_XYYX_train = Q_X.T @ K_YY_train @ Q_X

    # compute terms which need `reg_const`
    def QMCE_fit(reg_const):
        # outer product of eigen values
        inv_Lambda_XX_reg = np.outer(np.maximum(0, Lambda_X), np.maximum(0, Lambda_X))
        # add reg constant
        inv_Lambda_XX_reg += reg_const*n_train**2
        # "inverse"
        inv_Lambda_XX_reg = 1./inv_Lambda_XX_reg
        hadamard_Lambda_XYYX = np.multiply(inv_Lambda_XX_reg, Q_XYYX_train)
        H_krr = K_XX_val.T @ Q_X @ hadamard_Lambda_XYYX @ Q_X.T @ K_XX_val
        losses_val = (K_YY_val - H_krr)**2
        QMCE_est = np.diag(H_krr).mean()
        avg_loss_val = (losses_val.sum() - np.diag(losses_val).sum())/(n_val**2 - n_val)
        return QMCE_est, avg_loss_val

    results_est_loss = [QMCE_fit(reg_const) for reg_const in reg_consts]
    return np.array(results_est_loss)

def ECE_krr_fit(
    K_XX_train_train, K_YY_train_train
):
    """."""
    n_train = K_XX_train_train.shape[0]
    assert n_train == K_YY_train_train.shape[0]
    # compute terms which do not need `reg_const`
    Lambda_X, Q_X = np.linalg.eigh(K_XX_train_train)
    Q_XYYX_train = Q_X.T @ K_YY_train_train @ Q_X

    # compute terms which need `reg_const`
    def cal_model(K_XX_train_eval, reg_const):
        n_val = K_XX_train_eval.shape[1]
        # outer product of eigen values
        inv_Lambda_XX_reg = np.outer(np.maximum(0, Lambda_X), np.maximum(0, Lambda_X))
        # add reg constant divided by sqrt{n}
        inv_Lambda_XX_reg += reg_const*n_train**1.5
        # "inverse"
        inv_Lambda_XX_reg = 1./inv_Lambda_XX_reg
        hadamard_Lambda_XYYX = np.multiply(inv_Lambda_XX_reg, Q_XYYX_train)
        H_krr = K_XX_train_eval.T @ Q_X @ hadamard_Lambda_XYYX @ Q_X.T @ K_XX_train_eval
        return H_krr

    return cal_model

def ECE_kkrr_fit(
    K_XX_train_train, K_YY_train_train
):
    """."""
    n_train = K_XX_train_train.shape[0]
    assert n_train == K_YY_train_train.shape[0]
    
    # compute terms which do not need `reg_const`
    Lambda_X, Q_X = np.linalg.eigh(K_XX_train_train)
    Lambda_X = np.maximum(0, Lambda_X)

    def cal_model(K_XX_train_eval, reg_const):
        n_val = K_XX_train_eval.shape[1]
        W_XX = Q_X @ np.diag(1.0/(Lambda_X + reg_const*n_train**0.75)) @ Q_X.T
        H_krr = K_XX_train_eval.T @ W_XX @ K_YY_train_train @ W_XX.T @ K_XX_train_eval
        return H_krr

    return cal_model

def inner_prod_QMCE_CV(
    Y, X,
    reg_consts,
    kernel_X=pairwise.linear_kernel,
    kernel_Y=delta_kernel,
    k_folds=5
):
    assert Y.shape[0]==X.shape[0]
    K_XX = kernel_X(X)
    K_YY = kernel_Y(Y)
    kf = KFold(n_splits=k_folds)
    est_value = []
    loss_val = []
    for train_ind, val_ind in kf.split(X):
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
    argmin = np.argmin(loss_val_cleaned)
    print(f'Picked lambda #{argmin} in range(0, {len(reg_consts)})')
    return est_value[argmin], loss_val_cleaned[argmin]

def slow_inner_prod_QMCE(Y, X, V=None, U=None, kernel_X=pairwise.linear_kernel, kernel_Y=delta_kernel, reg_const=1):
    # for debugging / unit tests (scales O(n^6))

    assert Y.shape[0]==X.shape[0]
    assert V.shape[0]==U.shape[0]
    n_XY = Y.shape[0]
    n_UV = V.shape[0]
    XU = np.concatenate([X, U])
    K_XUXU = kernel_X(XU)
    K_XX = K_XVXV[:n_YX, :n_YX]
    K_UU = K_XVXV[n_YX:, n_YX:]
    K_XUX = K_XUXU[:, :n_YX]
    K_XUU = K_XUXU[:, n_YX:]
    K_YV = kernel_Y(Y, V)
    W_XUXU = np.linalg.inv(np.kron(K_XX, K_UU) + reg_const*n_XY*n_UV*np.eye(n_XY*n_UV))
    avg_K_XU = K_XUX.T @ K_XUU / (n_XY + n_UV)
    return (avg_K_XU.reshape(1, -1) @ W_XUXU @ K_YV.reshape(1, -1).T)[0,0]

def slow_slow_CV_QMCE(K_XX_train, K_YY_train, K_XX_val, K_YY_val, reg_const):
    # for debugging / unit tests (scales O(n^6))
    n_train = K_YY_train.shape[0]
    n_val = K_YY_val.shape[0]
    W_XX = np.linalg.inv(np.kron(K_XX_train, K_XX_train) + reg_const*n_train**2*np.eye(n_train**2))

    def loss_term(i,j):
        y_val = K_YY_val[i,j]
        pred_val = np.outer(K_XX_val[:,i], K_XX_val[:,j]).reshape(1, -1) @ W_XX @ K_YY_train.reshape(1, -1).T
        return (y_val - pred_val[0,0])**2

    val_risk = np.mean([
        loss_term(i,j)
        for i in range(n_val) for j in range(n_val)
    ])
    return val_risk

def slow_CV_QMCE_fit(K_XX_train, K_YY_train, K_XX_val, K_YY_val, reg_consts):
    # for debugging / unit tests (scales O(n^5))
    n_train = K_XX_train.shape[0]
    n_val = K_XX_val.shape[1]
    # compute terms which do not need `reg_const`
    Lambda_X, Q_X = np.linalg.eigh(K_XX_train)
    sq_K_XX_val = K_XX_val @ K_XX_val.T
    Q_XXXX_val = Q_X.T @ sq_K_XX_val @ Q_X
    Q_XYYX_train = Q_X.T @ K_YY_train.T @ Q_X
    Q_XYYX_val = Q_X.T @ K_XX_val @ K_YY_val @ K_XX_val.T @ Q_X

    # compute terms which need `reg_const`
    def QMCE_fit(reg_const):
        # outer product of eigen values
        inv_Lambda_XX_reg = np.outer(np.maximum(0, Lambda_X), np.maximum(0, Lambda_X))
        # add reg constant
        inv_Lambda_XX_reg += reg_const*n_train**2
        # "inverse"
        inv_Lambda_XX_reg = 1./inv_Lambda_XX_reg
        hadamard_Lambda_XYYX = np.multiply(inv_Lambda_XX_reg, Q_XYYX_train)
        QMCE_est = np.trace(Q_XXXX_val @ hadamard_Lambda_XYYX) / n_val
        def loss_term(i,j):
            y_val = K_YY_val[i,j]
            Q_XXXX_pred = Q_X.T @ np.outer(K_XX_val[:,i], K_XX_val[:,j]) @ Q_X
            pred_val = np.trace(Q_XXXX_pred @ hadamard_Lambda_XYYX)
            return (y_val - pred_val)**2

        val_risk = np.mean([
            loss_term(i,j)
            for i in range(n_val) for j in range(n_val) if i!=j
        ])
        return QMCE_est, val_risk

    results_est_loss = [QMCE_fit(reg_const) for reg_const in reg_consts]
    return np.array(results_est_loss)
