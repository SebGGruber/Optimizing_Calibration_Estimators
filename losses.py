import numpy as np

from scipy.special import softmax
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score, pairwise
from sklearn.preprocessing import label_binarize

def ce_losses(Ps, Ys, cal_preds):
    """
    Ps: (n, c) np array
    Ys: (n,) np array
    cal_preds: (n, n) np array
    returns: (n, n) np array
    """
    # for binarized cases, like top-label or classwise calibration
    if len(Ps.shape) < 2:
        diffs_outer = np.outer(Ps-Ys, Ps-Ys)
    # for multiclass case, i.e. canonical calibration
    else:
        n_classes = Ps.shape[1]
        assert n_classes > 1
        one_hot_Ys = np.zeros((Ys.shape[0], n_classes))
        one_hot_Ys[np.arange(Ys.shape[0]), Ys] = 1
        diffs_outer = pairwise.linear_kernel(Ps-one_hot_Ys)
    loss_values = (
        diffs_outer - cal_preds
    )**2
    return loss_values

def ce_risk(Ps, Ys, cal_preds, ignore_nan=False):
    """
    Ps: (n, c) np array
    Ys: (n,) np array
    cal_preds: (n, n) np array
    returns: scalar
    """
    loss_values = ce_losses(Ps, Ys, cal_preds)
    n_size = loss_values.shape[0]
    n_size = (n_size**2 - n_size)
    if ignore_nan:
        n_nans = np.isnan(loss_values).sum() - np.isnan(np.diag(loss_values)).sum()
        n_size -= n_nans
        emp_risk = np.nansum(loss_values) - np.nansum(np.diag(loss_values))
    else:
        emp_risk = loss_values.sum() - np.diag(loss_values).sum()
    return emp_risk / n_size

def BS(logits, labels, logits_are_probs=False):
    if not logits_are_probs:
        p = softmax(logits, axis=1)
    else:
        p = logits
    y = label_binarize(np.array(labels), classes=range(logits.shape[1]))
    return np.average(np.sum((p - y)**2, axis=1))

def NLL(logits, labels, logits_are_probs=False):
    if not logits_are_probs:
        p = softmax(logits, axis=1)
    else:
        p = logits
    return log_loss(labels, p)