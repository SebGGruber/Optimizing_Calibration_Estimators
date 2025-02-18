# Source: ChatGPT
import numpy as np

def ECE_bin_fit(Ps, Ys, n_bins):
    probs = np.asarray(Ps)
    labels = np.asarray(Ys)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    # Digitize the predicted probabilities into bins
    bin_indices = np.digitize(probs, bin_boundaries, right=True) - 1

    # Initialize arrays to accumulate bin values
    bin_conf = np.repeat(np.mean(probs), n_bins)
    bin_acc = np.repeat(np.mean(labels), n_bins)

    # Accumulate values for each bin
    for i in range(n_bins):
        in_bin = bin_indices == i
        bin_counts = np.sum(in_bin)
        if bin_counts > 0:
            bin_sums = np.sum(probs[in_bin])
            bin_true = np.sum(labels[in_bin])
            bin_conf[i] = bin_sums / bin_counts
            bin_acc[i] = bin_true / bin_counts

    def cal_model(new_Ps):
        bin_indices = np.digitize(new_Ps, bin_boundaries, right=True) - 1
        diff_preds = np.zeros(new_Ps.shape[0])
        diff_preds = bin_conf[bin_indices] - bin_acc[bin_indices]
        return np.outer(diff_preds, diff_preds)
        
    return cal_model

# 15 bins is used in torchmetrics and Guo et al 2017
def ECE_bin(Ps, Ys, n_bins=15):
    probs = np.asarray(Ps)
    labels = np.asarray(Ys)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    # Digitize the predicted probabilities into bins
    bin_indices = np.digitize(probs, bin_boundaries, right=True) - 1
    ece_est = 0

    # Accumulate values for each bin
    for i in range(n_bins):
        in_bin = bin_indices == i
        bin_prob = np.mean(in_bin)
        if bin_prob > 0:
            bin_conf = np.mean(probs[in_bin])
            bin_acc = np.mean(labels[in_bin])
            ece_est += bin_prob*(bin_conf-bin_acc)**2

    return ece_est

# 15 bins is used in torchmetrics and Guo et al 2017
def CWCE_bin(Ps, Ys, n_bins=15):
    probs = np.asarray(Ps)
    labels = np.asarray(Ys)

    n_classes = Ps.shape[1]
    one_hot_Ys = np.zeros((Ys.shape[0], n_classes))
    one_hot_Ys[np.arange(Ys.shape[0]), Ys] = 1
    CWCE_est = 0
    for k in range(n_classes):
        CWCE_est += ECE_bin(Ps[:, k], one_hot_Ys[:, k], n_bins=n_bins)

    return CWCE_est