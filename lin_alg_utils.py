import math
import numpy as np

from sklearn.metrics.pairwise import linear_kernel

def compute_mat_W_X(X, X_prime=None, reg_const=1, kernel=linear_kernel):
    """
    Computes the matrix :math:`W`, as specified in the paper.

    Matrix :math:`W` is defined as follows :

    .. math::
        W := (K_Z + n \lambda I_n)^{-1}

    Parameters
    ----------
    X : array_like
        The kernel Gram matrix for the :math:`Z` variable :math:`K_Z`.
    reg_const : float
        The regularisation constant.

    Returns
    -------
    array_like
        The matrix :math:`W` as defined in the paper.
    """

    mat_K_X = kernel(X)
    if X_prime is not None:
        mat_K_X_prime = kernel(X_prime)
        mat_K_X = np.kron(mat_K_X, mat_K_X_prime)
        
    n = mat_K_X.shape[0]
    mat_W = np.linalg.inv(
        mat_K_X + n * reg_const * np.identity(n)
    )

    return mat_W

def compute_vec_k_X_in_x(x, X, X_prime=None, kernel=linear_kernel):
    """
    Evaluates the function :math:`\mathbf{k}_X(x)`, as defined in the paper.

    Function :math:`\mathbf{k}_X(x)` is defined on :math:`\mathcal{X}` by

    .. math::
        \mathbf{k}_X(x) = (k_X(X_1, x), \dots , k_Z(X_n, x))^T

    Parameters
    ----------
    x : array_like
        The evaluation point :math:`x`.
    X : array_like
        The observations (data) in :math:`\mathcal{X}` domain.
    kernel : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}` .

    Returns
    -------
    array_like
        The vector valued function evaluated at :math:`x`, :math:`k_X(x)`.
    """
    mat_K_X = kernel(X, [x])
    if X_prime is not None:
        mat_K_X_prime = kernel(X_prime, [x])
        mat_K_X = np.kron(mat_K_X, mat_K_X_prime)

    return mat_K_X