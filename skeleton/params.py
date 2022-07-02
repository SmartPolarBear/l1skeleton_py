import numpy as np
import time

from typing import Final

SMALL_THRESHOLD: Final[float] = 1e-20


def get_density_weights(points, hd, for_center=False, center=None):
    """
    INPUTS:
        x: 1x3 center we of interest, np.ndarray
        points: Nx3 array of all the points, np.ndarray
        h: size of local neighboorhood, float
    RETURNS:
        - np.array Nx1 of density weights assoiscated to each point
    """
    if center is None:
        center = [0, 0, 0]

    density_weights = []

    if for_center:
        r = points - center
        r2 = np.einsum('ij,ij->i', r, r)
        density_weights = 1 + np.einsum('i->', np.exp((-r2) / ((hd / 2) ** 2)))
    else:
        for point in points:
            r = point - points
            r2 = np.einsum('ij,ij->i', r, r)
            r2 = r2[r2 > SMALL_THRESHOLD]

            density_weight = 1 + np.einsum('i->', np.exp((-r2) / ((hd / 2.0) ** 2)))
            density_weights.append(density_weight)

    return np.array(density_weights)


def get_term1(center: np.ndarray, points: np.ndarray, h: float, density_weights: np.ndarray):
    """
    :param center: 1x3 center we of interest, np.ndarray
    :param points: Nx3 array of all the points, np.ndarray
    :param h: size of local neighborhood, float
    :param density_weights:
    :return: term1 of the equation as float
    """

    r = points - center
    r2 = np.einsum('ij,ij->i', r, r)

    thetas = np.exp(-r2 / ((h / 2) ** 2))

    r2[r2 <= SMALL_THRESHOLD] = 1
    alphas = thetas / np.sqrt(r2)
    alphas /= density_weights

    denom = np.einsum('i->', alphas)
    if denom > SMALL_THRESHOLD:
        # term1 = np.sum((points.T*alphas).T, axis = 0)/denom
        term1 = np.einsum('j,jk->k', alphas, points) / denom
    else:
        term1 = np.array(False)

    return term1


def get_term2(center: np.ndarray, centers: np.ndarray, h: float):
    """
    :param center:  1x3 center we of interest, np.ndarray
    :param centers:  Nx3 array of all the centers (excluding the current center), np.ndarray
    :param h:  size of local neighborhood, float
    :return:  term2 of the equation as float
    """

    x = center - centers

    r2 = np.einsum('ij,ij->i', x, x)

    indexes = r2 > SMALL_THRESHOLD
    r2 = r2[indexes]
    x = x[indexes]

    thetas = np.exp((-r2) / ((h / 2) ** 2))

    betas = thetas / r2

    denom = np.einsum('i->', betas)

    if denom > SMALL_THRESHOLD:
        num = np.einsum('j,jk->k', betas, x)
        term2 = num / denom
    else:
        term2 = np.array(False)

    return term2


def get_sigma(center, centers, local_sigmas, h, k=5):
    # These are the weights
    r = centers - center
    r2 = np.einsum('ij,ij->i', r, r)

    indexes = r2 > SMALL_THRESHOLD
    r = r[indexes]
    r2 = r2[indexes]

    thetas = np.exp((-r2) / ((h / 2) ** 2))

    cov = np.einsum('j,jk,jl->kl', thetas, r, r)

    # Get eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov)

    if np.iscomplex(values).any():
        values = np.real(values)

        vectors = np.real(vectors)
        vectors_norm = np.sqrt(np.einsum('ij,ij->i', vectors, vectors))
        vectors = vectors / vectors_norm

    # argsort always works from low --> to high so taking the negative values will give us high --> low indices
    sorted_indices = np.argsort(-values)

    sigma = np.max(values) / np.sum(values)

    knn = np.argsort(r2)
    knn_sigmas = np.sum(local_sigmas[knn[:(k - 1)]])
    sigma = (sigma + knn_sigmas) / k  # smoothing sigma

    vectors_sorted = vectors[:, sorted_indices]

    return sigma, vectors_sorted
