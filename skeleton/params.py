import numpy as np
import time

EXTREME_SMALL = 10 ** -323
VEARY_SMALL = 10 ** -10


def get_thetas(r, h):
    """
    :param r: variable r
    :param h: local neighborhodd size
    :return: theta(r)
    """
    thetas = np.exp((-r ** 2) / ((h / 2) ** 2))
    # Clip to JUST not zero
    thetas = np.clip(thetas, EXTREME_SMALL, None)
    return thetas


def get_alphas(x: np.ndarray, points: np.ndarray, h: float):
    """
    :param x:  1x3 center we of interest, np.ndarray
    :param points:  Nx3 array of all the points, np.ndarray
    :param h: size of local neighborhood, float
    :return: alpha(i,j)
    """
    r = np.linalg.norm(x - points, axis=1) + VEARY_SMALL
    theta = get_thetas(r, h)

    alphas = theta / r
    return alphas


def get_betas(x, points, h):
    """
    :param x:  1x3 center we of interest, np.ndarray
    :param points:  Nx3 array of all the points, np.ndarray
    :param h: size of local neighborhood, float
    :return: beta(i,i')
    """
    r = np.linalg.norm(x - points, axis=1) + VEARY_SMALL
    theta = get_thetas(r, h)

    betas = theta / r ** 2

    return np.array(betas)


def get_density_weights(points, h0, for_center=False, center=None):
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
        density_weights = np.einsum('i->', np.exp((-r2) / ((h0 / 4) ** 2)))
    else:

        for point in points:
            r = point - points
            r2 = np.einsum('ij,ij->i', r, r)
            # This calculation includes the point itself thus one entry will be zero resultig in the needed + 1 in
            # formula dj = 1+ sum(theta(p_i - p_j))
            density_weight = np.einsum('i->', np.exp((-r2) / ((h0 / 4) ** 2)))
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

    t1_t = time.perf_counter()

    r = points - center
    r2 = np.einsum('ij,ij->i', r, r)

    thetas = np.exp(-r2 / ((h / 2) ** 2))
    # Clip to JUST not zero
    # thetas =  np.clip(thetas, 10**-323, None)

    # DIFFERS FROM PAPER
    # r_norm = np.sqrt(r_norm, axis = 1)
    # alphas = thetas/r_norm

    alphas = thetas / density_weights

    denom = np.einsum('i->', alphas)
    if denom > 10 ** -20:
        # term1 = np.sum((points.T*alphas).T, axis = 0)/denom
        term1 = np.einsum('j,jk->k', alphas, points) / denom
    else:
        term1 = np.array(False)

    t2_t = time.perf_counter()
    tt = round(t2_t - t1_t, 5)

    return term1, tt


def get_term2(center: np.ndarray, centers: np.ndarray, h: float):
    """
    :param center:  1x3 center we of interest, np.ndarray
    :param centers:  Nx3 array of all the centers (excluding the current center), np.ndarray
    :param h:  size of local neighborhood, float
    :return:  term2 of the equation as float
    """

    t1 = time.perf_counter()

    x = center - centers
    r2 = np.einsum('ij,ij->i', x, x)
    r = 1 / np.sqrt(r2)
    # r3 = np.sum(r**1.2, axis = 1)
    thetas = np.exp((-r2) / ((h / 2) ** 2))

    # r_norm = np.linalg.norm(r,axis = 1)
    # DIFFERS FROM PAPER
    # betas =np.einsum('i,i->i', thetas, density_weights)# / r2
    betas = np.einsum('i,i->i', thetas, r)

    denom = np.einsum('i->', betas)

    if denom > 10 ** -20:
        num = np.einsum('j,jk->k', betas, x)

        term2 = num / denom
    else:
        term2 = np.array(False)

    t2 = time.perf_counter()
    tt = round(t2 - t1, 4)
    return term2, tt


def get_sigma(center, centers, h):
    t1 = time.perf_counter()
    # These are the weights
    r = centers - center
    r2 = np.einsum('ij,ij->i', r, r)
    thetas = np.exp((-r2) / ((h / 2) ** 2))

    # thetas = get_thetas(r,h)
    # Thetas are further clipped to a minimum value to prevent infinite covariance
    # weights = np.clip(thetas, 10**-10, None)
    # substract mean then calculate variance\
    cov = np.einsum('j,jk,jl->kl', thetas, r, r)
    # cov = np.zeros((3,3))
    # for index in range(len(r)):
    #     cov += weights[index]*np.outer(r[index],r[index])
    # centers -= np.mean(centers, axis = 0)
    # # print(centers)
    # cov = np.cov(centers.T, aweights=weights)

    # Get eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov)

    if np.iscomplex(values).any():
        values = np.real(values)
        vectors = np.real(vectors)
        vectors_norm = np.sqrt(np.einsum('ij,ij->i', vectors, vectors))
        vectors = vectors / vectors_norm

    # Argsort always works from low --> to high so taking the negative values will give us high --> low indices
    sorted_indices = np.argsort(-values)

    values_sorted = values[sorted_indices]
    vectors_sorted = vectors[:, sorted_indices]

    sigma = values_sorted[0] / np.sum(values_sorted)

    t2 = time.perf_counter()

    return sigma, vectors_sorted, t2 - t1


def get_h0(points):
    x_max = points[:, 0].max()
    x_min = points[:, 0].min()

    y_max = points[:, 1].max()
    y_min = points[:, 1].min()

    z_max = points[:, 2].max()
    z_min = points[:, 2].min()
    print("BB values: \n\tx:", x_max - x_min, "\n\ty:", y_max - y_min, "\n\tz:", z_max - z_min)

    diagonal = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** .5

    n_points = len(points)

    return 2 * diagonal / (n_points ** (1. / 3))
