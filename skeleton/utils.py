import numpy as np
import random


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def get_local_points(points, centers, h, max_local_points=50000):
    # Get local_points points around this center point
    local_indices = []
    for center in centers:
        x, y, z = center

        # 1) first get the square around the center
        where_square = ((points[:, 0] >= (x - h)) & (points[:, 0] <= (x + h)) & (points[:, 1] >= (y - h)) &
                        (points[:, 1] <= (y + h)) & (points[:, 2] >= (z - h)) & (points[:, 2] <= (z + h)))

        square = points[where_square]
        indices_square = np.where(where_square == True)[0]

        # Get points which comply to x^2, y^2, z^2 <= r^2
        square_squared = np.square(square - [x, y, z])
        where_sphere = np.sum(square_squared, axis=1) <= h ** 2
        local_sphere_indices = indices_square[where_sphere]

        local_indices.append(local_sphere_indices)

    if len(local_indices) > max_local_points:
        return random.sample(local_indices, max_local_points)

    return local_indices


def project_one_point(q, p, n):
    """
    :param q: a point
    :param p: the point on the plane
    :param n: the normal vector of the plane
    :return: the projected point
    """
    n = unit_vector(n)
    return q - np.dot(q - p, n) * n
