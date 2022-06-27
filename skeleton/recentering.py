import skeleton.utils as utils


def recenter_around(center, neighbors):
    normal = center.eigen_vectors[:, 0]
    normal = utils.unit_vector(normal)
    p = center.center

    projected = [utils.project_one_point(c.center, p, normal) for c in neighbors]

    return center
