import numpy as np

import skeleton.utils as utils


def ls_ellipsoid(xx, yy, zz):
    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    z = zz[:, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x * x, y * y, z * z, x * y, x * z, y * z, x, y, z))
    K = np.ones_like(x)  # column of ones

    # np.hstack performs a loop over all samples and creates
    # a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ);
    ABC = np.dot(InvJTJ, np.dot(JT, K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa = np.append(ABC, -1)

    return eansa


def get_ellipsoid_center(vec):
    # convert the polynomial form of the 3D-ellipsoid to parameters
    # center, axes, and transformation matrix
    # vec is the vector whose elements are the polynomial
    # coefficients A..J
    # returns (center, axes, rotation matrix)

    # Algebraic form: X.T * Amat * X --> polynomial form

    Amat = np.array(
        [
            [vec[0], vec[3] / 2.0, vec[4] / 2.0, vec[6] / 2.0],
            [vec[3] / 2.0, vec[1], vec[5] / 2.0, vec[7] / 2.0],
            [vec[4] / 2.0, vec[5] / 2.0, vec[2], vec[8] / 2.0],
            [vec[6] / 2.0, vec[7] / 2.0, vec[8] / 2.0, vec[9]]
        ])

    # See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3 = Amat[0:3, 0:3]
    A3inv = np.linalg.inv(A3)
    ofs = vec[6:9] / 2.0
    center = -np.dot(A3inv, ofs)

    return center


def recenter_around(center, neighbors):
    normal = center.eigen_vectors[:, 0]
    normal = utils.unit_vector(normal)

    p = center.center

    projected = np.array([utils.project_one_point(c.center, p, normal) for c in neighbors])

    xs = projected[::, 0]
    ys = projected[::, 1]
    zs = projected[::, 2]

    print(xs)

    vec = ls_ellipsoid(xs, ys, zs)

    try:
        ellipsoid_center = get_ellipsoid_center(vec)
        center.center = ellipsoid_center
    except:
        print("Singular mat:", vec)

    return center
