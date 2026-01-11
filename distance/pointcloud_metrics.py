import numpy as np
from scipy.spatial.ckdtree import cKDTree

from distance.ex_lib import last_axis_norm


# TODO - more distance computes:
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
# ---------------------------------------------------------------------------------------------------------------------#
#                                         P-Norm Point to Point Metrics
# ---------------------------------------------------------------------------------------------------------------------#

def pdist(v1, v2, p=2, take_root=True, max_dist=None):
    """
    Compute the pairwise distance matrix between a and b which both have size [m, n, d] or [n, d].
    The result is a tensor of size [m, n, n] (or [n, n]) whose entry [m, i, j] contains the distance_tensor between
    a[m, i, :] and b[m, j, :].
    """
    v1, v2 = np.atleast_2d(v1), np.atleast_2d(v2)
    squeezed = False
    if len(v1.shape) == 2 and len(v2.shape) == 2:
        v1 = v1[np.newaxis, :, :]
        v2 = v2[np.newaxis, :, :]
        squeezed = True

    if len(v1.shape) != 3:
        raise ValueError("Invalid shape for v1. Must be [m, n, d] or [n, d] but got", v1.shape)
    if len(v2.shape) != 3:
        raise ValueError("Invalid shape for v2. Must be [m, n, d] or [n, d] but got", v2.shape)

    D = np.power(np.abs(v1[:, :, np.newaxis, :] - v2[:, np.newaxis, :, :]), p).sum(3)
    if take_root:
        D = D ** (1 / p)

    if max_dist is not None:
        D[D > max_dist] = np.inf  # Remove all these entries

    if squeezed:
        D = np.squeeze(D)

    return D


def euclidean_distance_matrix(v, take_root=True, max_dist=None):
    # Faster than pdist(x,x,2), and equivalent up to 1e-9
    r = np.sum(v * v, 1)
    r = r.reshape(-1, 1)
    D = r - 2 * np.dot(v, v.T) + r.T
    D[D < 0] = 0  # Remove negative zeros
    if take_root:
        D = np.sqrt(D)
    if max_dist is not None:
        D[D > max_dist] = np.inf  # Remove all these entries
    return D


def nearest_neighbor(src, dst, k=1, p=2, max_dist=np.inf, without_self_match=True):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor

    NOTE: If k is very large, it is better to use vertex_distance_matrix with knn_truncation in terms of time.
    Results are identical
    NOTE: This implementation is faster than the one in sklearn.neighbors
    """
    if without_self_match:
        k += 1
    assert src.shape == dst.shape
    distances, indices = cKDTree(src).query(dst, p=p, k=k,
                                            distance_upper_bound=max_dist)  # See options eps, n_jobs here
    if without_self_match:
        return distances[:, 1:], indices[:, 1]
    return distances, indices


def closest_neighbor(src, dst, p=2):
    return nearest_neighbor(src, dst, p=p, k=1, without_self_match=True)


def vertex_centroid_distance(v):
    c = np.mean(v, axis=0, keepdims=True)
    cd = last_axis_norm(v - c)
    return cd, c


# ---------------------------------------------------------------------------------------------------------------------#
#                                                  Metrics
# ---------------------------------------------------------------------------------------------------------------------#
# noinspection PyArgumentList
def directional_chamfer(source, target, p=2):
    """
    Compute the chamfer distance between two point clouds a,b by the Lp norm
    :param source: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param target: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :param p: Norm to use for the distance_tensor
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    M = pdist(source, target, p)
    if len(M.shape) == 2:
        M = M[np.newaxis, :, :]

    # TODO - make sure the source and target and not reversed
    return M.min(1).sum(1), M.min(2).sum(1)  # sum the two for the full chamfer distance


def full_chamfer(source, target, p=2):
    return sum(directional_chamfer(source, target, p))


def euclidean_directional_hausdorff(source, target, return_index=True):
    """
    # TODO -Take a loot at scipy.spatial.distance
    The Hausdorff distance is the longest distance you can be forced to travel by an adversary who chooses
    a point in one of the two sets, from where you then must travel to the other set.
    In other words, it is the greatest of all the distances from a point in one set to
    the closest point in the other set.
    source : n by 3 array of representing a set of n points (each row is a point of dimension 3)
    target : m by 3 array of representing a set of m points (each row is a point of dimension 3)
    return_index : Optionally return the index pair `(i, j)` into source and target such that
               `source[i, :]` and `target[j, :]` are the two points with maximum shortest distance.
    # Note: Take a max of the one sided dist to get the two sided Hausdorff distance
      return max(hausdorff_a_to_b, hausdorff_b_to_a)
    :param source:
    :param target:
    :param return_index
    :return:
    """
    import point_cloud_utils as pcu
    # Compute each one sided *squared* Hausdorff distances
    return pcu.hausdorff(source, target, return_index=return_index)
