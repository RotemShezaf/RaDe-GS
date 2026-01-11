import warnings
import numpy as np
from typing import Union
import scipy.sparse as sp
import numbers

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def is_number(obj):
    return isinstance(obj, numbers.Number)

def weighted_vertex_adjacency(v, f, weight_cls='uniform'):
    if weight_cls == 'euclidean':
        W = euclidean_weights(v, f, inverse=False)
    elif weight_cls in {'uniform', 'combinatorial', 'graph', 'umbrella'}:
        W = vertex_adjacency(v, f, include_diagonal=False)
    else:
        raise NotImplementedError(f'Unknown weight class: {weight_cls})')

    return W

def vertex_adjacency(v, f, include_diagonal=False):
    vf = vertex_face_adjacency(v, f)
    A = vf @ vf.transpose()
    if not include_diagonal:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # WARNING: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
            A.setdiag(False)
            A.eliminate_zeros()
    return A


def vertex_face_adjacency(v, f, weight: Union[str, None, np.ndarray] = None):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A weight vector can be passed which is then used instead of booleans - for example, the face areas
    weight vector format: [face0,face0,face0,face1,face1,face1,...]
    """
    row = f.ravel()  # Flatten indices
    col = np.repeat(np.arange(len(f)), 3)  # Data for vertices

    if weight is None:
        weight = np.ones(len(col), dtype=np.bool)
    # Otherwise, we suppose that 'weight' is a vector of the needed size.

    vf = sp.csr_matrix((weight, (row, col)), shape=(v.shape[0], len(f)), dtype=weight.dtype)
    return vf


def euclidean_weights(v, f, inverse=False):
    # TODO -  I'm not sure adj matrices are equivalent
    de, e = edge_lengths(v, f, return_edges=True)
    if inverse:
        de = 1 / de
    ii, jj = np.concatenate((e[:, 0], e[:, 1])), np.concatenate((e[:, 1], e[:, 0]))
    vals = np.concatenate((de, de))
    return sp.csr_matrix((vals, (ii, jj)), shape=(len(v), len(v)), dtype='float64')


def edge_lengths(v, f, return_edges=False):
    e = edges(v, f, unique=True)
    d = last_axis_norm(v[e[:, 0], :] - v[e[:, 1], :])
    if return_edges:
        return d, e
    return d


def edges(v, f, unique=True, return_index=False, return_inverse=False, return_counts=False):
    e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)
    return np.unique(e, axis=0, return_inverse=return_inverse, return_index=return_index,
                     return_counts=return_counts) if unique else e


def last_axis_norm(mat):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(np.power(mat, 2), axis=-1))


def subdivide(v, f, fi=None, n=1, return_map=False):
    # May be implemented via https://github.com/PyMesh/PyMesh/blob/master/python/pymesh/meshutils/generate_box_mesh.py
    # face_index : faces to subdivide.
    #   if None: all faces of mesh will be subdivided
    #   if (n,) int array of indices: only specified faces
    import trimesh.remesh
    # TODO - add some measurement of the memory complexity and stop the computation if too high
    new_v, new_f = v, f
    for i in range(n):
        new_v, new_f = trimesh.remesh.subdivide(new_v, new_f, face_index=fi)
    assert np.max(new_f) < 2 ** 31, "Cast problem"
    new_f = new_f.astype(np.int32)

    if return_map:
        return new_v, new_f, np.arange(v.shape[0])  # Map is trivial
    return new_v, new_f


def make_symmetric_by_averaging(M):
    return 0.5 * (M + M.T)


def stochastic_matrix_normalization(W, normalization_cls='lazy_col_stochastic'):
    if normalization_cls in {'row_stochastic', 'row_random_walk', 'lazy_row_random_walk', 'lazy_row_stochastic'}:
        Dinv = sp.spdiags(1 / np.array(np.sum(W, 1)).squeeze(), 0, W.shape[0], W.shape[0])
        W = Dinv @ W  # Normalize Rows
        if normalization_cls in {'lazy_row_random_walk', 'lazy_row_stochastic'}:
            # Intuition - We add 1 to every eigenvalue, thus avoiding bad condition numbers (more stable!).
            W = 0.5 * (W + sp.eye(W.shape[0]))
    elif normalization_cls in {'col_stochastic', 'col_random_walk', 'lazy_col_random_walk', 'lazy_col_stochastic'}:
        Dinv = sp.spdiags(1 / np.array(np.sum(W, 1)).squeeze(), 0, W.shape[0], W.shape[0])
        W = W @ Dinv  # Normalize Cols
        if normalization_cls in {'lazy_col_random_walk', 'lazy_col_stochastic'}:
            # Intuition - We add 1 to every eigenvalue, thus avoiding bad condition numbers (more stable!).
            # Very related to the symmetric normalized graph laplacian (~flipud spectrum)
            W = 0.5 * (W + sp.eye(W.shape[0]))
    elif normalization_cls in {'symmetric'}:
        D_half_inv = sp.spdiags(1 / np.sqrt(np.array(np.sum(W, 1)).squeeze()), 0, W.shape[0], W.shape[0])
        # normalization = np.sum(K, axis=-1, keepdims=True) # TODO
        #     P = K / np.sqrt(normalization@normalization.T) - FASTER
        W = D_half_inv @ W @ D_half_inv
        # np.sum(W,axis=0).squeeze() == np.sum(W,axis=1).squeeze(), but it is not doubly stochastic...
    else:
        raise NotImplementedError(f'Unknown normalization class: {normalization_cls})')

    return W
