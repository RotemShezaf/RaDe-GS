import numpy as np
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

from distance.ex_lib import is_number, make_symmetric_by_averaging, stochastic_matrix_normalization, subdivide, \
    weighted_vertex_adjacency
from distance.pointcloud_metrics import euclidean_distance_matrix, pdist

_METHODS = ('euclidean_graph', 'graph', 'euclidean', 'exact_geodesic_via_vtp', 'exact_geodesic_via_gdist',
            'geodesic_via_fmm', 'geodesic_via_heat_method')

_GEODESIC_METHODS = ('euclidean_graph', 'exact_geodesic_via_gdist',
                     'exact_geodesic_via_vtp', 'geodesic_via_fmm', 'geodesic_via_heat_method')

_METHODS_NAME_MAP = {
    'euclidean_graph': 'Dijkstra on Distance Weighted Edges',
    'exact_geodesic_via_gdist': 'Exact MMP Method (geodesic library)',
    'exact_geodesic_via_vtp': 'Exact VTP Method (optimized MMP)',
    'geodesic_via_fmm': 'Unoptimized Fast Marching',
    'geodesic_via_heat_method': 'Optimized Heat Method'
}


# ---------------------------------------------------------------------------------------------------------------------#
#                                                      Collectors
# ---------------------------------------------------------------------------------------------------------------------#
# noinspection PyTypeChecker
def vertex_distance_matrix(v, f=None, cls='euclidean_graph', max_dist=None, knn_truncation=None, subdivide_order=0,
                           make_symmetric=False, squared=False):
    nv = v.shape[0]
    if subdivide_order:
        # assert cls in ['geodesic_via_fmm', 'geodesic_via_heat_method'] # TODO - where is this effective?
        v, f = subdivide(v, f, n=subdivide_order)
    if cls == 'euclidean_graph':
        # Fast approximation to the true geodesic distances
        D = graph_distance_matrix(v=v, f=f, weight_cls='euclidean', max_dist=max_dist)
    elif cls == 'graph':
        D = graph_distance_matrix(v=v, f=f, weight_cls=None, max_dist=max_dist)
    elif cls == 'euclidean':
        D = euclidean_distance_matrix(v=v, max_dist=max_dist, take_root=not squared)
    elif cls == 'exact_geodesic_via_vtp':
        D = exact_geodesic_via_vtp_distance_matrix(v, f, max_dist)
    elif cls == 'exact_geodesic_via_gdist':
        D = exact_geodesic_via_gdist_distance_matrix(v, f, max_dist)
    elif cls == 'geodesic_via_fmm':
        D = geodesic_via_fmm_distance_matrix(v=v, f=f, max_dist=max_dist)
    else:
        raise NotImplementedError(f'Unknown class {cls}')
    if subdivide_order:
        D = D[:nv, :nv]  # Truncate additional distances - not useful to us

    if squared and cls != 'euclidean':
        D = np.power(D, 2)

    if make_symmetric:
        D = make_symmetric_by_averaging(D)

    if knn_truncation is not None:
        # https://github.com/lmcinnes/umap/issues/114
        indices = np.argsort(D)
        indices_row = np.multiply(np.ones(D.shape[1], dtype=np.int32), np.arange(D.shape[0], dtype=np.int32)[:, None])
        D = D[indices_row, indices]  # Sorted D
        indices = indices[:, : knn_truncation]
        D = D[:, : knn_truncation]
        return D, indices
    else:
        return D


def vertex_dist(v, f, src_vi, cls='euclidean_graph', sources_are_disjoint=True, subdivide_order=0):
    nv = v.shape[0]
    if subdivide_order:
        # assert cls in ['geodesic_via_fmm', 'geodesic_via_heat_method'] # TODO - where is this effective?
        v, f = subdivide(v, f, n=subdivide_order)
    src_vi = np.array([src_vi]) if is_number(src_vi) else np.array(src_vi)
    if cls == 'euclidean_graph':
        # Fast approximation to the true geodesic distances
        D = graph_vertex_distance(v, f, src_vi, weight_cls='euclidean')
    elif cls == 'graph':
        D = graph_vertex_distance(v, f, src_vi)
    elif cls == 'euclidean':
        D = pdist(v[src_vi, :], v, 2)  # Handles singleton case
    elif cls == 'exact_geodesic_via_vtp':
        D = exact_geodesic_via_vtp_vertex_distance(v, f, src_vi)
    elif cls == 'exact_geodesic_via_gdist':
        D = exact_geodesic_via_gdist_vertex_distance(v, f, src_vi, sources_are_disjoint=sources_are_disjoint)
    elif cls == 'geodesic_via_fmm':
        D = geodesic_via_fmm_vertex_distance(v, f, src_vi, sources_are_disjoint=sources_are_disjoint)
    else:
        raise NotImplementedError(f'Unknown class {cls}')
    if subdivide_order:
        D = D[..., :nv]

    if D.ndim == 2:
        if D.shape[0] == 1:  # Remove singleton
            D = np.asanyarray(D).ravel()
        elif not sources_are_disjoint:
            D = np.min(D, axis=0)  # Handle multiple sources for the first 3 metrics here
    return D


def n_radical_point_vertex_dist(v, f, n=6, cls="geodesic_via_fmm"):
    # Approximate the geodesic distance matrix by up to 6 furthest points in the mesh
    src_vis = n_radical_point_indices(v=v, n=n)
    return vertex_dist(v=v, f=f, src_vi=src_vis, cls=cls, sources_are_disjoint=True)


def diffusion_distance_matrix(v, f, t=10, dist_cls='geodesic_via_fmm', normalization_cls='col_stochastic'):
    D = vertex_distance_matrix(v, f, cls=dist_cls)
    K = np.exp(-0.5 * (D / t) ** 2)  # 2 * t^2 == EPS from Wikipedia Article
    P = stochastic_matrix_normalization(K, normalization_cls=normalization_cls)
    Pt = np.linalg.matrix_power(P, int(t))  # TODO - solve coupling of eps and the time here.
    # Usually we need to multiply by speye (matrix of singleton distributions) - but this is redundant
    # We conclude that the columns of Pt are the feature vectors
    return euclidean_distance_matrix(Pt.T)


def diffusion_distance_vector(v, f, src_vi, t=10, dist_cls='geodesic_via_fmm', normalization_cls='col_stochastic'):
    # TODO - reduce complexity
    D = diffusion_distance_matrix(v=v, f=f, t=t, dist_cls=dist_cls, normalization_cls=normalization_cls)
    return D[:, src_vi]


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def n_radical_point_indices(v, n):
    # TODO - consider moving to a utility file
    assert 1 <= n <= 6, "MaxX,MinX,MaxY,MinY,MaxZ,MinZ (ordered) are the possible radical points in 3D"
    src_vis = []
    for i in range(n):
        curr_index = i // 2  # 0 , 0 , 1, ,1 , 2, 2
        curr_function = np.argmax if i % 2 == 0 else np.argmin
        src_vis.append(curr_function(v[:, curr_index]))
    return src_vis


def vertex_kdtree(v):
    from scipy.spatial import cKDTree
    tree = cKDTree(v)
    return tree


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Distance Matrices
# ---------------------------------------------------------------------------------------------------------------------#

# noinspection PyTupleItemAssignment
def graph_distance_matrix(v, f, weight_cls=None, max_dist=None):
    G = weighted_vertex_adjacency(v, f, weight_cls=weight_cls)
    D = shortest_path(G, directed=False, return_predecessors=False, unweighted=False, indices=None)
    if max_dist is not None:
        D[D > max_dist] = np.inf  # Remove all these entries
    return D


def geodesic_via_fmm_distance_matrix(v, f, max_dist=None):
    # TODO - does not turn out symmetric...
    import external.fmm.fmmdist as fmm
    D = fmm.geodesic_matrix(v.astype(np.float64), f.astype(np.int32))  # Assumed double,int
    if max_dist is not None:
        D[D > max_dist] = np.inf  # Remove all these entries
    return D


def exact_geodesic_via_vtp_distance_matrix(v, f, max_dist=None):
    import external.vtp.vtpdist as vtp
    D = vtp.geodesic_matrix(v.astype(np.float64), f.astype(np.int32))  # Assumed double,int
    if max_dist is not None:
        D[D > max_dist] = np.inf  # Remove all these entries
    return D


def exact_geodesic_via_gdist_distance_matrix(v, f, max_dist=None):
    # TODO - should we parallelize?
    import gdist as gd
    max_dist = np.inf if max_dist is None else max_dist
    return gd.local_gdist_matrix(v.astype(np.float64), f.astype(np.int32), max_dist)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                 Distance Vectors
# ---------------------------------------------------------------------------------------------------------------------#
def geodesic_via_fmm_vertex_distance(v, f, src_vi, sources_are_disjoint):
    import distance.external.fmm.fmmdist as fmm
    if sources_are_disjoint:
        D = np.zeros((len(src_vi), v.shape[0]))
        for i, vi in enumerate(tqdm(src_vi, desc="Computing FMM geodesics")):
            D[i, :] = fmm.geodesic_distance(v, f.astype('int32'), np.array([vi], dtype=np.int32))
        return D
    return fmm.geodesic_distance(v.astype('float64'), f.astype('int32'), np.array(src_vi, dtype=np.int32))


def exact_geodesic_via_gdist_vertex_distance(v, f, src_vi, sources_are_disjoint):
    import gdist as gd
    if sources_are_disjoint:
        D = np.zeros((len(src_vi), v.shape[0]))
        for i, vi in enumerate(tqdm(src_vi, desc="Computing exact geodesics (gdist)")):
            D[i, :] = gd.compute_gdist(v, f.astype('int32'), np.array([vi], dtype=np.int32))
        return D
    return gd.compute_gdist(v, f.astype('int32'), np.array(src_vi, dtype=np.int32))


def exact_geodesic_via_vtp_vertex_distance(v, f, src_vi, sources_are_disjoint):
    import distance.external.vtp.vtpdist as vtp
    if sources_are_disjoint:
        D = np.zeros((len(src_vi), v.shape[0]))
        for i, vi in enumerate(tqdm(src_vi, desc="Computing exact geodesics (VTP)")):
            D[i, :] = vtp.geodesic_distance(v, f.astype('int32'), vi)
        return D
    return vtp.geodesic_distance(v.astype('float64'), f.astype('int32'), np.array(src_vi, dtype=np.int32))


def graph_vertex_distance(v, f, src_vi, weight_cls=None):
    G = weighted_vertex_adjacency(v, f, weight_cls=weight_cls)
    return shortest_path(G, directed=False, return_predecessors=False, unweighted=False, indices=src_vi)
