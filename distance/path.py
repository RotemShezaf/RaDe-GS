import numpy as np

from distance.ex_lib import weighted_vertex_adjacency


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def mesh_path(v, f, src_vi, tgt_vi, distance_cls='euclidean_graph'):
    # TODO - generalize to multiple sources/targets
    # src_vi/tgt_vi either int or None
    # src_vi = np.array([src_vi]) if is_number(src_vi) else np.array(src_vi)
    # tgt_vi = np.array([tgt_vi]) if is_number(tgt_vi) else np.array(tgt_vi)
    if distance_cls == 'euclidean_graph':
        D = graph_path_by_dijkstra(v, f, src_vi, tgt_vi, weight_cls='euclidean')
    elif distance_cls == 'graph':
        D = graph_path_by_dijkstra(v, f, src_vi, tgt_vi, weight_cls=None)
    elif distance_cls == 'geodesic':
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown class {distance_cls}')
    return D


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def graph_path_by_dijkstra(v, f, src_vi, tgt_vi, weight_cls=None):
    import networkx
    G = networkx.from_scipy_sparse_matrix(weighted_vertex_adjacency(v, f, weight_cls=weight_cls))
    path = networkx.algorithms.shortest_path(G, source=src_vi, target=tgt_vi, weight='weight', method='dijkstra')
    path_len = networkx.algorithms.shortest_path_length(G, source=src_vi, target=tgt_vi, weight='weight',
                                                        method='dijkstra')
    return path, path_len
