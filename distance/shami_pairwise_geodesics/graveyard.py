import multiprocessing as mp

import gdist
import gdist as gd
import geom.np.mesh.distance.external.fmm.fmmdist as fmm
import geom.np.mesh.distance.external.vtp.vtpdist as vtp
import numpy as np

from geom.np.mesh.surface.mass import surface_area
from numeric.np import print_l1_metrics
from util.performance import timer


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def exact_geodesics(v, f, src=None, dest=None, normalize_by_area=True,
                    max_distance=float('inf'), num_workers=0):
    """
    :param v: [nv x 3] vertices
    :param f: [nf x 3] faces
    :param src: np.array of source nodes or None to use all
    :param dest: np.array of destination nodes or None to use all
    :param normalize_by_area: Normalizes by sqrt(mesh_surface_area)
    :param max_distance: If supplied, will only yield results for geodesic distances with less than max_distance.
    This will speed up runtime dramatically.
    :param num_workers: How many subprocesses to use for calculating geodesic distances.
        num_workers = 0 means that computation takes place in the main process.
        num_workers = -1 means that the available amount of CPU cores is used.
    :return:
    """

    def _parallel_loop(pos, face, src, dest, max_distance, norm, i):
        s = src[i:i + 1]
        d = None if dest is None else dest[i:i + 1]
        return gdist.compute_gdist(pos, face, s, d, max_distance * norm) / norm

    area_normalization = np.sqrt(surface_area(v, f)) if normalize_by_area else 1

    if src is None and dest is None:
        out = gdist.local_gdist_matrix(v, f, max_distance * area_normalization) / area_normalization
    if src is None:
        src = np.arange(v.shape[0], dtype=np.int32)

    dest = None if dest is None else dest

    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            outs = pool.starmap(
                _parallel_loop,
                [(v, f, src, dest, max_distance, normalize_by_area, i)
                 for i in range(len(src))])
    else:
        outs = [
            _parallel_loop(v, f, src, dest, max_distance, normalize_by_area, i)
            for i in range(len(src))
        ]

    return np.cat(outs, dim=0)


def main():
    from cfg import Assets
    from geom.tool.vis.vista import plot_mesh_montage,highlight_connected_components
    v, f = Assets.MAN1_HW3.load()
    highlight_connected_components(v,f)
    #
    with timer('FMM'):
        d1 = fmm.geodesic_distance(v.astype(np.float64), f.astype(np.int32), np.array([0, 400]).astype(np.int32))
    #
    # with timer('VTP'):
    #     d2 = vtp.geodesic_distance(v, f, 0)

    # with timer('GDIST'):
    #     d3 = gd.compute_gdist(v.astype(np.float64), f.astype(np.int32), np.array([0, 600],dtype=np.int32))

    plot_mesh_montage(vs=[v] , fs=f, colors=[d1])

    with timer('FMM'):
        d1 = fmm.geodesic_matrix(v, f)
    #
    # with timer('VTP'):
    #     d2 = vtp.geodesic_matrix(v, f)

    # with timer('GDIST'):
    #     d3 = gd.local_gdist_matrix(v.astype(np.float64), f.astype(np.int32), np.inf)

    # print_l1_metrics(d1, d2)


if __name__ == "__main__":
    main()
