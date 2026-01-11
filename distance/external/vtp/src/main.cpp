#include "stdafx.h"
#include "geodesic_mesh.h"
#include "geodesic_algorithm_exact.h" 

using namespace std;

std::vector<double> gds(double* points, unsigned num_v, int* faces, unsigned num_f, unsigned int source)
{
	// Build Mesh
	geodesic::Mesh mesh;

	mesh.initialize_mesh_data(num_v, points, num_f, faces);
	geodesic::GeodesicAlgorithmExact algorithm(&mesh);

	// Propagation
	algorithm.propagate(source);	//cover the whole mesh

	// Output Geodesic Distances
	std::vector<geodesic::Vertex>& v = mesh.vertices();
    std::vector<double> distances(num_v);
	for(unsigned i=0; i<num_v; ++i)
		distances[i] =  v[i].geodesic_distance();

    return distances;
}

void gds_matrix(double* points, unsigned num_v, int* faces, unsigned num_f,
                                            double* result)
{
   #pragma omp parallel for
    for(unsigned j=0; j<num_v; j++){
        geodesic::Mesh mesh;
        mesh.initialize_mesh_data(num_v, points, num_f, faces);
        std::vector<geodesic::Vertex>& v = mesh.vertices();
    	geodesic::GeodesicAlgorithmExact algorithm(&mesh);
        algorithm.propagate(j);
        for(unsigned i=0; i<num_v; ++i)
		    result[num_v*j + i] =  v[i].geodesic_distance();
    }
}