#include "march.hpp"
#include <algorithm>
using namespace std;


/*
Only five functions are public in march.hpp:
[1] Constructor:l (const double *X, const double *Y, const double *Z, const int *TRI, int vnum, int tnum)
    You'll need to supply an array for X,Y,Z and the flattended triangled + counts
[2,3,4] Destructor + Two simple getters - No need to talk about these
[3] March(double *SourceVal, double *DistanceData) -
Receives two arrays of size vnum:
- One empty and initialized (DistanceData) - this is will hold the geodesic distances at the very end
- One defined by: SourceVal[i] == 0 if vertex i is a source, else MAX_DISTANCE (defined in globals.hpp)
Note that this code supports multiple source points - which is great :-)

Steps:
    Given Inputs: (char* mesh file name, int* source_indices)
        [1] Load a mesh using https://github.com/tinyobjloader/tinyobjloader for example
        [2] Translate the structure received into the simple parameters needed by the constructor
        [3] Initialize DistanceData and SourceVal as needed
        [4] Call March(SourceVal,DistanceData)
        [5] Print DistanceData to a file or to the screen
        * I've edited this version blindly - Might be some errors here!
Now all we need to do is to copy over the bindings from VTP and adjust some

*/


void gds(double* points_x, double* points_y, double* points_z,
                        int* faces, int num_v, int num_f,
                        double* sourceVal, double* result)
{
    FastMarchingAlgorithm algo(points_x, points_y, points_z, faces, num_v, num_f);
    algo.March(sourceVal, result);
}

void gds_matrix(double* points_x, double* points_y, double* points_z,
                        int* faces, int num_v, int num_f, double* result)
{
   #pragma omp parallel for
	for (int j = 0; j < num_v; j++) {
	    FastMarchingAlgorithm algo(points_x, points_y, points_z, faces, num_v, num_f);
        double* sourceVal = new double[num_v];
        std::fill_n(sourceVal, num_v, MAX_DISTANCE);
        sourceVal[j] = 0;
        algo.heap->reset();
        algo.MarchAlg(algo.V, num_v, sourceVal); //, SourceVal+Vnum);	/* Perform the march */
        for (int i = 0; i < num_v; i++) {	/* Fill up the result matrix with data.	*/
			result[j*num_v+i] = algo.V[i].U;
		}
		delete sourceVal;
	}
}