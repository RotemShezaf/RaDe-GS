#ifndef MARCH
#define MARCH
#include "globals.hpp"
#include "heap.h"
class FastMarchingAlgorithm {

public:

    FastMarchingAlgorithm(const double *X, const double *Y, const double *Z, const int *TRI, int vnum, int tnum) : Vnum(vnum), Tnum(3 * tnum) {
		CombineVectors(&V, &T, &NonSplitTnum, X, Y, Z, TRI, Vnum, Tnum);
        InitGraph(T,V, Vnum, NonSplitTnum, Tnum); /* init V to isolated vertices */
		heap = new Heap(Vnum);
	}
	void March(double *SourceVal, double *DistanceData) {
		heap->reset();
		MarchAlg(V, Vnum, SourceVal); //, SourceVal+Vnum);	/* Perform the march */
		for (int i = 0; i < Vnum; i++) {	/* Fill up the result matrix with data.	*/
			*(DistanceData++) = V[i].U;
		}
	}

	~FastMarchingAlgorithm() {
		/* Free the resources used by the program. */
		delete heap;
		free(T);
		free(V);
	}

	int GetNumberOfVertices() { return Vnum; }
	int GetNumberOfTrianglesTimes3() { return Tnum; }

public:

	void InitGraph(Triangle * T, Vertex * V, int Vnum, int NonSplitTnum, int Tnum);
	void CombineVectors(Vertex *(V[]), Triangle *(T[]), int *NonSplitTnum,
		const double *X, const double *Y, const double *Z, const int *TrianglesData,
		int Vnum, int Tnum);
	void InitMarch(double srcval[], int Vnum);
	void MarchAlg(Vertex  *temp_V, int temp_Vnum, double *srcval);
	void  CallTUpdateNeighbors(
		int i,	/* vertex number */
		int start_from		/* The source from which the current march began.	*/
	);
    void TUpdateNeighbors(
		int    i,						/* The vertex to update				   */
		int	   becomes_alive			/* The vertex that now becomes alive.  */
	);
    double update(Stencil *current_stencil, int k, int l);

public:

	Triangle *T; 	/* triangles, numbered 1..Tnum */
	Vertex   *V; 	/*  vertices coordinates 1..VNum */
	int	Tnum, Vnum;
	int NonSplitTnum;		/* The number of triangles with no splitting. */
							/* Names of files that are used for writing the results. */
	Heap *heap;

};

void FastMarchingAlgorithm::InitGraph(Triangle* T, Vertex* V, int Vnum, int NonSplitTnum, int Tnum){
	int i, k, ti, v1, v2, count = 0, mcount = 0;
	bool         found;			/* Used for adding the neighbours to
									   every triangle.						*/
	double          ca;
	 Stencil *p_st;
	int ind, si_count;				/* Used for precalculting values for the
									   stencil.								*/


									   /* Initialize the vertixes. */
	for (i = 0; i < Vnum; i++) {	/* zero counters of all vertices */
		V[i].si = 0;				/* # of connections to other triangles */
		V[i].vn = 0;				/* # of connections to other vertices */
	}

	/* Set the split field of the triangles that exist now, before the splitting to false.	*/
	for (i = 0; i < NonSplitTnum; i++)
		T[i].Split = false;


	for (i = 0; i < Vnum; i++) {		/* scan all vertices */
		for (ti = 0; ti < V[i].ti; ti++) {/* scan connected triangles */
			if (V[i].Tind[ti] < NonSplitTnum) {	/* if valid triangle */
												/* Make v1 and v2 the neighbours.			*/
				if (T[V[i].Tind[ti]].Vind[0] == i) {
					v1 = T[V[i].Tind[ti]].Vind[1];
					v2 = T[V[i].Tind[ti]].Vind[2];
				}
				else if (T[V[i].Tind[ti]].Vind[1] == i) {
					v1 = T[V[i].Tind[ti]].Vind[2];
					v2 = T[V[i].Tind[ti]].Vind[0];
				}
				else if (T[V[i].Tind[ti]].Vind[2] == i) {
					v1 = T[V[i].Tind[ti]].Vind[0];
					v2 = T[V[i].Tind[ti]].Vind[1];
				}

				found = false;					/* Add v1 as a neighbour if it is not already
												   a neighbour.								*/
				for (k = 0; k < V[i].vn; k++)
					if (v1 == V[i].VN[k])
						found = true;
				if (!found)
					V[i].VN[V[i].vn++] = v1;

				found = false;					/* Add v2 as a neigbour if it is not already
												   a neighbour.								*/
				for (k = 0; k < V[i].vn; k++)
					if (v2 == V[i].VN[k])
						found = true;
				if (!found)
					V[i].VN[V[i].vn++] = v2;

				ca = CosAngle(i, v1, v2, V);
				if (ca < 0) {					/* If this triangle is an obtuse angle		*/
					count = Split(V[i].Tind[ti], i, v1, v2,
						T, V, NonSplitTnum, Vnum);
					if (count > mcount)			/* Update m count.							*/
						mcount = count;
				}
				else {							/* If no splitting was nessesery create
												   the stencil for this vertex and triangle.*/
					V[i].ST[V[i].si].Ctheta = ca;
					V[i].ST[V[i].si].v1 = v1;
					V[i].ST[V[i].si].l1 = Length(i, v1, V);
					V[i].ST[V[i].si].v2 = v2;
					V[i].ST[V[i].si].l2 = Length(i, v2, V);
					V[i].si++;
				}
			}
		}
	}

	for (ind = 0; ind < Vnum; ind++)					/* Calculate the data for each stencil.	*/
		for (p_st = V[ind].ST, si_count = V[ind].si - 1; si_count >= 0; p_st++, si_count--) {
			p_st->Stheta = 1 - SQR(p_st->Ctheta);
			p_st->Ctheta_mul_l1_div_l2_minus_one_mul_2 = p_st->Ctheta * p_st->l1 * 2 / p_st->l2 - 2;
			p_st->Ctheta_mul_l2_div_l1_minus_one_mul_2 = p_st->Ctheta * p_st->l2 * 2 / p_st->l1 - 2;
			p_st->sqr_l1 = p_st->l1 * p_st->l1;
			p_st->sqr_l2 = p_st->l2 * p_st->l2;
			p_st->shortcut1_1 = 1 - p_st->sqr_l1 * (p_st->Ctheta_mul_l2_div_l1_minus_one_mul_2 + 1) / p_st->sqr_l2;
			p_st->shortcut1_2 = 1 - p_st->sqr_l2 * (p_st->Ctheta_mul_l1_div_l2_minus_one_mul_2 + 1) / p_st->sqr_l1;
			p_st->shortcut2_1 = -p_st->Stheta * p_st->sqr_l1;
			p_st->shortcut2_2 = -p_st->Stheta * p_st->sqr_l2;
		}
}

void FastMarchingAlgorithm::CombineVectors( Vertex *(V[]),  Triangle *(T[]), int *NonSplitTnum,
	const double *X, const double *Y, const double *Z, const int *TrianglesData,
	int Vnum, int Tnum) {

	int i, j;

	*NonSplitTnum = Tnum / 3;

	/* Allocate memory for both triangles and
	   vertixes.							*/
	*T = ( Triangle *) malloc(sizeof( Triangle) * Tnum);
	if (*T == NULL) {
		fprintf(stderr, "Out of memory for triangles - exiting.\n");
		exit(-1);
	}
	*V = ( Vertex *)   malloc(sizeof( Vertex) * Vnum);
	if (*V == NULL) {
		free(T);
		fprintf(stderr, "Out of memory for vertices - exiting.\n");
		exit(-1);
	}

    for (i = 0; i < Vnum; i++) {
		(*V)[i].x = ((double)X[i]);
		(*V)[i].y = ((double)Y[i]);
		(*V)[i].z = ((double)Z[i]);
	}

	for (i = 0; i < 3; i++)
		for (j = 0; j < *NonSplitTnum; j++) {
			(*T)[j].Vind[i] = *((int *)TrianglesData++);
		}

	for (i = 0; i < Vnum; i++) {						/* Add every triangle to its vertixes.	*/
		(*V)[i].ti = 0;
	}
	/* Can be greatly improved! */
	for (i = 0; i < *NonSplitTnum; i++) {
		if ((*T)[i].Vind[0] != *NonSplitTnum) {
			(*V)[(*T)[i].Vind[0]].Tind[(*V)[(*T)[i].Vind[0]].ti++] = i;
			(*V)[(*T)[i].Vind[1]].Tind[(*V)[(*T)[i].Vind[1]].ti++] = i;
			(*V)[(*T)[i].Vind[2]].Tind[(*V)[(*T)[i].Vind[2]].ti++] = i;
		}
	}

	return;

}

void FastMarchingAlgorithm::InitMarch(double srcval[], int Vnum) {
	int i;
	for (i = 0; i < Vnum; i++) {

		if (srcval[i] >= MAX_DISTANCE) {
			V[i].U = MAX_DISTANCE;					/* Initialize the distance				*/
			heap->BP[i] = FAR;							/* Originally all are far.				*/
			V[i].source = false;
			//V[i].source_num = 0;
		}
		else {
			V[i].U = srcval[i];
			heap->insert(V[i].U, i);
			V[i].source = true;
			//V[i].source_num = srcnum[i];	/* Set the source num field.			*/
		}

	}
}

bool Quadratic(double qa, double qb, double qc, double *x1){

	double d;

	d = qb * qb - (qa + qa)*(qc + qc); /* Discremenat */

	if (d >= 0) {  /* in case the Discremenat >= 0 */
		*x1 = (sqrt(d) - qb) / (qa + qa);
		return true;
	}
	else {
		return false;
	}
}

double FastMarchingAlgorithm::update( Stencil *current_stencil, int k, int l) {

	double u1, u2;
	double u;
	double t;
	double /*ctheta_mul_mb_div_ma_minus_one_mul_2,*/ ctheta_mul_ma_div_mb_minus_one_mul_2;
	/* Both of them are between 0 and 1 becuase
	   the triangles are not obuse.				*/

	u1 = V[k].U;
	u2 = V[l].U;
	u = u2 - u1;
	if (u >= 0) {
		ctheta_mul_ma_div_mb_minus_one_mul_2 = current_stencil->Ctheta_mul_l2_div_l1_minus_one_mul_2;
		/* A shortcut */
		if (Quadratic(current_stencil->shortcut1_2,							/* If the quadratic equation has solutions */
			u * ctheta_mul_ma_div_mb_minus_one_mul_2,
			(u * u + current_stencil->shortcut2_2), &t)
			&& (-(u + u) >= ctheta_mul_ma_div_mb_minus_one_mul_2 * t)
			&& current_stencil->Ctheta_mul_l1_div_l2_minus_one_mul_2*(t - u) <= (u + u))
			return (u1 + t);
		else									/* If the quadratic equation has no solution,
												   or the solutions are out of the trinagle */
			return MAX_DISTANCE;
	}
	else {
		ctheta_mul_ma_div_mb_minus_one_mul_2 = current_stencil->Ctheta_mul_l1_div_l2_minus_one_mul_2;
		/* A shortcut */

		if (Quadratic(current_stencil->shortcut1_1,						/* If the quadratic equation has solutions */
			-u * ctheta_mul_ma_div_mb_minus_one_mul_2,
			(u * u + current_stencil->shortcut2_1), &t)
			&& (u + u >= ctheta_mul_ma_div_mb_minus_one_mul_2 * t)
			&& current_stencil->Ctheta_mul_l2_div_l1_minus_one_mul_2*(t + u) <= -(u + u))
			return (u2 + t);
		else									/* If the quadratic equation has no solution,
												   or the solutions are out of the trinagle */
			return MAX_DISTANCE;
	}



}

void FastMarchingAlgorithm::TUpdateNeighbors(int i, int becomes_alive){
	double          u, t3, u0;
	int             n = heap->BP[i];
	int             k, l;
	int				st_ind;			/* Index for the stencils.				*/
	 Stencil	*st_ptr;		/* Pointer to the current stencil.		*/


	u = V[i].U;
	u0 = u;
	/* Do for every stencil.						*/
	for (st_ptr = V[i].ST, st_ind = V[i].si; st_ind > 0; st_ind--, st_ptr++) {
		/* check all numerical connections (triangles)	*/
		k = st_ptr->v1;
		l = st_ptr->v2;
		if (k == becomes_alive || l == becomes_alive) {	/* Use this stencil to update only if one k or
														   l is the new alive vertex.					*/
			if (k == becomes_alive)						/* Do the one dimensional update.				*/
				u = MIN(u, V[k].U + st_ptr->l1);
			else
				u = MIN(u, V[l].U + st_ptr->l2);
			if (heap->BP[k] == ALIVE || heap->BP[l] == ALIVE) {		/* Do update only if the two other vertexes
														   of the stencil are alive, otherwise this
														   stencil will be called again anyway.		*/
				t3 = update(st_ptr, k, l);

				FAST_MIN(u, t3);						/* Minimize the distance.			*/
			}


//						if (u0 > u){
//							V[i].source_num = MAX(V[l].source_num, V[k].source_num);
//						}

		}
	}

	//	if (V[i].source) { u = V[i].U; }

	if (n == FAR) {		/* not in heap                 */
		heap->insert(u, i);	/* also changes BP to Trail	   */
		V[i].U = u;
	}
	else {				/* change position in the heap */
		if (heap->a[n].u > u) {
			heap->a[n].u = V[i].U = u;
			heap->upheap(n);
		}
	}
}

void FastMarchingAlgorithm::CallTUpdateNeighbors(int i, int start_from) {
	int local_vn_count;
	int *VN_ind;
	for (VN_ind = V[i].VN, local_vn_count = 0; local_vn_count < V[i].vn; VN_ind++, local_vn_count++){
		if (heap->BP[*VN_ind] is_not ALIVE) {
			TUpdateNeighbors(*VN_ind, i);
		}
    }
}

void FastMarchingAlgorithm::MarchAlg(Vertex *temp_V, int temp_Vnum, double *srcval)
{
	V = temp_V;
	Vnum = temp_Vnum;

	InitMarch(srcval, Vnum);
	while (heap->N != 0) {
		CallTUpdateNeighbors(heap->a[1].v, 0);
		heap->remove_top();
	}
}

#endif