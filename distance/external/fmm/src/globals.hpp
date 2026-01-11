#ifndef GLOBALS
#define GLOBALS
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// Aliases
#define mod 	%
#define is_not 	!=

// Enums
#define ALIVE 0		/* Alive and Far are the values that are placed in the Back Pointer array when the vertex is not on the heap.*/
#define FAR   -1			

// Constants
#define PI	3.141592654
#define MAX_DISTANCE  	999
#define MAX_TRI	 100	/* The maximum number of triangles that a vertex can participate in. */

// Operator Shorts
#define SQR(x)  ((x)*(x))
#define SQRT(x) (sqrt(x))
#define ABS(x) 		((x)>0?(x):(-(x)))
#define MAX(x,y) 	((x)>(y)?(x):(y))
#define MIN(x,y) 	((x)>(y)?(y):(x))
#define FAST_MIN(x,y) (((x) > (y)) ? (x) = (y) : 0)


struct Triangle { 	/* defines one triangle					*/
	int	Vind[3];	/* Index for the 3 vertices				*/
	double  b[5];   /* surface cooefficients				*/
					/* Du = (2b0x+b2y+b3,2b1y+b2x+b4)		*/
					/* for Vind[0]-Vind[1] = x exis			*/
	bool Visited;/* Determine whether this vertex was
					   already visited when backtracking to
					   find a smooth geodesic.				*/
	bool Split;	/* This field insdicates whether on of
					   the edges of this triangle is
					   splitting edge.						*/
};

struct Stencil {		/* virtual numerical connections  */
				/*   true triangles in most cases */
	double	Ctheta;		/* cos(theta) angle between the edges */
	double  Stheta; /* sin(theta)^2 when theta is the angle between the edges */
	int	v1, v2;		/* index of neigboring vertices */
	double	l1, l2;		/* edge length to v1 and to v2 */
	double Ctheta_mul_l1_div_l2_minus_one_mul_2, Ctheta_mul_l2_div_l1_minus_one_mul_2;
	/* Remember the value of the l1 and l2 multiplies
				by Ctheta minus one	multiplied by 2	*/
	double sqr_l1, sqr_l2; /* The values of l1 and l2 multiplied by themselves. */
						/* Replaces 1 - a2 * (ctheta_mul_mb_div_ma_minus_one_mul_2 + 1) / b2 */
	double shortcut1_1, shortcut1_2;
	/* Replaces - Stheta * a2*/
	double shortcut2_1, shortcut2_2;

};

struct Vertex { 		/* defines one Numerical Graph vertex */
	double	x, y, z;		/*x,y,z coordinates   */
	double	U;	   	/* U vlaue	*/
	int	Tind[MAX_TRI];	/* link back to triangles */
					/* MaxTri = Max # triangles at one vertex */
	int	si, vn;		/* number of vertex connections */
					/* si is the #of ST, vn is the index to VN*/
	int ti;			/* The number of triangles this vertex is member of */
	Stencil	ST[MAX_TRI];	/* numerical connection,
					  updating stenciles  */
	int	VN[3 * MAX_TRI]; /* Neighboring dirrectional vertices indexes*/
				/* Vertex to update */
	int	ST_For_VN[3 * MAX_TRI]; /* The first stenciel of the neighbour with this vertex in it */

	int Split[2];	/* The indexes of the vertexes that split the triangles of that vertex
					   or NO_DATA if there is no vertex.									*/

	bool source;	  /* Indicates wether the variable is a source in one of the marches
						 (meaning it appears in the xStart, yStart variables).				*/
	bool current_source;	/* Is it a source that got the U value 0 in the last run of
						 the fast march algoritem.											*/
	double MaxU;
};

struct Point {						/* defines a point in 3D with a U value					*/
	double  x, y, z;					/* x,y,z coordinates IN 3D								*/
	double  U;						/* The U value of the point.							*/
};

struct Vector {						/* defines a vector in 3D								*/
	double  x, y, z;				/* x,y,z coordinates of the vector in 3D.				*/
};

inline double L2(Vertex *V, int Vnum)
{
	int             i;
	double          d, x, y, z;
	double          sum = 0, dxdy = 1.0 / ((double)Vnum);
	/* find the zero point */
	for (i = 0; i < Vnum; i++)
		if (V[i].U < 0.00000001) {
			x = V[i].x;
			y = V[i].y;
			z = V[i].z;
			//fprintf ("Source at: (%g,%g,%g)\n",x,y,z);
		}
	for (i = 0; i < Vnum; i++) {
		d = sqrt(SQR(x - V[i].x) + SQR(y - V[i].y) + SQR(z - V[i].z));
		sum += SQR(V[i].U - d)*dxdy;
	}
	return (sqrt(sum));
}
/***************************************************************************/
/* L1  error norm the diff between the distance and U*/
/***************************************************************************/
inline double L1(Vertex *V, int Vnum)
{
	int             i;
	double          d, x, y, z;
	double          sum = 0.0, dxdy = 1.0 / ((double)Vnum);
	/* find the zero point */
	for (i = 0; i < Vnum; i++)
		if (V[i].U < 0.00000001) {
			x = V[i].x;
			y = V[i].y;
			z = V[i].z;
		}
	for (i = 0; i < Vnum; i++) {
		d = sqrt(SQR(x - V[i].x) + SQR(y - V[i].y) + SQR(z - V[i].z));
		sum += ABS(V[i].U - d)*dxdy;
	}
	return (sum);
}

/***************************************************************************/
/* CosAngle the cos of the angle at the vertex v0, between v1 and v2	   */
/***************************************************************************/
inline double CosAngle(int v0, int v1, int v2, Vertex *V)
{
	double x1, x2, y1, y2, z1, z2, res;
	if (v0 != -1 && v1 != -1 && v2 != -1) {
		x1 = V[v1].x - V[v0].x;
		x2 = V[v2].x - V[v0].x;
		y1 = V[v1].y - V[v0].y;
		y2 = V[v2].y - V[v0].y;
		z1 = V[v1].z - V[v0].z;
		z2 = V[v2].z - V[v0].z;
		res = x1 * x2 + y1 * y2 + z1 * z2;		/* dot product */
		res /= sqrt(x1*x1 + y1 * y1 + z1 * z1); /* normalize */
		res /= sqrt(x2*x2 + y2 * y2 + z2 * z2);
		return(res);
	}
	else
		return 0;
}
/***************************************************************************/
/* Length between the vertex v0 and v1									   */
/***************************************************************************/
double
inline Length(int v0, int v1, Vertex *V)
{
	double x1, y1, z1, res;
	if (v0 != -1 && v1 != -1) {
		x1 = V[v1].x - V[v0].x;
		y1 = V[v1].y - V[v0].y;
		z1 = V[v1].z - V[v0].z;
		res = sqrt(x1*x1 + y1 * y1 + z1 * z1); /* distance */
		return(res);
	}
	else
		return MAX_DISTANCE;
}
/***************************************************************************/
/* nextT next triangle to be unfolded. find the triangle #, and the vertex */
/* Returns true is the next triangle was found.							   */
/* v1 and v2 indicate the edge that is common to the original triangle	   */
/* and the triangle to be unfolded.										   */
/* ti is the original triangle and v3 is the other vertex of the triangle  */
/* to be unfolded.														   */
/* vn is the index of the triangle to be unfolded.						   */
/***************************************************************************/
inline bool nextT(int ti, int v1, int v2, int *v3, int *tn,
	Triangle * T, Vertex * V, int Tnum)
{
	bool				found = false;	/* Indicates whether we found the next triangle. */
	int					i,				/* Index for the loop.							 */
		tj,				/* A candidate tp be the next triangle.			 */
		k;				/* Index for the inner loop.					 */
/* scan every triangle of vi */
	for (i = 0; i < V[v1].ti && !found; i++) {
		tj = V[v1].Tind[i];
		if (tj < Tnum && tj != ti)
			/* search for tj in the list of v2 */
			for (k = 0; k < V[v2].ti && !found; k++)
				if (V[v2].Tind[k] == tj && !T[tj].Split) {
					found = true;
					*tn = tj;
				}
	}
	if (found) { /* find v3, the other vertex in the triangle to be unfolded.			 */
		if (T[*tn].Vind[0] == v1) {
			if (T[*tn].Vind[1] == v2)
				*v3 = T[*tn].Vind[2];
			else
				*v3 = T[*tn].Vind[1];
		}
		else if (T[*tn].Vind[1] == v1) {
			if (T[*tn].Vind[0] == v2)
				*v3 = T[*tn].Vind[2];
			else
				*v3 = T[*tn].Vind[0];
		}
		else {
			if (T[*tn].Vind[0] == v2)
				*v3 = T[*tn].Vind[1];
			else
				*v3 = T[*tn].Vind[0];
		}
	}
	return (found);
}

/***************************************************************************/
/* Split obtuse angles by unfolding splitting and connecting, return the   */
/* number of unfoldings that were nessesery.							   */
/* ti is the tirnalge to be splittined, V0 is the vertex with the obtuse   */
/* angle while V1 and V2 are the other vertexes of ti.					   */
/***************************************************************************/
inline int Split(int ti, int V0, int V1, int V2, Triangle * T, Vertex * V,
	int NonSplitTnum, int Vnum)
{
	double          xv1, x1, y1, yv1,
		x2,				/* The distance between V0 and V2 */
		y2,
		x3, y3, xt2, xt3, yt3,
		e0,				/* The distance between v1 and v2 */
		e1,				/* The distance between v2 and v3.*/
		e2,				/* The distance between v0 and v1 */
		ta,				/* Tan of alpha.				  */
		cb, sb;			/* Cos and Sin of beta.			  */
	int             v1 = V1, v2 = V2, v3,
		tm = ti,			/* The current triangle we are
						   working on.					  */
		tn,				/* The triangle returned by NextT */
		count = 0;		/* The number of triangles unfolded
						   so far.						  */
						   /* Becomes true when the split was done */
	bool         splitter = false;
	x2 = Length(V0, V2, V);
	y2 = 0;
	e0 = Length(V1, V2, V);
	e2 = Length(V0, V1, V);
	xv1 = x1 = (x2 * x2 + e2 * e2 - e0 * e0) / (2.0 * x2);/* translation */
	yv1 = y1 = sqrt(e2 * e2 - x1 * x1);
	ta = -x1 / y1;		/* tan (alpha) in Fig. 1 */
	/* if there is a next triangle and not splited */
	while (nextT(tm, v1, v2, &v3, &tn, T, V, NonSplitTnum) && (!splitter)) {
		count++;
		tm = tn;		/* Update the wording triangle. */
		cb = (x2 - x1) / sqrt(SQR(x2 - x1) + SQR(y2 - y1));	/* cos beta */
		sb = sqrt(1 - cb * cb);								/* sin beta */
		if (y2 < y1)	/* Adjast the sign of SIN(beta).				*/
			sb *= -1.0;
		xt2 = Length(v1, v2, V);
		e1 = Length(v2, v3, V);
		e2 = Length(v1, v3, V);
		xt3 = (xt2 * xt2 + e2 * e2 - e1 * e1) / (2.0 * xt2);
		yt3 = sqrt(e2 * e2 - xt3 * xt3);
		x3 = cb * xt3 - sb * yt3 + x1;
		y3 = sb * xt3 + cb * yt3 + y1;

		if (x3 > 0 && y3 / x3 > ta) {		/* if we found a splitter */
			splitter = true;
			/* Add the stencils involving the
			   splitting edge.			*/
			V[V0].ST[V[V0].si].Ctheta = (x3*xv1 + y3 * yv1) /
				sqrt((xv1*xv1 + yv1 * yv1)*(x3*x3 + y3 * y3));
			V[V0].ST[V[V0].si].v1 = V1;
			V[V0].ST[V[V0].si].v2 = v3;
			V[V0].ST[V[V0].si].l1 = Length(V0, V1, V);
			if (V[V0].si == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[V0].ST[V[V0].si++].l2 = sqrt(x3*x3 + y3 * y3);


			V[V0].ST[V[V0].si].Ctheta = x3 / sqrt(x3*x3 + y3 * y3);
			V[V0].ST[V[V0].si].v1 = v3;
			V[V0].ST[V[V0].si].v2 = V2;
			V[V0].ST[V[V0].si].l1 = sqrt(x3*x3 + y3 * y3);
			if (V[V0].si == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[V0].ST[V[V0].si++].l2 = Length(V0, V2, V);

			if (V[v3].vn == 3 * MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[v3].VN[V[v3].vn++] = V0; /* add dirrectional edge
											  to v3					*/

			T[NonSplitTnum + (ti * 2)].Vind[0] = V1;	/* Add the triangles of the splitting. */
			T[NonSplitTnum + (ti * 2)].Vind[1] = V0;
			T[NonSplitTnum + (ti * 2)].Vind[2] = v3;
			T[NonSplitTnum + (ti * 2)].Split = true;
			if (V[V1].ti == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[V1].Tind[V[V1].ti++] = NonSplitTnum + (ti * 2);
			if (V[V0].ti == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[V0].Tind[V[V0].ti++] = NonSplitTnum + (ti * 2);
			if (V[v3].ti == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[v3].Tind[V[v3].ti++] = NonSplitTnum + (ti * 2);
			T[NonSplitTnum + (ti * 2) + 1].Vind[0] = v3;
			T[NonSplitTnum + (ti * 2) + 1].Vind[1] = V0;
			T[NonSplitTnum + (ti * 2) + 1].Vind[2] = V2;
			T[NonSplitTnum + (ti * 2) + 1].Split = true;
			if (V[v3].ti == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[v3].Tind[V[v3].ti++] = NonSplitTnum + (ti * 2) + 1;
			if (V[V0].ti == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[V0].Tind[V[V0].ti++] = NonSplitTnum + (ti * 2) + 1;
			if (V[V2].ti == MAX_TRI - 1) {
				//	mexPrintf("Warning, too many triangles to one vertex, result quality will suffer\n");
			}
			else
				V[V2].Tind[V[V2].ti++] = NonSplitTnum + (ti * 2) + 1;


		}
		else {						   /* we have not found a splitter,
										  continue unfolding			*/
			if (x3 < 0) {
				v1 = v3; x1 = x3; y1 = y3;
			}
			else {
				v2 = v3; x2 = x3; y2 = y3;
			}
		}
	}
	return(count);				/* Return the number of triangles that were
								   unfolded.							   */
}

#endif

