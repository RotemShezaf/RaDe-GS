#ifndef HEAP
#define HEAP
#include "globals.hpp"

struct heap_element { 		/* element of the heap													*/
	double		u;			/* element value (u)													*/
	int 		v; 			/* back pointer to the vertex (the index of the vertex in the array V)	*/
};

class Heap {

public:
    void init_heap(int Vnum);
	explicit Heap(int Vnum);
	~Heap() { free_heap(); }

	void reset();
	void upheap(int k);
	void insert(double u, int v);
	void downheap(int k);
	void remove_top();

protected:

	void free_heap(void);

public:

	int			*BP;			/* back pointer to place in the list */
	int			N;				/* number of elements in the heap array */
	heap_element *a; 			/* heap array */

};

void Heap::init_heap(int Vnum) {
	/* The heap size is at most the number of vertices Vnum, allocate the
	   heap array and the back pointer array.							   */
	a = (heap_element *)malloc(sizeof(heap_element) *(Vnum + 1));
	BP = (int *)malloc(sizeof(int) *Vnum);

	a[0].u = -MAX_DISTANCE;			/* anchor the root of the heap		   */
	N = 0;							/* Originaly the heap is empty.		   */
}

void Heap::free_heap(void) {
	free(a);
	free(BP);
}

void Heap::reset(void) {
	a[0].u = -MAX_DISTANCE;			/* anchor the root of the heap		   */
	N = 0;							/* Originaly the heap is empty.		   */
}

void Heap::upheap(int k)
{
	double          u = a[k].u;	/* current value */
	int             v = a[k].v;

	while (a[k >> 1].u > u) {	/* While the place was not found move the vertex up */
		a[k] = a[k >> 1];
		BP[a[k].v] = k;	        /* back pointer */
		k >>= 1;
	}
	a[k].u = u;
	a[k].v = v;
	BP[a[k].v] = k;	/* back pointer */
}

void Heap::insert(double u, int v)
{
	N++;
	a[N].u = u;
	a[N].v = v;
	BP[v] = N;		/* back pointer */
	upheap(N);
}

void Heap::downheap(int k)
/* k index to the element to be moved down the
					   heap */
{
	double          u = a[k].u;	/* current value */
	int             v = a[k].v;
	int             j = k;				/* Moves down the heap, till a place for vertex v is found. */

	while ((j <<= 1) <= N) {			/* While have not reached the button and cont is still true */
		if (j < N && a[j].u > a[j + 1].u) /* If the brother of j is smaller, move j to the brother.*/
			j++;
		if (u <= a[j].u)				/* If vertex k is smaller than vertex j end search.			*/
			break;
		else {
			a[k] = a[j];				/* Move vertex k one plece ower.							*/
			BP[a[k].v] = k;				/* update back pointer */
			k = j;
		}
	}
	a[k].u = u;							/* We have found a place for v, place the vertex here.		*/
	a[k].v = v;
	BP[a[k].v] = k;	/* back pointer */
}

void Heap::remove_top()
{
	BP[a[1].v] = ALIVE;	/* set back pointer to Alive (not in list)  */
	a[1] = a[N];		/* Move the last vertex to be the first     */
	BP[a[1].v] = 1;		/* back pointer								*/
	N--;
	downheap(1);		/* Update the heap structure.				*/
}

Heap::Heap(int Vnum) {
    init_heap(Vnum);
}


#endif