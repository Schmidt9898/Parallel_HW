#include <mpi.h>

extern int my_rank;
extern int prev_y;
extern int next_y;
extern int next_x;
extern int prev_x;

extern int xmax_full;
extern int ymax_full;

extern int gbl_x_begin;
extern int gbl_y_begin;
extern int coords[2];
extern int dims[2];


void MPISetup(int *xmax, int *ymax);
void exchangeHalo(unsigned xmax, unsigned ymax, double *arr);