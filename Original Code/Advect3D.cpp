// Parallel 3D Advection Class
//   based on codes by Brendan Harding
// written by Peter Strazdins, May 13

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Advect3D.h"
#include "debug.h"


Advect3D::Advect3D(Vec3D<int> gridSz, Vec3D<double> v, double timef, 
		   double CFL, int verb, Timer *t, MPI_Comm myCW, int blk) {
  B = blk;  
  myCommWorld = myCW; 
  gridSize = gridSz;
  V = v; 
  delta.x = 1.0 / (gridSize.x - 1);  // implicitly on unit square    
  delta.y = 1.0 / (gridSize.y - 1); 
  delta.z = is2D()? 1.0: 1.0 / (gridSize.z - 1); 
  tf = timef;
  dt = CFL * std::min<double>(delta.x, std::min<double>(delta.y, delta.z));
  verbosity = verb;
  timer = t;
} //Advect3D()

  
void Advect3D::initGrid(HaloArray3D *u, ProcGrid3D *g) {
#pragma omp parallel for default(shared)
  for (int kj = 0; kj < u->l.z*u->l.y; kj++) {
    int k = kj / u->l.y, j = kj % u->l.y;
    double z = delta.z * (k + g->L2G0(2, gridSize.z));
    double y = delta.y * (j + g->L2G0(1, gridSize.y));
    for (int i=0; i < u->l.x; i++) {
      double x = delta.x  * (i + g->L2G0(0, gridSize.x)*B);
      Vh(u, i, j, k) = initialCondition(x, y, z, 0.0, V.x, V.y, V.z);
//      debug_out("initGrid: V(u, %d, %d, %d)\n", i+u->halo.x, j+u->halo.y, k+u->halo.z);
    } 
  }
}

double Advect3D::checkError(double t, HaloArray3D *u, ProcGrid3D *g) {
  double err = 0.0;
#pragma omp parallel for default(shared) reduction(+:err)
  for (int kj = 0; kj < u->l.z*u->l.y; kj++) {
    int k = kj / u->l.y, j = kj % u->l.y;
    double z = delta.z * (k + g->L2G0(2, gridSize.z));
    double y = delta.y * (j + g->L2G0(1, gridSize.y));
    for (int i=0; i < u->l.x; i++) {
      double x = delta.x  * (i + g->L2G0(0, gridSize.x)*B);
      err += std::abs(Vh(u, i, j, k) - initialCondition(x, y, z, t, 
							V.x, V.y, V.z));
    } 
  }
  return (err);
}

double Advect3D::checkMaxError(double t, HaloArray3D *u, ProcGrid3D *g) {
  double err = 0.0;
#pragma omp parallel for default(shared) reduction(max:err)
  for (int kj = 0; kj < u->l.z*u->l.y; kj++) {
    int k = kj / u->l.y, j = kj % u->l.y;
    double z = delta.z * (k + g->L2G0(2, gridSize.z));
    double y = delta.y * (j + g->L2G0(1, gridSize.y));
    for (int i=0; i < u->l.x; i++) {
      double x = delta.x  * (i + g->L2G0(0, gridSize.x)*B);
      double e = std::abs(Vh(u, i, j, k) - initialCondition(x, y, z, t, 
							    V.x, V.y, V.z));
      if (e > err) err = e; 
    } 
  }
  return (err);
}


void Advect3D::updateLW(HaloArray3D *u) {
  timer->start("updateLW", u->l.prod(), 1);
  if (!is2D())
    assert (false);
  else {
    extern bool opt_tmr;
    if ( opt_tmr ) updateLW2D_tmr(u);
    else updateLW2D(u);
  }
  timer->stop("updateLW");
} //updateLW() 


// values at grid rows/columns 0 and N-1 are indentical; therefore  
// boundaries must get their value the next innermost opposite row/column
void Advect3D::updateBoundary(HaloArray3D *u, ProcGrid3D *g) {
// Original: [
// assert(u->halo.x == 1  &&  u->halo.y == 1 && u->halo.z <= 1) ; //]

// Brian: [ halo can be 1 to 3 
  assert( u->halo.x > 0  &&  u->halo.x <= 3 
       && u->halo.y > 0 && u->halo.y <= 3 
       && u->halo.z <= 1 ) ; //]

  int lx = u->l.x, ly = u->l.y, lz = u->l.z, sx = u->s.x, sy = u->s.y;
  int hx = u->halo.x, hy = u->halo.y, hz = u->halo.z; 

  timer->start("updateBoundary", hz*lx*ly + ly*lz + lz*lx, 1);
  double *bufS, *bufR, *b; int i, j, k;
  MPI_Request req; MPI_Status stat;

  //printf("P[%d] lx: %d, ly: %d\n", g->myrank, lx, ly);
  if (g->P.x == 1) {  
    for (k=hz; k < lz+hz; k++) {
      for (j=hy; j < ly+hy; j++) { 
        for (int ib = 0; ib < B*hx; ib++) { 
	  V(u, ib, j, k) = V(u, lx+ib-B, j, k);
        } //_for(ib)
      } 

      for (j=hy; j < ly+hy; j++) {
        for (int ib = 0; ib < B*hx; ib++) {  
	  V(u, lx+hx+ib, j, k) = V(u, hx+ib+B, j, k);
        } 
      } 
    } //_for(k)
  } else {
//    bufS = new double[ly*lz*B]; bufR = new double[ly*lz*B];
    bufS = new double[hx*ly*lz*B]; bufR = new double[hx*ly*lz*B];

    int xOffs = (g->id.x == g->P.x-1) ? u->l.x-B: u->l.x;

    for( k=hz, b=bufS; k < lz+hz; k++) {
      for( j=hy; j<ly+hy;j++) {
        for( int ib = 0; ib < B*hx; ib++, b++) { 
  	        *b = V(u, xOffs+ib, j, k);
            //printf("RS[%d](%d, %d, %d)\n", g->myrank, xOffs+ib, j, k);
          }
        }
    }
    //printf("%d: comm right boundary %dx%d to %d\n",g->myrank, ly, lz, g->neighbour(+1,0));

    MPI_Isend(bufS, ly*lz*B*hx, MPI_DOUBLE, g->neighbour(+1, 0), 0, g->comm, &req);
    MPI_Recv(bufR, ly*lz*B*hx, MPI_DOUBLE, g->neighbour(-1, 0), 0, g->comm, &stat);

    // update u from bufR
    for (k=hz, b=bufR; k < lz+hz; k++) {
      for (j=hy; j < ly+hy; j++) {
         for (int ib = 0; ib < B*hx; ib++, b++) { 
	       V(u, 0+ib, j, k) = *b;
           //printf("RR[%d](%d, %d, %d)\n", g->myrank, ib, j, k);
         }
      }
    }

    MPI_Wait(&req, &stat);
    
    // fill up bufS for left boundary    
//    xOffs = (g->id.x == 0) ? 2*B: 1*B;
    xOffs = (g->id.x == 0) ? B*hx+B: B*hx;
    for (k=hz, b=bufS; k < lz+hz; k++) {
      for (j=hy; j < ly+hy; j++) {
         for (int ib = 0; ib < B * hx; ib++, b++) {  
	       *b = V(u, xOffs+ib, j, k);
           //printf("LS[%d](%d, %d, %d) with %d\n", g->myrank, xOffs+ib, j, k, xOffs);
         }
      }
    }

    //printf("%d: comm left boundary %dx%d to %d\n",g->myrank, ly, lz, g->neighbour(-1,0));
    MPI_Isend(bufS, ly*lz*hx*B, MPI_DOUBLE, g->neighbour(-1, 0), 0, g->comm, &req);
    MPI_Recv(bufR, ly*lz*hx*B, MPI_DOUBLE, g->neighbour(+1, 0), 0, g->comm, &stat);

    for (k=hz, b=bufR; k < lz+hz; k++) {
      for (j=hy; j < ly+hy; j++) {
        for (int ib = 0; ib < B * hx; ib++, b++) { 
	      V(u, lx+hx+ib, j, k) = *b;
            //printf("LR[%d](%d, %d, %d) with %d\n", g->myrank, lx+hx+ib, j, k, xOffs);
        }
      }
    }
    MPI_Wait(&req, &stat);

    delete[] bufS; delete[] bufR;
  }
   
  if (g->P.y == 1) {
    for (k=hz; k < lz+hz; k++) {
      //printf("Sxy[%d]( %d, %d )\n", g->myrank, sx, sy);
      for (j=0; j<hy; j++){
        //printf( "Bdy1(%d): V(u, %d<%d, %d, %d) = V(u, %d<%d, %d, %d)\n",
                        //g->myrank, 0,sx*B, j, k,      0,sx*B, ly+j-1, k);
        for (i=0; i < sx*B; i++) {
          V(u, i, j, k)    = V(u, i, ly+j-1, k);
        }//_for(i)
      }//_for(j)

      for (j=0;j<hy;j++){
        //printf( "Bdy2(%d): V(u, %d<%d, %d, %d) = V(u, %d<%d, %d, %d)\n",
                  //g->myrank, 0, sx*B, ly+hy+j, k,     0, sx*B,hy+j+1, k);
        for (i=0; i < sx*B; i++){
          V(u, i, ly+hy+j, k) = V(u, i, hy+j+1, k);
        }//_for(i)
      }//_for(j) //]_Brian

    } //_for(k) 
  } else {
    bufS = new double[sx*lz*hy]; bufR = new double[sx*lz*hy]; 

    int yOffs = (g->id.y == g->P.y-1) ? ly-1: ly;
    //printf("P[%d] sx(%d) yOffs:ly(%d:%d) \n", g->myrank, sx, yOffs, ly);

    for (k=hz, b=bufS; k < lz+hz; k++)
      for (j=0; j < hy; j++){
        //printf("P[%d] BS(%d<%d, %d/%d)\n", g->myrank, 0,sx, j+yOffs, ly);
        for (i=0; i < sx; i++, b++) 
  	      *b = V(u, i, j+yOffs, k);
      }

    //printf("%d: comm bottom boundary to %d\n",g->myrank,g->neighbour(+1,1));
     MPI_Isend(bufS, sx*lz*hy, MPI_DOUBLE, g->neighbour(+1, 1), 0, g->comm, &req);
     MPI_Recv(bufR, sx*lz*hy, MPI_DOUBLE, g->neighbour(-1, 1), 0, g->comm, &stat);

     for (k=hz, b=bufR; k < lz+hz; k++)
       for (j=0; j < hy; j++){
         //printf("P[%d] BR(%d<%d, %d/%d)\n", g->myrank, 0,sx, j, ly);
         for (i=0; i < sx; i++, b++)
	       V(u, i, j, k) = *b;
       }

     MPI_Wait(&req, &stat);
        
     yOffs = (g->id.y == 0) ? hy+1: hy;
     for (k=hz, b=bufS; k < lz+hz; k++)
       for (j=0; j < hy; j++){
        //printf("P[%d] TS(%d<%d, %d/%d)\n", g->myrank, 0,sx, j+yOffs, ly);
         for (i=0; i < sx; i++, b++) 
	       *b = V(u, i, j+yOffs, k);
       }

     //printf("%d: comm top boundary to %d\n", g->myrank, g->neighbour(-1, 1));
     MPI_Isend(bufS, sx*lz*hy, MPI_DOUBLE, g->neighbour(-1, 1), 0, g->comm, &req);
     MPI_Recv(bufR, sx*lz*hy, MPI_DOUBLE, g->neighbour(+1, 1), 0, g->comm, &stat);

     for (k=hz, b=bufR; k < lz+hz; k++)
       for (j=0; j < hy; j++){
         //printf("P[%d] TR(%d<%d, %d/%d)\n", g->myrank, 0,sx, j+ly+hy, ly);
         for (i=0; i < sx; i++, b++)
           V(u, i, j+ly+hy, k) = *b;
       }

     MPI_Wait(&req, &stat);

     delete[] bufS; delete[] bufR;
  }

  if (hz == 0) { // no halo in z dimension
    timer->stop("updateBoundary");
    return;
  }

  if (g->P.z == 1) {
    for (j=0; j < sy; j++) {
      for (i=0; i < sx; i++)
	V(u, i, j, 0)    = V(u, i, j, lz-1);
      for (i=0; i < sx; i++)
	V(u, i, j, lz+1) = V(u, i, j, 2);
    }
  } else {
    bufS = new double[sx*sy]; bufR = new double[sx*sy]; 

    int zOffs = (g->id.z == g->P.z-1) ? lz-1: lz;
    for (j=0, b=bufS; j < sy; j++)
      for (i=0; i < sx; i++, b++) 
	*b = V(u, i, j, zOffs);
    //printf("%d: comm bottom boundary %dx%d to %d\n",g->myrank,sx, sy, g->neighbour(+1,1));
     MPI_Isend(bufS, sx*sy, MPI_DOUBLE, g->neighbour(+1, 2), 0, g->comm, &req);
     MPI_Recv(bufR, sx*sy, MPI_DOUBLE, g->neighbour(-1, 2), 0, g->comm, &stat);
     for (j=0, b=bufR; j < sy; j++)
       for (i=0; i < sx; i++, b++)
	 V(u, i, j, 0) = *b;
     MPI_Wait(&req, &stat);
        
     zOffs = (g->id.z == 0) ? 2: 1;
     for (j=0, b=bufS; j < sy; j++)
       for (i=0; i < sx; i++, b++) 
	 *b = V(u, i, j, zOffs);
     //printf("%d: comm top boundary %dx%d to %d\n", g->myrank, sx, sy, g->neighbour(-1, 1));
     MPI_Isend(bufS, sx*sy, MPI_DOUBLE, g->neighbour(-1, 2), 0, g->comm, &req);
     MPI_Recv(bufR, sx*sy, MPI_DOUBLE, g->neighbour(+1, 2), 0, g->comm, &stat);
     for (j=0, b=bufR; j < sy; j++)
       for (i=0; i < sx; i++, b++)
	 V(u, i, j, lz+1) = *b;
     MPI_Wait(&req, &stat);

     delete[] bufS; delete[] bufR;
  }
  timer->stop("updateBoundary");
} //updateBoundary()

double Advect3D::simulateAdvection(HaloArray3D *u, ProcGrid3D *g, double dtA) {

  timer->start("simulateAdvection", u->l.prod(), 0);

  double t = 0.0;  
  int s = 0;
  struct timeval ts, tf; // start time, final time

  if (is2D()) // check that u, g have been set up appropriately
    assert (u->halo.z == 0  &&  u->l.z <= 1  &&  g->P.z == 1);

  if (g->myrank == 0) 
    gettimeofday(&ts, NULL);

  while (t < dtA) {
    extern bool opt_warming;
    if (s==1 && g->myrank == 0 && opt_warming) 
      gettimeofday(&ts, NULL);

    updateLW(u);

    updateBoundary(u, g);

    t += dt; s++;
    if (verbosity > 3) {
      char s[64];
      sprintf(s, "after time %.4e, field is:\n", t);
      u->print(g->myrank, s);
    }

    if (t >= dtA && g->myrank == 0) { 
      gettimeofday(&tf, NULL);
      printf("Elapsed time: %.2f secs\n", 
           ( tf.tv_sec*1e6 + tf.tv_usec 
           - ts.tv_sec*1e6 - ts.tv_usec) / 1e6);
    }

    // Exit if -i is specificed. Only for debugging
    extern bool opt_oneshot;
    if ( opt_oneshot ) exit(1);
  }
  

  timer->stop("simulateAdvection");

  return t;
} //simulateAdvection()


