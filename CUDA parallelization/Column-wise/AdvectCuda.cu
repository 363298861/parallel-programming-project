
#include <stdio.h>
#include <stdlib.h>
#include "AdvectCoeff.h"
#include "Advect3D.h"

#undef V
#undef Vh
#define V(u, i, j) (*((u) -> ix(i, j, 0)))
#define Vh(u, i, j) (*((u) -> ix_h(i, j, 0)))
#define VT(u, i, j, T) (*((u) -> ix(i, j, T)))
#define VhT(u, i, j, T) (*((u) -> ix_h(i, j, T)))
extern bool opt_tmr;
extern int gdim, bdim;

__device__ double median3cuda(double a, double b, double c){
    if ( a < b ) {
        if ( b < c ) return b;
        if ( a > c ) return a;
        else return c;
    }else { // b < a
        if ( a < c ) return a;
        else if ( b > c) return b;
        else return c;
    }
}

__global__ void boundary(HaloArray3D* u){
    int lx = u->l.x, ly = u->l.y, sx = u->s.x;
    int hx = u->halo.x, hy = u->halo.y;


    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;

    for(int j = j0; j < ly + hy; j+= dj){
        for(int ib = i0; ib < hx; ib += di) {
            V(u, ib, j) = V(u, lx + ib - u->B, j);
            V(u, lx + hx + ib, j) = V(u, hx + ib + u->B, j);
        }
    }

    for(int j = j0; j < hy; j += dj){
        for(int i = i0; i < sx; i += di){
            V(u, i, j) = V(u, i, ly + j - 1);
            V(u, i, ly + hy + j) = V(u, i, hy + j + 1);
        }
    }
}

__global__ void LW2DtmrKernel1(HaloArray3D* u){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    for(int j = j0; j < u->s.y; j += dj){
        for(int i = i0; i < u->s.x; i += di){
            VT(u, i, j, 1) = VT(u, i, j, 2) =  VT(u, i, j, 0);
        }
    }
}

__global__ void LW2DtmrKernel2(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy, int t){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    for(int j = j0; j < u->l.y; j += dj) {
        for (int i = i0; i < u->l.x; i += di) {
            V(uh, i, j) = Vy * (Vy - 1.0) * (Ux * (Ux - 1.0) * VhT(u, i + 1, j + 1, t) / 2
                                             + Ux * (Ux + 1.0) * VhT(u, i - 1, j + 1, t) / 2
                                             + (1.0 - Ux * Ux) * VhT(u, i, j + 1, t)) / 2
                          + Vy * (Vy + 1.0) * (Ux * (Ux - 1.0) * VhT(u, i + 1, j - 1, t) / 2
                                               + Ux * (Ux + 1.0) * VhT(u, i - 1, j - 1, t) / 2
                                               + (1.0 - Ux * Ux) * VhT(u, i, j - 1, t)) / 2
                          + (1.0 - Vy * Vy) * (Ux * (Ux - 1.0) * VhT(u, i + 1, j, t) / 2
                                               + Ux * (Ux + 1.0) * VhT(u, i - 1, j, t) / 2
                                               + (1.0 - Ux * Ux) * VhT(u, i, j, t));
        }
    }
}

__global__ void LW2DtmrKernel3(HaloArray3D* u, HaloArray3D* uh1, HaloArray3D* uh2, HaloArray3D* uh3){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    for(int j = j0; j < u->l.y; j += dj) {
        for (int i = i0; i < u->l.x; i += di) {
            VhT(u, i, j, 0) = median3cuda(Vh(uh1, i, j), Vh(uh2, i, j), Vh(uh3, i, j));
        }
    }
}

void Advect3D::updateLW2D_tmr_Cuda(HaloArray3D* u){
    HaloArray3D *uh[3];
    const dim3 gridSize(gdim, gdim, 1);
    const dim3 blockSize(bdim, bdim, 1);

    for(int t = 0; t < 3; t++){
        uh[t] = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    }
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    LW2DtmrKernel1<<<gridSize, blockSize>>>(u);
    cudaDeviceSynchronize();

    HaloArray3D* d_uh1, *d_uh2, *d_uh3;
    double* uuh1, *uuh2, *uuh3;
    cudaMallocManaged((void**) &d_uh1, sizeof(HaloArray3D));
    cudaMallocManaged((void**) &d_uh2, sizeof(HaloArray3D));
    cudaMallocManaged((void**) &d_uh3, sizeof(HaloArray3D));
    cudaMallocManaged((void**) &uuh1, sizeof(double) * uh[0]->s.prod() * 3);
    cudaMallocManaged((void**) &uuh2, sizeof(double) * uh[1]->s.prod() * 3);
    cudaMallocManaged((void**) &uuh3, sizeof(double) * uh[2]->s.prod() * 3);
    d_uh1->u = uuh1;
    d_uh1->l = uh[0]->l;
    d_uh1->s = uh[0]->s;
    d_uh1->halo = uh[0]->halo;
    d_uh1->B = uh[0]->B;

    d_uh2->u = uuh2;
    d_uh2->l = uh[1]->l;
    d_uh2->s = uh[1]->s;
    d_uh2->halo = uh[1]->halo;
    d_uh2->B = uh[1]->B;

    d_uh3->u = uuh3;
    d_uh3->l = uh[2]->l;
    d_uh3->s = uh[2]->s;
    d_uh3->halo = uh[2]->halo;
    d_uh3->B = uh[2]->B;


    LW2DtmrKernel2<<<gridSize, blockSize>>>(u, d_uh1, Ux, Vy, 0);
    cudaDeviceSynchronize();
    LW2DtmrKernel2<<<gridSize, blockSize>>>(u, d_uh2, Ux, Vy, 1);
    cudaDeviceSynchronize();
    LW2DtmrKernel2<<<gridSize, blockSize>>>(u, d_uh3, Ux, Vy, 2);
    cudaDeviceSynchronize();
    LW2DtmrKernel3<<<gridSize, blockSize>>>(u, d_uh1, d_uh2, d_uh3);
    cudaDeviceSynchronize();

    cudaFree(d_uh1);
    cudaFree(d_uh2);
    cudaFree(d_uh3);
    cudaFree(uuh1);
    cudaFree(uuh2);
    cudaFree(uuh3);
}


__global__ void LWN2kernel1(HaloArray3D* u, HaloArray3D* uh, double Ux, double Uy, int optimized){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            double cim1, ci0, cip1;
            double cjm1, cj0, cjp1;
            N2Coeff(Ux, cim1, ci0, cip1);
            N2Coeff(Uy, cjm1, cj0, cjp1);

            Vh(uh, i, j) = cim1 * (cjm1 * Vh(u, i - 1, j - 1) + cj0 * Vh(u, i - 1, j) + cjp1 * Vh(u, i - 1, j + 1)) +
                           ci0 * (cjm1 * Vh(u, i, j - 1) + cj0 * Vh(u, i, j) + cjp1 * Vh(u, i, j + 1)) +
                           cip1 * (cjm1 * Vh(u, i + 1, j - 1) + cj0 * Vh(u, i + 1, j) + cjp1 * Vh(u, i + 1, j + 1));
        }
    }
}

__global__ void LWN2kernel2(HaloArray3D* u, HaloArray3D* uh, int optimized){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            Vh(u, i, j) = Vh(uh, i, j);
        }
    }
}

void Advect3D::updateLWN2Cuda(HaloArray3D *u) {

    double Ux = V.x * dt / delta.x, Uy = V.y * dt / delta.y;

    HaloArray3D *uh = new HaloArray3D(u->l, Vec3D<int>(0), 1);
    int sizeuh = uh->s.prod();
    HaloArray3D* d_uh;
    double* uuh;
    cudaMallocManaged((void**) &d_uh, sizeof(HaloArray3D));
    cudaMallocManaged((void**) &uuh, sizeof(double) * sizeuh);
    d_uh->u = uuh;
    d_uh->l = uh->l;
    d_uh->s = uh->s;
    d_uh->halo = uh->halo;
    d_uh->B = uh->B;

    const dim3 gridSize(gdim, gdim, 1);
    const dim3 blockSize(bdim, bdim, 1);

    LWN2kernel1<<<gridSize, blockSize>>>(u, d_uh, Ux, Uy, 1);
    cudaDeviceSynchronize();
    LWN2kernel2<<<gridSize, blockSize>>>(u, d_uh, 1);
    cudaDeviceSynchronize();
    cudaFree(d_uh);
    cudaFree(uuh);
} //updateLWN2()

__global__ void LWN4kernel1(HaloArray3D* u, HaloArray3D* uh, double Ux, double Uy){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            double cim2, cim1, ci0, cip1, cip2;
            double cjm2, cjm1, cj0, cjp1, cjp2;
            N4Coeff(Ux, cim2, cim1, ci0, cip1, cip2);
            N4Coeff(Uy, cjm2, cjm1, cj0, cjp1, cjp2);
            Vh(uh, i, j) =
                    cim2 * (cjm2 * Vh(u, i - 2, j - 2) + cjm1 * Vh(u, i - 2, j - 1) + cj0 * Vh(u, i - 2, j) +
                            cjp1 * Vh(u, i - 2, j + 1) + cjp2 * Vh(u, i - 2, j + 2)) +
                    cim1 * (cjm2 * Vh(u, i - 1, j - 2) + cjm1 * Vh(u, i - 1, j - 1) + cj0 * Vh(u, i - 1, j) +
                            cjp1 * Vh(u, i - 1, j + 1) + cjp2 * Vh(u, i - 1, j + 2)) +
                    ci0 * (cjm2 * Vh(u, i, j - 2) + cjm1 * Vh(u, i, j - 1) + cj0 * Vh(u, i, j) +
                           cjp1 * Vh(u, i, j + 1) + cjp2 * Vh(u, i, j + 2)) +
                    cip1 * (cjm2 * Vh(u, i + 1, j - 2) + cjm1 * Vh(u, i + 1, j - 1) + cj0 * Vh(u, i + 1, j) +
                            cjp1 * Vh(u, i + 1, j + 1) + cjp2 * Vh(u, i + 1, j + 2)) +
                    cip2 * (cjm2 * Vh(u, i + 2, j - 2) + cjm1 * Vh(u, i + 2, j - 1) + cj0 * Vh(u, i + 2, j) +
                            cjp1 * Vh(u, i + 2, j + 1) + cjp2 * Vh(u, i + 2, j + 2));
        }
    }
}

__global__ void LWN4kernel2(HaloArray3D* u, HaloArray3D* uh){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            Vh(u, i, j) = Vh(uh, i, j);
        }
    }
}

void Advect3D::updateLWN4Cuda(HaloArray3D *u){
    const dim3 gridSize(gdim, gdim, 1);
    const dim3 blockSize(bdim, bdim, 1);
    double Ux = V.x * dt / delta.x, Uy = V.y * dt / delta.y;
    HaloArray3D *uh = new HaloArray3D(u->l, Vec3D<int>(0), 1);
    int sizeuh = uh->s.prod();
    HaloArray3D* d_uh;
    double* uuh;
    cudaMallocManaged((void**) &d_uh, sizeof(HaloArray3D));
    cudaMallocManaged((void**) &uuh, sizeof(double) * sizeuh);
    d_uh->u = uuh;
    d_uh->l = uh->l;
    d_uh->s = uh->s;
    d_uh->halo = uh->halo;
    d_uh->B = uh->B;

    LWN4kernel1<<<gridSize, blockSize>>>(u, d_uh, Ux, Uy);
    cudaDeviceSynchronize();
    LWN4kernel2<<<gridSize, blockSize>>>(u, d_uh);
    cudaDeviceSynchronize();
    cudaFree(d_uh);
    cudaFree(uuh);
}

__global__ void NN2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = Vy * (Vy - 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 1) / 2
                                             + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 1) / 2
                                             + (1.0 - Ux * Ux) * Vh(u, i, j + 1)) / 2
                          + Vy * (Vy + 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 1) / 2
                                               + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 1) / 2
                                               + (1.0 - Ux * Ux) * Vh(u, i, j - 1)) / 2
                          + (1.0 - Vy * Vy) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j) / 2
                                               + Ux * (Ux + 1.0) * Vh(u, i - 1, j) / 2
                                               + (1.0 - Ux * Ux) * Vh(u, i, j));
        }
    }
}

__global__ void NW2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = Vy * (Vy / 2 - 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 2) / 2
                                                 + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 2) / 2
                                                 + (1.0 - Ux * Ux) * Vh(u, i, j + 2)) / 4
                          + Vy * (Vy / 2 + 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 2) / 2
                                                   + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 2) / 2
                                                   + (1.0 - Ux * Ux) * Vh(u, i, j - 2)) / 4
                          + (-Vy * Vy / 4 + 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j) / 2
                                                    + Ux * (Ux + 1.0) * Vh(u, i - 1, j) / 2
                                                    + (1.0 - Ux * Ux) * Vh(u, i, j));
        }
    }
}

__global__ void NF2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = (-Vy * Vy / 16 + 9.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 1) / 2
                                                        + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 1) / 2
                                                        + (1.0 - Ux * Ux) * Vh(u, i, j - 1))
                          + (-Vy * Vy / 16 + 9.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 1) / 2
                                                          + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 1) / 2
                                                          + (1.0 - Ux * Ux) * Vh(u, i, j + 1))
                          + (Vy * Vy / 16 - Vy / 6 - 1.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 3) / 2
                                                                  + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 3) / 2
                                                                  + (1.0 - Ux * Ux) * Vh(u, i, j + 3))
                          + (Vy * Vy / 16 + Vy / 6 - 1.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 3) / 2
                                                                  + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 3) / 2
                                                                  + (1.0 - Ux * Ux) * Vh(u, i, j - 3));
        }
    }
}

__global__ void WN2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = Vy * (Vy - 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 1) / 4
                                             + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 1) / 4
                                             + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 1)) / 2
                          + Vy * (Vy + 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 1) / 4
                                               + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 1) / 4
                                               + (-Ux * Ux / 4 + 1.0) * Vh(u, i, j - 1)) / 2
                          + (1.0 - Vy * Vy) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j) / 4
                                               + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j) / 4
                                               + (1.0 - Ux * Ux / 4) * Vh(u, i, j));
        }
    }
}

__global__ void WW2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = Vy * (Vy / 2 - 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 2) / 4
                                                 + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 2) / 4
                                                 + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 2)) / 4
                          + Vy * (Vy / 2 + 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 2) / 4
                                                   + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 2) / 4
                                                   + (1.0 - Ux * Ux / 4) * Vh(u, i, j - 2)) / 4
                          + (1.0 - Vy * Vy / 4) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j) / 4
                                                   + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j) / 4
                                                   + (1.0 - Ux * Ux / 4) * Vh(u, i, j));
        }
    }
}

__global__ void WF2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = (9.0 / 16 - Vy * Vy / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 1) / 4
                                                       + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 1) / 4
                                                       + (1.0 - Ux * Ux / 4) * Vh(u, i, j - 1))
                          + (9.0 / 16 - Vy * Vy / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 1) / 4
                                                         + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 1) / 4
                                                         + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 1))
                          + (Vy * Vy / 16 - Vy / 6 - 1.0 / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 3) / 4
                                                                  + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 3) / 4
                                                                  + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 3))
                          + (Vy * Vy / 16 + Vy / 6 - 1.0 / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 3) / 4
                                                                  + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 3) / 4
                                                                  + (-Ux * Ux / 4 + 1) * Vh(u, i, j - 3));
        }
    }
}

__global__ void FN2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){
    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = Vy * (Vy - 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 1)
                                             + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 1)
                                             + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 1)
                                             + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 1)) / 2
                          + Vy * (Vy + 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 1)
                                               + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 1)
                                               + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 1)
                                               + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 1)) / 2
                          + (1.0 - Vy * Vy) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j)
                                               + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j)
                                               + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j)
                                               + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j));
        }
    }
}

__global__ void FW2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = Vy * (Vy / 2 - 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 2)
                                                 + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 2)
                                                 + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 2)
                                                 + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 2)) / 4
                          + Vy * (Vy / 2 + 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 2)
                                                   + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 2)
                                                   + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 2)
                                                   + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 2)) / 4
                          + (1.0 - Vy * Vy / 4) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j)
                                                   + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j)
                                                   + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j)
                                                   + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j));
        }
    }
}

__global__ void FF2Kernel(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            V(uh, i, j) = (9.0 / 16 - Vy * Vy / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 1)
                                                       + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 1)
                                                       + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 1)
                                                       + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 1))
                          + (-Vy * Vy / 16 + 9.0 / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 1)
                                                          + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 1)
                                                          + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 1)
                                                          + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 1))
                          + (Vy * Vy / 16 - Vy / 6 - 1.0 / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 3)
                                                                  + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 3)
                                                                  + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 3)
                                                                  + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 3))
                          + (Vy * Vy / 16 + Vy / 6 - 1.0 / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 3)
                                                                  +(9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 3)
                                                                  + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 3)
                                                                  + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 3));
        }
    }
}


__global__ void LW2Dkernel1(HaloArray3D* u, HaloArray3D* uh, double Ux, double Vy, unsigned int opt_stset){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            double sts[9];
            int count = 0;
            if (STS_OPT(ST_NN2))
                sts[count++] = Vy * (Vy - 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 1) / 2
                                                  + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 1) / 2
                                                  + (1.0 - Ux * Ux) * Vh(u, i, j + 1)) / 2
                               + Vy * (Vy + 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 1) / 2
                                                    + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 1) / 2
                                                    + (1.0 - Ux * Ux) * Vh(u, i, j - 1)) / 2
                               + (1.0 - Vy * Vy) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j) / 2
                                                    + Ux * (Ux + 1.0) * Vh(u, i - 1, j) / 2
                                                    + (1.0 - Ux * Ux) * Vh(u, i, j));

            if (STS_OPT(ST_NF2))
                sts[count++] = (-Vy * Vy / 16 + 9.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 1) / 2
                                                             + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 1) / 2
                                                             + (1.0 - Ux * Ux) * Vh(u, i, j - 1))
                               + (-Vy * Vy / 16 + 9.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 1) / 2
                                                               + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 1) / 2
                                                               + (1.0 - Ux * Ux) * Vh(u, i, j + 1))
                               + (Vy * Vy / 16 - Vy / 6 - 1.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 3) / 2
                                                                       + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 3) / 2
                                                                       + (1.0 - Ux * Ux) * Vh(u, i, j + 3))
                               + (Vy * Vy / 16 + Vy / 6 - 1.0 / 16) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 3) / 2
                                                                       + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 3) / 2
                                                                       + (1.0 - Ux * Ux) * Vh(u, i, j - 3));

            if (STS_OPT(ST_WW2))
                sts[count++] = Vy * (Vy / 2 - 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 2) / 4
                                                      + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 2) / 4
                                                      + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 2)) / 4
                               + Vy * (Vy / 2 + 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 2) / 4
                                                        + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 2) / 4
                                                        + (1.0 - Ux * Ux / 4) * Vh(u, i, j - 2)) / 4
                               + (1.0 - Vy * Vy / 4) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j) / 4
                                                        + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j) / 4
                                                        + (1.0 - Ux * Ux / 4) * Vh(u, i, j));

            if (STS_OPT(ST_FW2))
                sts[count++] = Vy * (Vy / 2 - 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 2)
                                                      + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 2)
                                                      + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 2)
                                                      + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 2)) / 4
                               + Vy * (Vy / 2 + 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 2)
                                                        + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 2)
                                                        + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 2)
                                                        + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 2)) / 4
                               + (1.0 - Vy * Vy / 4) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j)
                                                        + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j)
                                                        + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j)
                                                        + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j));

            if (STS_OPT(ST_WF2))
                sts[count++] = (9.0 / 16 - Vy * Vy / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 1) / 4
                                                            + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 1) / 4
                                                            + (1.0 - Ux * Ux / 4) * Vh(u, i, j - 1))
                               + (9.0 / 16 - Vy * Vy / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 1) / 4
                                                              + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 1) / 4
                                                              + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 1))
                               + (Vy * Vy / 16 - Vy / 6 - 1.0 / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 3) / 4
                                                                       + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 3) / 4
                                                                       + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 3))
                               + (Vy * Vy / 16 + Vy / 6 - 1.0 / 16) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 3) / 4
                                                                       + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 3) / 4
                                                                       + (-Ux * Ux / 4 + 1) * Vh(u, i, j - 3));

            if (STS_OPT(ST_NW2))
                sts[count++] = Vy * (Vy / 2 - 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j + 2) / 2
                                                      + Ux * (Ux + 1.0) * Vh(u, i - 1, j + 2) / 2
                                                      + (1.0 - Ux * Ux) * Vh(u, i, j + 2)) / 4
                               + Vy * (Vy / 2 + 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j - 2) / 2
                                                        + Ux * (Ux + 1.0) * Vh(u, i - 1, j - 2) / 2
                                                        + (1.0 - Ux * Ux) * Vh(u, i, j - 2)) / 4
                               + (-Vy * Vy / 4 + 1.0) * (Ux * (Ux - 1.0) * Vh(u, i + 1, j) / 2
                                                         + Ux * (Ux + 1.0) * Vh(u, i - 1, j) / 2
                                                         + (1.0 - Ux * Ux) * Vh(u, i, j));

            if (STS_OPT(ST_WN2))
                sts[count++] = Vy * (Vy - 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j + 1) / 4
                                                  + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j + 1) / 4
                                                  + (1.0 - Ux * Ux / 4) * Vh(u, i, j + 1)) / 2
                               + Vy * (Vy + 1.0) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j - 1) / 4
                                                    + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j - 1) / 4
                                                    + (-Ux * Ux / 4 + 1.0) * Vh(u, i, j - 1)) / 2
                               + (1.0 - Vy * Vy) * (Ux * (Ux / 2 - 1.0) * Vh(u, i + 2, j) / 4
                                                    + Ux * (Ux / 2 + 1.0) * Vh(u, i - 2, j) / 4
                                                    + (1.0 - Ux * Ux / 4) * Vh(u, i, j));

            if (STS_OPT(ST_FN2))
                sts[count++] = Vy * (Vy - 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 1)
                                                  + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 1)
                                                  + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 1)
                                                  + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 1)) / 2
                               + Vy * (Vy + 1.0) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 1)
                                                    + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 1)
                                                    + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 1)
                                                    + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 1)) / 2
                               + (1.0 - Vy * Vy) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j)
                                                    + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j)
                                                    + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j)
                                                    + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j));

            if (STS_OPT(ST_FF2))
                sts[count++] = (9.0 / 16 - Vy * Vy / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 1)
                                                            + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 1)
                                                            + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 1)
                                                            + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 1))
                               + (-Vy * Vy / 16 + 9.0 / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 1)
                                                               + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 1)
                                                               + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 1)
                                                               + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 1))
                               + (Vy * Vy / 16 - Vy / 6 - 1.0 / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j + 3)
                                                                       + (9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j + 3)
                                                                       + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j + 3)
                                                                       + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j + 3))
                               + (Vy * Vy / 16 + Vy / 6 - 1.0 / 16) * ((9.0 / 16 - Ux * Ux / 16) * Vh(u, i - 1, j - 3)
                                                                       +(9.0 / 16 - Ux * Ux / 16) * Vh(u, i + 1, j - 3)
                                                                       + (Ux * Ux / 16 - Ux / 6 - 1.0 / 16) * Vh(u, i + 3, j - 3)
                                                                       + (Ux * Ux / 16 + Ux / 6 - 1.0 / 16) * Vh(u, i - 3, j - 3));

            if (count == 1)
                V(uh, i, j) = sts[count - 1];
            else{
                if ((count % 2) == 0 ) {
                    V(uh, i, j) = (sts[count / 2] + sts[(count + 1) / 2]) / 2;
                }
                else
                    V(uh, i, j) =  sts[count / 2];
            }
        }
    }
}

__global__ void LW2Dkernel2(HaloArray3D* u, HaloArray3D* uh){

    int i0 = blockIdx.x * blockDim.x + threadIdx.x, di = blockDim.x*gridDim.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y, dj = blockDim.y*gridDim.y;
    int x = i0 + j0 * di, total = di * dj;
    for (int j = x; j < u->l.x; j += total) {
        for(int i = 0; i < u->l.y; i++){
            Vh(u, i, j) = Vh(uh, i, j);
        }
    }
}

void Advect3D::updateLW2DCuda(HaloArray3D *u){
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;
    const dim3 gridSize(gdim, gdim, 1);
    const dim3 blockSize(bdim, bdim, 1);
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), 1);
    int sizeuh = uh->s.prod();
    HaloArray3D* d_uh;
    double* uuh;
    cudaMallocManaged((void**) &d_uh, sizeof(HaloArray3D));
    cudaMallocManaged((void**) &uuh, sizeof(double) * sizeuh);
    d_uh->u = uuh;
    d_uh->l = uh->l;
    d_uh->s = uh->s;
    d_uh->halo = uh->halo;
    d_uh->B = uh->B;
    switch (opt_stset){
        case STB_NN2:
            NN2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_NW2:
            NW2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_NF2:
            NF2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_WN2:
            WN2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_WW2:
            WW2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_WF2:
            WF2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_FN2:
            FN2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_FW2:
            FW2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        case STB_FF2:
            FF2Kernel<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy);
            break;
        default:
            LW2Dkernel1<<<gridSize, blockSize>>>(u, d_uh, Ux, Vy, opt_stset);
    }

    cudaDeviceSynchronize();
    LW2Dkernel2<<<gridSize, blockSize>>>(u, d_uh);
    cudaDeviceSynchronize();
    cudaFree(d_uh);
    cudaFree(uuh);
}

double Advect3D::simulateAdvectionCuda(HaloArray3D* u, double dtA){
    const dim3 gridSize(gdim, gdim, 1);
    const dim3 blockSize(bdim, bdim, 1);
    double t = 0.0;
    int s = 0;
    HaloArray3D* d_u;
    double* uu;
    int sizeu;
    if(opt_tmr){
        sizeu = u->s.prod() * 3;
        cudaMallocManaged((void**) &d_u, sizeof(HaloArray3D));
        cudaMallocManaged((void**) &uu, sizeof(double) * sizeu);
        for(int i = 0; i < sizeu; i++)
            uu[i] = u->u[i];
        d_u->u = uu;
        d_u->l = u->l;
        d_u->s = u->s;
        d_u->halo = u->halo;
        d_u->B = u->B;

        while (t < dtA) {
            updateLW2D_tmr_Cuda(d_u);
            boundary<<<gridSize, blockSize>>>(d_u);
            cudaDeviceSynchronize();
            t += dt; s++;
        }
    }else{
        sizeu = u->s.prod();
        cudaMallocManaged((void**) &d_u, sizeof(HaloArray3D));
        cudaMallocManaged((void**) &uu, sizeof(double) * sizeu);
        for(int i = 0; i < sizeu; i++)
            uu[i] = u->u[i];
        d_u->u = uu;
        d_u->l = u->l;
        d_u->s = u->s;
        d_u->halo = u->halo;
        d_u->B = u->B;

        if (opt_stset == STC_XX) {
            while (t < dtA) {
                updateLWN2Cuda(d_u);
                boundary<<<gridSize, blockSize>>>(d_u);
                cudaDeviceSynchronize();
                t += dt; s++;
            }
        } else if (opt_stset == STC_XX4) {
            while(t < dtA){
                updateLWN4Cuda(d_u);
                boundary<<<gridSize, blockSize>>>(d_u);
                cudaDeviceSynchronize();
                t += dt; s++;
            }
        }else{
            while(t < dtA){
                updateLW2DCuda(d_u);
                boundary<<<gridSize, blockSize>>>(d_u);
                cudaDeviceSynchronize();
                t += dt; s++;
            }
        }
    }

    for(int i = 0; i < sizeu; i++) {
        u->u[i] = d_u->u[i];
    }
    cudaFree(d_u);
    cudaFree(uu);
    return t;
}