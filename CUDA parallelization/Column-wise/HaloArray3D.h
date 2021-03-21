#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef CUDAADV_HALOARRAY3D_H
#define CUDAADV_HALOARRAY3D_H

#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <string>
#include "Vec3D.h"
extern bool opt_tmr;

class HaloArray3D {
public:
    double *u;
    Vec3D<int> l, s;
    Vec3D<int> halo;
    int B;

    HaloArray3D (Vec3D<int> l_, Vec3D<int> h, int blk = 1){
        l = l_;
        B = blk;
        l.x = l.x * B;
        halo = h;
        s = l + Vec3D<int> (B, 1, 1) * 2 * halo;

        if(s.prod() > 0){
            if(opt_tmr)
                u = new double[s.prod() * 3];
            else
                u = new double[s.prod()];
        }else{
            u = 0;
        }
    }

    ~HaloArray3D(){
        if(u != 0)
            delete[] u;
    }

    CUDA_HOSTDEV inline double* ix(int i, int j, int k){
        return (&u[i + s.x * (j + s.y * k)]);
    }

    CUDA_HOSTDEV inline double* ix_h(int i, int j, int k) {
        return ix(i + halo.x*B, j + halo.y, k + halo.z);
    }
};

#define V(u, i, j, k) (*((u)->ix(i, j, k)))
#define Vh(u, i, j, k) (*((u)->ix_h(i, j, k)))

#endif //CUDAADV_HALOARRAY3D_H
