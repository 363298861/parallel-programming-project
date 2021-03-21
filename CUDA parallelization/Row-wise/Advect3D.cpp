
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Advect3D.h"
extern bool opt_tmr;
extern bool parallel;

Advect3D::Advect3D(Vec3D<int> gridSz, Vec3D<double> v, double timef, double CFL, int blk) {

    B = blk;
    gridSize = gridSz;
    V = v;
    delta.x = 1.0 / (gridSize.x - 1);
    delta.y = 1.0 / (gridSize.y - 1);
    delta.z = is2D()? 1.0: 1.0 / (gridSize.z - 1);
    tf = timef;
    dt = CFL * std::min<double>(delta.x, std::min<double>(delta.y, delta.z));
}

void Advect3D::initGrid(HaloArray3D *u) {
    for (int kj = 0; kj < u->l.z*u->l.y; kj++) {
        int k = kj / u->l.y, j = kj % u->l.y;
        double z = delta.z * k;
        double y = delta.y * j;
        for (int i=0; i < u->l.x; i++) {
            double x = delta.x  * i;
            Vh(u, i, j, k) = initialCondition(x, y, z, 0.0, V.x, V.y, V.z);
        }
    }
}

double Advect3D::checkError(double t, HaloArray3D *u) {
    double err = 0.0;
    for (int kj = 0; kj < u->l.z*u->l.y; kj++) {
        int k = kj / u->l.y, j = kj % u->l.y;
        double z = delta.z * k;
        double y = delta.y * j;
        for (int i=0; i < u->l.x; i++) {
            double x = delta.x  * i;
            err += std::abs(Vh(u, i, j, k) - initialCondition(x, y, z, t, V.x, V.y, V.z));
        }
    }
    return (err);
}

double Advect3D::checkMaxError(double t, HaloArray3D *u) {
    double err = 0.0;

    for (int kj = 0; kj < u->l.z*u->l.y; kj++) {
        int k = kj / u->l.y, j = kj % u->l.y;
        double z = delta.z * k;
        double y = delta.y * j;
        for (int i=0; i < u->l.x; i++) {
            double x = delta.x  * i;
            double e = std::abs(Vh(u, i, j, k) - initialCondition(x, y, z, t, V.x, V.y, V.z));
            if (e > err) err = e;
        }
    }
    return (err);
}

void Advect3D::updateLW(HaloArray3D *u) {
    if(opt_tmr){
        updateLW2D_tmr(u);
    }else{
        updateLW2D(u);
    }
}

void Advect3D::updateBoundary(HaloArray3D *u) {

    int lx = u->l.x, ly = u->l.y, lz = u->l.z, sx = u->s.x, sy = u->s.y;
    int hx = u->halo.x, hy = u->halo.y, hz = u->halo.z;
    int i, j;

    for (j = hy; j < ly + hy; j++) {
        for (int ib = 0; ib < hx; ib++) {
            V(u, ib, j, 0) = V(u, lx + ib - B, j, 0);
        }
    }

    for (j = hy; j < ly + hy; j++) {
        for (int ib = 0; ib < hx; ib++) {
            V(u, lx + hx + ib, j, 0) = V(u, hx + ib + B, j, 0);
        }
    }

    for (j = 0; j < hy; j++) {
        for (i = 0; i < sx; i++) {
            V(u, i, j, 0) = V(u, i, ly + j - 1, 0);
        }
    }

    for (j = 0; j < hy; j++) {
        for (i = 0; i < sx; i++) {
            V(u, i, ly + hy + j, 0) = V(u, i, hy + j + 1, 0);
        }
    }
}

double Advect3D::simulateAdvection(HaloArray3D *u, double dtA) {

    double t = 0.0;
    int s = 0;

    if (is2D()) // check that u, g have been set up appropriately
        assert (u->halo.z == 0  &&  u->l.z <= 1);

    GpuTimer timer;
    timer.Start();
    if(!parallel){
        while (t < dtA) {
            updateLW(u);
            updateBoundary(u);
            t += dt; s++;
        }
        timer.Stop();
        printf("Time elapsed using serial programming = %g ms\n", timer.Elapsed());
    }else{
        t = simulateAdvectionCuda(u, dtA);
        timer.Stop();
        printf("Time elapsed using parallel programming = %g ms\n", timer.Elapsed());
    }
    return t;
}













