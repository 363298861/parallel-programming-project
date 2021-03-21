#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <queue>
#include "AdvectCoeff.h"
#include "Advect3D.h"

#undef V
#undef Vh
#define V(u, i, j) (*((u) -> ix(i, j, 0)))
#define Vh(u, i, j) (*((u) -> ix_h(i, j, 0)))
#define VT(u, i, j, T) (*((u) -> ix(i, j, T)))
#define VhT(u, i, j, T) (*((u) -> ix_h(i, j, T)))

static int compare_doubles(const void* a, const void* b){
    if ( *(double*)a < *(double*)b) return -1;
    if ( *(double*)a > *(double*)b) return 1;
    return 0;
}

double median3(double a, double b, double c){
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

static double medianN(double* sts, int nitems){
    double valid_sts[9];
    int valid_count = 0;

    for (int i=0; i<nitems; i++){
        if ( std::isnan(sts[i]) || std::isinf(sts[i])) continue;
        valid_sts[valid_count++] = sts[i];
    }

    qsort( valid_sts, valid_count, sizeof(double), compare_doubles);

    if ( nitems != valid_count ){
        printf("%d:", valid_count);
        for (int j=0; j<valid_count; j++){
            std::cout << valid_sts[j] << ";";
        }
        printf("\n");
    }

    if ( (valid_count%2) == 0 ) {
        return (sts[valid_count/2] + sts[(valid_count+1)/2]) / 2;
    }
    return sts[valid_count/2];
}

void Advect3D::updateLW2D_tmr(HaloArray3D* u){
    HaloArray3D *uh[3];

    for(int t = 0; t < 3; t++){
        uh[t] = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    }

    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for(int j = 0; j < u->s.y; j++){
        for(int i = 0; i < u-> s.x; i++){
            VT(u,i,j,1) = VT(u,i,j,2) =  VT(u,i,j,0);
        }
    }

    for (int j=0; j < u->l.y; j++) { //uh->l.y
        for (int i = 0; i < u->l.x; i++) { //uh->l.x{
            for (int t = 0; t < 3; t++) {
                V(uh[t], i, j) = Vy * (Vy - 1.0) * (Ux * (Ux - 1.0) * VhT(u, i + 1, j + 1, t) / 2
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

    for (int j=0; j < u->l.y; j++) {//uh->l.y
        for (int i = 0; i < u->l.x; i++) { //uh->l.x{
            VhT(u, i, j, 0) = median3(Vh(uh[0], i, j), Vh(uh[1], i, j), Vh(uh[2], i, j));
        }
    }

    for (int t=0; t<3; t++)
        delete uh[t];
}

void Advect3D::updateLWN2(HaloArray3D *u) {
    HaloArray3D *uh = new HaloArray3D(u->l, Vec3D<int>(0), 1);
    double Ux = V.x * dt / delta.x, Uy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            double cim1, ci0, cip1;
            double cjm1, cj0, cjp1;
            N2Coeff(Ux, cim1, ci0, cip1);
            N2Coeff(Uy, cjm1, cj0, cjp1);
            Vh(uh, i, j) = cim1 * (cjm1 * Vh(u, i - 1, j - 1) + cj0 * Vh(u, i - 1, j) + cjp1 * Vh(u, i - 1, j + 1)) +
                           ci0 * (cjm1 * Vh(u, i, j - 1) + cj0 * Vh(u, i, j) + cjp1 * Vh(u, i, j + 1)) +
                           cip1 * (cjm1 * Vh(u, i + 1, j - 1) + cj0 * Vh(u, i + 1, j) + cjp1 * Vh(u, i + 1, j + 1));
        }
    }

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = Vh(uh, i, j);
        }
    }
    delete uh;
} //updateLWN2()

void Advect3D::updateLWN4(HaloArray3D *u) {
    HaloArray3D *uh = new HaloArray3D(u->l, Vec3D<int>(0), 1);
    double Ux = V.x * dt / delta.x, Uy = V.y * dt / delta.y;

    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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

    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = Vh(uh, i, j);
        }
    }
    delete uh;
} //updateLWN4()

void Advect3D::updateNN2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateNW2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateNF2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateWN2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateWW2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateWF2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateFN2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateFW2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateFF2(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);
    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateSTC30(HaloArray3D* u){
    updateWF2(u);
}

void Advect3D::updateSTC31(HaloArray3D* u){
    updateFW2(u);
}

void Advect3D::updateSTC32(HaloArray3D* u){
    updateNF2(u);
}

void Advect3D::updateSTC50(HaloArray3D* u){
    updateFW2(u);
}

void Advect3D::updateSTC70(HaloArray3D* u){
    updateWF2(u);
}

void Advect3D::updateElse(HaloArray3D* u){
    HaloArray3D *uh = new HaloArray3D(Vec3D<int>(u->s.x - 1, u->s.y - 1, 1), Vec3D<int>(0), B);

    double Ux = V.x * dt / delta.x, Vy = V.y * dt / delta.y;

    for (int j = 0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
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
            else
                V(uh, i, j) = medianN(sts, count);
        }
    }

    for (int j=0; j < u->l.y; j++) {
        for (int i = 0; i < u->l.x; i++) {
            Vh(u, i, j) = V(uh, i, j);
        }
    }
    delete uh;
}

void Advect3D::updateLW2D(HaloArray3D *u){
    switch (opt_stset){
        case STC_XX:
            updateLWN2(u);
            break;
        case STC_XX4:
            updateLWN4(u);
            break;
        case STB_NN2:
            updateNN2(u);
            break;
        case STB_NW2:
            updateNW2(u);
            break;
        case STB_NF2:
            updateNF2(u);
            break;
        case STB_WN2:
            updateWN2(u);
            break;
        case STB_WW2:
            updateWW2(u);
            break;
        case STB_WF2:
            updateWF2(u);
            break;
        case STB_FN2:
            updateFN2(u);
            break;
        case STB_FW2:
            updateFW2(u);
            break;
        case STB_FF2:
            updateFF2(u);
            break;
//        case STC_30:
//            updateSTC30(u);
//            break;
//        case STC_31:
//            updateSTC31(u);
//            break;
//        case STC_32:
//            updateSTC32(u);
//            break;
//        case STC_50:
//            updateSTC50(u);
//            break;
//        case STC_70:
//            updateSTC70(u);
//            break;
        default:
            updateElse(u);
    }
}
