// Parallel 3D Advection Class
// based on 2D advection class
// written by Peter Strazdins, June 14

#include <stdio.h>
#include <cmath> // M_PI
#include "HaloArray3D.h"
#include "ProcGrid3D.h"
#include "Timer.h"

#define STB_NN2 (0x0001)
#define STB_NW2 (0x0002)
#define STB_NF2 (0x0004)

#define STB_WN2 (0x0008)
#define STB_WW2 (0x0010)
#define STB_WF2 (0x0020)

#define STB_FN2 (0x0040)
#define STB_FW2 (0x0080)
#define STB_FF2 (0x0100)

// RS Stencil combination STC_(num stencils)(set number)
#define STC_30 ( STB_WF2 | STB_FW2 | STB_FF2 )
#define STC_31 ( STB_WW2 | STB_WF2 | STB_FW2 )
#define STC_32 ( STB_NN2 | STB_NW2 | STB_NF2 )

#define STC_50 ( STB_NN2 | STB_WW2 | STB_WF2 | STB_FW2 | STB_FF2 )
#define STC_70 ( STB_NN2 | STB_NW2 | STB_NF2 | STB_WN2 | STB_WF2 \
               | STB_FN2 | STB_FW2)

#define STC_XX 0
#define STC_XX4 0x200

#define ST_NN2 0
#define ST_NW2 1
#define ST_NF2 2

#define ST_WN2 3
#define ST_WW2 4
#define ST_WF2 5

#define ST_FN2 6
#define ST_FW2 7
#define ST_FF2 8

extern unsigned int opt_stset;
#define STS_OPT(__pos__ )\
    ( (opt_stset) & ( 1<<(__pos__) ) )

class Advect3D {
    public:
    double tf /*final time*/,
         dt /*timestep*/;
    Vec3D<double> delta /*grid spacing*/, V /*advection velocity*/;
    Vec3D<int> gridSize /*number of points in grid*/;
    int verbosity; // for debugging output
    Timer *timer;
    int B;           // size of individual elements
    MPI_Comm myCommWorld;

    Advect3D(){}

    Advect3D(Vec3D<int> gridSz, Vec3D<double> v, double timef, double CFL,
       int verb, Timer *timer, MPI_Comm myCW = MPI_COMM_WORLD, int blk = 1);

    virtual ~Advect3D(){}

    // update u for a timestep by various schemes
    void updateLW(HaloArray3D *u);
    void updateLWN2(HaloArray3D *u);
    void updateLWN4(HaloArray3D *u);
    void updateLW2D(HaloArray3D *u);
    void updateLW2D_tmr(HaloArray3D *u);
    void updateLW2D_ST_XX(HaloArray3D *u); // original implementation, base

    inline bool is2D() {
    return (gridSize.z == 1);
    }

    inline double initialCondition(double x, double y, double z, double t=0.0,
                 double vx=1.0, double vy=1.0, double vz=1.0) {
    x = x - vx*t;
    y = y - vy*t;
    z = z - vz*t;
    double u = std::sin(4.0*M_PI*x) * std::sin(2.0*M_PI*y) ;
    if (!is2D())
    u *= std::sin(6.0*M_PI*z);
    return u;
    }

    void initGrid(HaloArray3D *u, ProcGrid3D *g);

    double checkError(double t, HaloArray3D *u, ProcGrid3D *g);

    double checkMaxError(double t, HaloArray3D *u, ProcGrid3D *g);

    void updateBoundary(HaloArray3D *u, ProcGrid3D *g);

    double simulateAdvection(HaloArray3D *u, ProcGrid3D *g, double dtA);

    //Each stencil has its own function
    void updateNN2(HaloArray3D* u);
    void updateNW2(HaloArray3D* u);
    void updateNF2(HaloArray3D* u);
    void updateWN2(HaloArray3D* u);
    void updateWW2(HaloArray3D* u);
    void updateWF2(HaloArray3D* u);
    void updateFN2(HaloArray3D* u);
    void updateFW2(HaloArray3D* u);
    void updateFF2(HaloArray3D* u);
//    void updateSTC30(HaloArray3D* u);
//    void updateSTC31(HaloArray3D* u);
//    void updateSTC32(HaloArray3D* u);
//    void updateSTC50(HaloArray3D* u);
//    void updateSTC70(HaloArray3D* u);
    void updateElse(HaloArray3D* u);

};


