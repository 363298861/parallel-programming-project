#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include "Advect3D.h"

#define USAGE   "twoDimAdvect [-S steps] [g.x g.y] [-x stc] [-d gridDim,blockDim] [-t] [-c]\n\t"
#define CONSTRAINTS "\n\t "
#define DEFAULTS "g.x=g.y=6 tF=0.25 halo=3 stc=STC_50"
#define OPTCHARS "S:x:d:ctz:"
#define NOT_SET -1

bool parallel = false;
static bool is2D = true;     // if set, perform 2D simulation (force g.z==0)
static Vec3D<int> grid;
static double tF = 0.25, CFL = 0.25;
static int steps = NOT_SET;  // #timesteps for sparse grid (dept. on grid & TF)
int gdim = 2, bdim = 32;
int size = 8;

unsigned int opt_stset = STC_50;
static char stsopt[16];
static int opt_halosize = 3;
bool opt_tmr = false;

void usage(std::string msg) {
    printf("twoDimAdvect: %s\n", msg.c_str());
    printf("usage: %s\n\tconstraints: %s\n\tdefault values: %s\n\t", USAGE, CONSTRAINTS, DEFAULTS);
    fflush(stdout);
    exit(1);
}

Vec3D<int> gridSz(Vec3D<int> gix) {
    Vec3D<int> g;
    g.x = 1 + (1 << gix.x);
    g.y = 1 + (1 << gix.y);
    g.z = (1 << gix.z);
    return (g);
}

void getArgs(int argc, char *argv[]) {
    extern char *optarg; // points to option argument (for -p option)
    extern int optind;   // index of last option parsed by getopt()
    extern int opterr;
    char optchar;        // option character returned my getopt()
    opterr = 0;          // suppress getopt() error message for invalid option
    bzero(stsopt, sizeof(stsopt));

    while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
        // extract next option from the command line
        switch (optchar) {
            case 'x':
                sscanf(optarg, "%8s", stsopt);
                // for new 2d stencils
                if      ( strncmp( stsopt, "NN2", 3) == 0 ) opt_stset=STB_NN2;
                else if ( strncmp( stsopt, "NW2", 3) == 0 ) opt_stset=STB_NW2;
                else if ( strncmp( stsopt, "NF2", 3) == 0 ) opt_stset=STB_NF2;

                else if ( strncmp( stsopt, "WN2", 3) == 0 ) opt_stset=STB_WN2;
                else if ( strncmp( stsopt, "WW2", 3) == 0 ) opt_stset=STB_WW2;
                else if ( strncmp( stsopt, "WF2", 3) == 0 ) opt_stset=STB_WF2;
                else if ( strncmp( stsopt, "FN2", 3) == 0 ) opt_stset=STB_FN2;
                else if ( strncmp( stsopt, "FW2", 3) == 0 ) opt_stset=STB_FW2;
                else if ( strncmp( stsopt, "FF2", 3) == 0 ) opt_stset=STB_FF2;

                else if ( strncmp( stsopt, "C30", 3) == 0 ) opt_stset=STC_30;
                else if ( strncmp( stsopt, "C31", 3) == 0 ) opt_stset=STC_31;
                else if ( strncmp( stsopt, "C32", 3) == 0 ) opt_stset=STC_32;
                else if ( strncmp( stsopt, "C50", 3) == 0 ) opt_stset=STC_50;
                else if ( strncmp( stsopt, "C70", 3) == 0 ) opt_stset=STC_70;
                else if ( strncmp( stsopt, "XX", 3) == 0 )  opt_stset=STC_XX;
                else if ( strncmp( stsopt, "XX4", 3) == 0 ) opt_stset=STC_XX4;
                else usage("bad value for stencils (-x)");
                break;
            case 'S':
                if (sscanf(optarg, "%d", &steps) != 1)
                    usage("bad value for steps");
                break;
            case 'c':
                parallel = true;
                break;
            case 't':
                opt_tmr = true;
                break;
            case 'd':
                if(sscanf(optarg, "%d,%d", &gdim, &bdim) != 2)
                    usage("bad value for GPU dimensions");
                break;
            case 'z':
                if (sscanf(optarg, "%d", &size) != 1)
                    usage("bad value for steps");
                break;
            default:
                usage("unknown option");
                break;
        }
    }

    grid.x = grid.y = 6; grid.z = 0;
    if (optind < argc)
        if (sscanf(argv[optind], "%d", &grid.x) != 1)
            usage("bad value g.x");
    if (optind+1 < argc)
        if (sscanf(argv[optind+1], "%d", &grid.y) != 1)
            usage("bad value g.y");
    if (optind+2 < argc)
        if (sscanf(argv[optind+2], "%d", &grid.z) != 1)
            usage("bad value g.z");
    if (is2D)
        grid.z = 0;

    int dx = 1 << std::max(grid.x,  std::max(grid.y, grid.z));
    if (steps == NOT_SET)
        steps = (int) (tF/CFL * dx);
    else
        tF = CFL * steps / (1.0*dx);

    printf("Stencil %s=%x TMR=%d halo=%d on grid " V3DFMT "\n", stsopt, opt_stset, opt_tmr, opt_halosize, V3DLST(grid));
}

static void printLocGlobVals(bool isMax, std::string name, double total, int nlVals, bool is2d, Vec3D<int> gix) {
    int ngVals = gridSz(gix).prod();
    printf("grid (%d,%d,%d): %s %s is %.2e\n", gix.x, gix.y,
               gix.z, isMax? "max": "avg", name.c_str(),
               isMax? total: (nlVals==0? 0.0: total / nlVals));
}

int main(int argc, char** argv) {
    getArgs(argc, argv);
    Vec3D<int> haloSize(opt_halosize, opt_halosize, 0);
    HaloArray3D* u = new HaloArray3D(gridSz(grid) ,haloSize, 1);
    Advect3D* adv = new Advect3D(gridSz(grid), Vec3D<double>(1.0), tF, CFL);

    printf("2D advection on grid %d,%d,%d for time %e = max %d steps (dt=%e)\n",
            grid.x, grid.y, grid.z, tF, steps, adv->dt);
    if(parallel)
        printf("Grid dim is %d * %d and block dim is %d * %d\n", gdim, gdim, bdim, bdim);

    int nPtsL = u->l.prod();
    adv->initGrid(u);
    adv->updateBoundary(u);
    double t = 0.0;
    t += adv->simulateAdvection(u, tF);
    double uError = adv->checkError(t, u);
    printLocGlobVals(false, "error of final field", uError, nPtsL, is2D, grid);
    uError = adv->checkMaxError(t, u);
    printLocGlobVals(true, "error of final field", uError, nPtsL, is2D, grid);

    delete adv;
    delete u;
    return 0;
}