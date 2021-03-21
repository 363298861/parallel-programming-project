// Parallel 3D Advection program
// written by Peter Strazdins, Jun 14

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt(), gethostname()
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <string.h> //strncmp()
#include <strings.h> //bzero()
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Advect3D.h"
#include "MemCorrupter.h"

#define USAGE   "twoDimAdvect [-v v] [-V vT] [-p pRow] [-P P] [-T tF] [-S steps] [g.x [g.y [g.z]]]] [-x stc] [-i] [-t] [-a] [-w] [-h halo] [-e Pbitflip] \n\t "
#define CONSTRAINTS "\n\t "
#define DEFAULTS "v=vT=0 g.x=g.y=6 pRow=1 P=1 tF=0.25 halo=3 stc=STC_50 Pbitflip=0"
#define NOTES "-t: use TMR, -w: warming -e: mem corrrupt, -i: oneshot"
#define OPTCHARS "v:p:P:T:S:V:x:ie:H:tw"

#define NOT_SET -1

/*  0: top-level messages from rank 0, 1: 1-off messages from all ranks,
    2: per-iteration messages, 3: dump data: one-off, 4: dump data, per itn.
 */
static int verbosity = 0;    // v above    
static int verbTim  = -1;    // vT above. Verbosity of timer output 
static bool is2D = true;     // if set, perform 2D simulation (force g.z==0)
static Vec3D<int> grid;
static double tF = 0.25,     // final time; must it be a multiple of 0.25?
	      CFL = 0.25;    // CFL condition number 
//	      CFL = 0.125;   // CFL condition number 
//	      CFL = 0.5;     // CFL condition number 
static int rank, nprocs;     // MPI values
static int pRow = 1;         // number of processes across row of process grid
static int P = 1;            // number of OpenMP threads per process
static int steps = NOT_SET;  // #timesteps for sparse grid (dept. on grid & TF)

unsigned int opt_stset = STC_50;
static  char stsopt[16];

bool   opt_oneshot = false;
bool   opt_memcorrupt = false;
float  opt_bitflipprob = 0;
static int opt_halosize = 3;
bool   opt_tmr = false;
bool   opt_warming = false;

#define IS_POWER2(n) (((n) & (n-1)) == 0)

// print a usage message for this program and exit with a status of 1
void usage(std::string msg) {
  if (rank==0) {
    printf("twoDimAdvect: %s\n", msg.c_str());
    printf("usage: %s\n\tconstraints: %s\n\tdefault values: %s\n\tnotes: %s", 
	   USAGE, CONSTRAINTS, DEFAULTS, NOTES);
    fflush(stdout);
  }
  exit(1);
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
                if (strncmp(stsopt, "NN2", 3) == 0) opt_stset = STB_NN2;
                else if (strncmp(stsopt, "NW2", 3) == 0) opt_stset = STB_NW2;
                else if (strncmp(stsopt, "NF2", 3) == 0) opt_stset = STB_NF2;

                else if (strncmp(stsopt, "WN2", 3) == 0) opt_stset = STB_WN2;
                else if (strncmp(stsopt, "WW2", 3) == 0) opt_stset = STB_WW2;
                else if (strncmp(stsopt, "WF2", 3) == 0) opt_stset = STB_WF2;
                else if (strncmp(stsopt, "FN2", 3) == 0) opt_stset = STB_FN2;
                else if (strncmp(stsopt, "FW2", 3) == 0) opt_stset = STB_FW2;
                else if (strncmp(stsopt, "FF2", 3) == 0) opt_stset = STB_FF2;

                else if (strncmp(stsopt, "C30", 3) == 0) opt_stset = STC_30;
                else if (strncmp(stsopt, "C31", 3) == 0) opt_stset = STC_31;
                else if (strncmp(stsopt, "C32", 3) == 0) opt_stset = STC_32;
                else if (strncmp(stsopt, "C50", 3) == 0) opt_stset = STC_50;
                else if (strncmp(stsopt, "C70", 3) == 0) opt_stset = STC_70;
                else if (strncmp(stsopt, "XX", 3) == 0) opt_stset = STC_XX;
                else if (strncmp(stsopt, "XX4", 3) == 0) opt_stset = STC_XX4;
                else usage("bad value for stencils (-x)");
                break;
            case 'w':
                opt_warming = true;
                break;
            case 'H':
                if (sscanf(optarg, "%d", &opt_halosize) != 1) // invalid integer
                    usage("bad value for Halosize");
                break;
            case 't':
                opt_tmr = true;
                break;
            case 'i':
                opt_oneshot = true;
                break;
            case 'e':
                opt_memcorrupt = true;
                if (sscanf(optarg, "%f", &opt_bitflipprob) != 1)
                    usage("bad value for bitflipprob");
                break;
            case 'v':
                if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer
                    usage("bad value for verbose");
                break;
            case 'p':
                if (sscanf(optarg, "%d", &pRow) != 1)
                    usage("bad value for pRow");
                break;
            case 'P':
                if (sscanf(optarg, "%d", &P) != 1)
                    usage("bad value for P");
                break;
            case 'S':
                if (sscanf(optarg, "%d", &steps) != 1)
                    usage("bad value for steps");
                break;
            case 'T':
                if (sscanf(optarg, "%lf", &tF) != 1)
                    usage("bad value for tF");
                break;
            case 'V':
                if (sscanf(optarg, "%d", &verbTim) != 1)
                    usage("bad value for vT");
                break;
            default:
                usage("unknown option");
                break;
        } //switch
    } //while

    grid.x = grid.y = 6;
    grid.z = 0;
    if (optind < argc)
        if (sscanf(argv[optind], "%d", &grid.x) != 1)
            usage("bad value g.x");
    if (optind + 1 < argc)
        if (sscanf(argv[optind + 1], "%d", &grid.y) != 1)
            usage("bad value g.y");
    if (optind + 2 < argc)
        if (sscanf(argv[optind + 2], "%d", &grid.z) != 1)
            usage("bad value g.z");
    if (is2D)
        grid.z = 0;

    int dx = 1 << std::max(grid.x, std::max(grid.y, grid.z));
    if (steps == NOT_SET)
        steps = (int) (tF / CFL * dx);
    else
        tF = CFL * steps / (1.0 * dx);

#ifdef _OPENMP
    omp_set_num_threads(P);
#endif
    if (rank == 0) {
        printf("Stencil %s=%x TMR=%d halo=%d on grid " V3DFMT " Pbitflip=%f (%d MPI procs, %d per row)\n",
               stsopt, opt_stset, opt_tmr, opt_halosize, V3DLST(grid),
               opt_bitflipprob, nprocs, pRow);
#ifdef _OPENMP
        if (omp_get_max_threads() > 1  &&  verbosity > 0) {
          printf("\t%d threads per proc\n", omp_get_max_threads());
#pragma omp parallel
         {
           printf("%d: hello from thread %d\n", rank, omp_get_thread_num());
          }
#endif
    }
}

}//getArgs()


Vec3D<int> gridSz(Vec3D<int> gix) {
  Vec3D<int> g;
  g.x = 1 + (1 << gix.x);
  g.y = 1 + (1 << gix.y);
  g.z = (1 << gix.z);
  return (g);
}  

static void printLocGlobVals(bool isMax, std::string name, double total, 
			     int nlVals, bool is2d, Vec3D<int> gix, 
			     MPI_Comm comm) {
  int ngVals = gridSz(gix).prod();
  double v[1];  
  if (verbosity > 0)  
    printf("%d: grid (%d,%d,%d): local %s %s is %.2e\n", rank, gix.x, gix.y, 
	   gix.z, isMax? "max": "avg", name.c_str(), 
	   isMax? total: (nlVals==0? 0.0: total / nlVals));
  MPI_Reduce(&total, v, 1, MPI_DOUBLE, isMax? MPI_MAX: MPI_SUM, 0, comm);
  int grank;
  MPI_Comm_rank(comm, &grank);
  if (grank == 0)
    printf("%d: grid (%d,%d,%d): %s %s %.2e\n", rank, gix.x, gix.y, gix.z, 
	   isMax? "max": "avg", name.c_str(), isMax? v[0]: v[0] / ngVals);
}


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm myGridComm = MPI_COMM_WORLD;
  getArgs(argc, argv);

  Timer *timer = new Timer;

  Vec3D<int> P = Vec3D<int>(pRow, nprocs/pRow, 1);
  ProcGrid3D *g = new ProcGrid3D(rank, P, myGridComm);

  Vec3D<int> haloSize(opt_halosize, opt_halosize, 0);
  HaloArray3D *u = new HaloArray3D(g->G2L(gridSz(grid)), 
                    haloSize, 1);
   
  Advect3D *adv = new Advect3D(gridSz(grid), Vec3D<double>(1.0), 
			       tF, CFL, verbosity, timer);
  if (rank == 0) {
    printf("%dD advection on grid %d,%d,%d for time %e = max %d steps (dt=%e)\n", 
	   is2D? 2: 3, grid.x, grid.y, grid.z, tF, steps, adv->dt); 
  }
  if (verbosity > 0) { 
    char hostName[128];
    gethostname(hostName, sizeof(hostName));
    printf("%d: process (%d,%d,%d) grid (%d,%d,%d) with %dx%dx%d processes", 
	   rank, g->id.x, g->id.y, g->id.z, 
	   grid.x, grid.y, grid.z, g->P.x, g->P.y, g->P.z);
    printf(" computes %dx%dx%d points from (%d,%d,%d) on host %s\n", 
	   u->l.x, u->l.y, u->l.z, g->L2G0(0, adv->gridSize.x), 
	   g->L2G0(1, adv->gridSize.y), g->L2G0(2, adv->gridSize.z), hostName);
  }

  int nPtsL = u->l.prod();
  adv->initGrid(u, g);
  adv->updateBoundary(u, g);
  if (verbosity > 3)
    u->print(rank, "initial field");

  double t = 0.0; 
  t += adv->simulateAdvection(u, g, tF);
  double uError = adv->checkError(t, u, g);                 
  printLocGlobVals(false, "error of final field", uError, nPtsL, is2D, grid,
		   myGridComm);  
  uError = adv->checkMaxError(t, u, g);                 
  printLocGlobVals(true, "error of final field", uError, nPtsL, is2D, grid,
		   myGridComm);  
  if (verbosity > 3)
    u->print(rank, "final field");
  
  timer->dump(myGridComm, verbTim);

  if (opt_memcorrupt){
    MemCorrupter *mc = MemCorrupter::getInstance();
    
    unsigned long long it = mc->exit();
    printf("it: %llu\n", it);
    printf("Each bit was flipped with probability %g per timestep (%g,%g)\n",
          opt_bitflipprob * (double)it / steps, (double)it, (double)steps);
    delete mc;
  }

  delete g; delete adv; 
  delete u; 
  delete timer;

  MPI_Finalize();
  return 0;
} //main()

