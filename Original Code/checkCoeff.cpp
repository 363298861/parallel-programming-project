// written by Peter Strazdins, Oct 19

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <string>   //std::string
#include <math.h>   //fabs()
#include "AdvectCoeff.h"

#define USAGE   "checkCoeff u v\n"
#define OPTCHARS ""

static double u, v;          // CFL number x & y-dim

// print a usage message for this program and exit with a status of 1
void usage(std::string msg) {
  printf("checkCoeff: %s\n", msg.c_str());
  printf("usage: %s\n", USAGE);
  exit(1);
}

void getArgs(int argc, char *argv[]) {
  float fu=0.0, fv=0.0;
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  char optchar;        // option character returned my getopt()
  opterr = 0;          // suppress getopt() error message for invalid option

  while (0 && (optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    default:
      usage("unknown option");
      break;
    } //switch 
   } //while

  if (optind+1 >= argc)
    usage("u or v missing");
  if (sscanf(argv[optind], "%f", &fu) != 1)
    usage("bad value u");
  if (sscanf(argv[optind+1], "%f", &fv) != 1)
    usage("bad value v");
  u = fu; v = fv;
} //getArgs()


double checkAdvCoeff(int n, double *v) {
  double s = 0.0; int i;
  for (i=0; i < n; i++)
    s += v[i];
  return (s-1.0);
} 

int main(int argc, char** argv) {
  double x1C[5], y1C[5], x2C[3], y2C[3];
  double xy1C[5][5], xy2C[3][3];
  int i, j;

  getArgs(argc, argv);
  N4Coeff(u, x1C[0], x1C[1], x1C[2], x1C[3], x1C[4]); 
  N4Coeff(v, y1C[0], y1C[1], y1C[2], y1C[3], y1C[4]);
  N2Coeff(u, x2C[0], x2C[1], x2C[2]); 
  N2Coeff(v, y2C[0], y2C[1], y2C[2]);

  for (i=0; i < 5; i++)
    for (j=0; j < 5; j++)
      xy1C[i][j] = x1C[i] * y1C[j];
  for (i=0; i < 3; i++)
    for (j=0; j < 3; j++)
      xy2C[i][j] = x2C[i] * y2C[j];

  for (i=0; i < 5; i++) {
    for (j=0; j < 5; j++)
      printf("%+.1e ", xy1C[i][j]);
    printf("\n");
  } 
  printf("\n");   
  for (i=0; i < 3; i++) {
    for (j=0; j < 3; j++)
      printf("%+.1e ", xy2C[i][j]);
    printf("\n");
  } 
  printf("\n");   
  for (i=0; i < 3; i++) {
    for (j=0; j < 3; j++)
      printf("%+.1e ", (xy1C[i+1][j+1]-xy2C[i][j]));
    printf("\n");
  }    

  printf("checks: N4=%.2e,%.2e N2=%.2e,%.2e\n", checkAdvCoeff(5, x1C), 
	 checkAdvCoeff(5, y1C), checkAdvCoeff(3, x2C), checkAdvCoeff(3, y2C));

  F2Coeff(u, x1C[0], x1C[1], x1C[2], x1C[3]); 
  F2Coeff(v, y1C[0], y1C[1], y1C[2], y1C[3]);
  W2Coeff(u, x2C[0], x2C[1], x2C[2]);
  W2Coeff(v, y2C[0], y2C[1], y2C[2]);

  printf("checks: F2=%.2e,%.2e W2=%.2e,%.2e\n", checkAdvCoeff(4, x1C), 
	 checkAdvCoeff(4, y1C), checkAdvCoeff(3, x2C), checkAdvCoeff(3, y2C));

  return 0;
} //main()

