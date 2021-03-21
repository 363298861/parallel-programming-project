.SUFFIXES:
.PRECIOUS: %.o

NVCC=nvcc

HDRS=Advect3D.h HaloArray3D.h Vec3D.h AdvectCoeff.h gputimer.h
OBJS=Advect3D.o Advect2D.o AdvectCuda.o
PROG=twoDimAdvect
CCFLAGS=-O3

all: $(PROG)
%: %.o $(OBJS)
	$(NVCC) -o $* $*.o $(OBJS)
%.o: %.cpp $(HDRS)
	$(NVCC) $(CCFLAGS) -c $*.cpp
%.o: %.cu $(HDRS)
	$(NVCC) $(CCFLAGS) -c $*.cu
clean:
	rm -f *.o $(PROG)
