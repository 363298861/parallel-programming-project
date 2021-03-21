.SUFFIXES:
.PRECIOUS: %.o

HDRS=Advect3D.h HaloArray3D.h ProcGrid3D.h Vec3D.h Timer.h MemCorrupter.h
OBJS=Advect3D.o Advect2D.o MemCorrupter.o
PROG=twoDimAdvect
CCFLAGS= -D_BDEBUG_ -O3 -fopenmp -std=c++11 -pthread

#CCFLAGS += -D_ENABLE_ALL_STS_
CCFLAGS += -D__MEM_CORRUPTOR__

#CCFLAGS += -g
# SSE
#CCFLAGS += -mavx

# check OS for memory corruptor
#OS := $(shell uname)
#ifneq ($(OS),Darwin)
#HDRS += MemCorrupter.h
#OBJS += MemCorrupter.o
#endif

all: $(PROG)

checkCoeff: checkCoeff.o
	mpic++ -o checkCoeff checkCoeff.o

%: %.o $(OBJS)
	mpic++ -o $* $*.o $(OBJS) -lgomp -lpthread
%.o: %.cpp $(HDRS)
	mpic++ -Wall $(CCFLAGS) -c $*.cpp
clean:
	rm -f *.o $(PROG)
