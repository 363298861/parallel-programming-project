#!/bin/bash

STS="XX NN2 NW2 NF2 WN2 WW2 WF2 FN2 FW2 FF2 C30 C31 C32 C50"
threads="1 2 4 8 16 32"
GRID="10 10"
STEPS=512

./twoDimAdvect -S ${STEPS} -t ${GRID} #TMR
for ST in ${STS}; do
  for thread in ${threads}; do
    echo ./twoDimAdvect -S ${STEPS} -x ${ST} ${GRID} -P ${thread}
    ./twoDimAdvect -S ${STEPS} -x ${ST} ${GRID} -P ${thread}
  done
done				   
    
