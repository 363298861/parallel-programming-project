#!bin/bash

STS="XX NN2 NW2 NF2 WN2 WW2 WF2 FN2 FW2 FF2 C30 C31 C32 C50 C70"
#declare -a GRID=("4 4" "5 5" "6 6" "7 7" "8 8" "9 9" "10 10" "11 11" "12 12")
GRID="10 10"
STEPS=512

#./twoDimAdvect -S ${STEPS} -t ${GRID}
#./twoDimAdvect -S ${STEPS} -t ${GRID} -c
#for ((i = 0; i < ${#GRID[@]}; i++)) do
for ST in ${STS}; do
    #./twoDimAdvect -x ${ST} ${GRID} -S ${STEPS}
    ./twoDimAdvect -x ${ST} ${GRID} -S ${STEPS} -c
done
#done
#./twoDimAdvect -S ${STEPS} -t ${GRID} 
