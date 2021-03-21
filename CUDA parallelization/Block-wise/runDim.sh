#!bin/bash

declare -a DIM=("4,32" "8,16" "16,8" "32,4" "64,2" "128,1")
GRID="12 12"
STEPS=64

#./twoDimAdvect -S ${STEPS} -t ${GRID}
#./twoDimAdvect -S ${STEPS} -t ${GRID} -c
for ((i = 0; i < ${#DIM[@]}; i++)) do
    ./twoDimAdvect -d ${DIM[i]} -S ${STEPS} ${GRID} -c
done
