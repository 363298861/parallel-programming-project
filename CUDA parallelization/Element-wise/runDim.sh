#!bin/bash

declare -a DIM=("2,32" "4,16" "8,8" "16,4")
GRID="12 12"
STEPS=64

#./twoDimAdvect -S ${STEPS} -t ${GRID}
#./twoDimAdvect -S ${STEPS} -t ${GRID} -c
for ((i = 0; i < ${#DIM[@]}; i++)) do
    ./twoDimAdvect -d ${DIM[i]} -S ${STEPS} ${GRID} -c
done
