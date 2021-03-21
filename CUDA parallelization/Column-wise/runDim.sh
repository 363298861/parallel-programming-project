#!bin/bash

#declare -a DIM=("2,2" "2,4" "2,6" "2,8" "2,10" "2,12" "2,14" "2,16" "2,18" "2,20" "2,22" "2,24" "2,26" "2,28" "2,30" "2,32" "2,34"  "2,36"  "2,38" "2,40")
declare -a DIM=("4,32" "8,16" "16,8" "32,4" "64,2" "128,1")
GRID="12 12"
STEPS=64

#./twoDimAdvect -S ${STEPS} -t ${GRID}
#./twoDimAdvect -S ${STEPS} -t ${GRID} -c
for ((i = 0; i < ${#DIM[@]}; i++)) do
    ./twoDimAdvect -d ${DIM[i]} -S ${STEPS} ${GRID} -c
done