#!bin/bash

declare -a GRID=("4 4" "5 5" "6 6" "7 7" "8 8" "9 9" "10 10" "11 11" "12 12" "13 13" "14 14")
#GRID="10 10"
STEPS=64

#./twoDimAdvect -S ${STEPS} -t ${GRID}
#./twoDimAdvect -S ${STEPS} -t ${GRID} -c
for ((i = 0; i < ${#GRID[@]}; i++)) do
    ./twoDimAdvect ${GRID[i]} -S ${STEPS}
    ./twoDimAdvect ${GRID[i]} -S ${STEPS} -c
done
#done
#./twoDimAdvect -S ${STEPS} -t ${GRID}
