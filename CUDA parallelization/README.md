These are cuda parallelized program, each directory is one implementation. Their execution commands are very similar.
I rewrite CUDA version program so that the program can execute on GPU. However, most of the code remain unchanged.
I only add one file called "AdvectCuda.cu", which includs all the CUDA program and kernels. 

To be able to execute this program, first use make to compile the program.

```bash
make
```

After execution, you can use "make clean" to remove all binary files

```bash
make clean
```

The command to execute this program is

```bash
./twoDimAdvect 
```

Here are the parameters you can choose:

    -t: use TMR (default: no)
    
    -c: use GPU to execute program in parallel (default: no)
    if you are going to execute this in parallel, remember to add "-c" in  the parameters, otherwise program will run sequentially on CPU

    -x stencil: select a stencil to use. cannot select multiple stencils. 
    stencil: XX, NN2, NW2, NF2, WN2, WW2, WF2, FN2, FW2, FF2, C30, C32, C50, C70 (default: C50) 
    eg) -x XX, -x NF2, ...
 
    -S steps: the number of iterations

    [g.x, g.y]: the data grid size (default 6 6)
    eg) 10 10
    
    -d thread dimension: choose the thread dimension to execute the program, the first number is the 
    grid size in x and y dimension and the second number is the thread size in each block in x and y dimension. 
    This will not work if you didn't have "-c" in the command. (default 2,32)
    eg) -d 2,8 (means 2*2 thread blocks and 8*8 threads per block, which is 256 threads in total)
    
    -z block size: specify the data block size in the block-wise implementation. In that implementation, one thread will calculate all elements in that block.
    note that this only works in block-wise implementation. Get error if used in other implementations.
    
If we are using a 10 * 10 data grid 256 steps on stencil FW2 with (2*2)*(32*32) = 4096 threads execute in parallel, the command is
```bash
./twoDimAdvect 10 10 -S 256 -x FW2 -d 2,32 -c
```

The commands that generated the results in the report are in file "runStencils.sh", use
```bash
bash runStencils.sh
```
to execute all.

In element-wise implementation, I also investigated the impact of the thread dimensions, the commands are in file "runDim.sh".
```bash
bash runDim.sh
```