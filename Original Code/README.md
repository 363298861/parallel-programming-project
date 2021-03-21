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


    -H number(1-3): specifies the size of boundary of halo array (default: 3)
    eg) -H 3

    -t: use TMR (default: no)

    -x stencil: select a stencil to use. cannot select multiple stencils. 
    stencil: XX, NN2, NW2, NF2, WN2, WW2, WF2, FN2, FW2, FF2, C30, C32, C50, C70 (default: C50) 
    eg) -x XX, -x NF2, ...
 
    -S steps: the number of iterations

    [g.x, g.y]: the data grid size (default 6 6)
    eg) 10 10
    
If we are using a 10 * 10 data grid 256 steps on stencil FW2, the command is
```bash
./twoDimAdvect 10 10 -S 256 -x FW2
```

The commands that generated the results in the report are in file "runStencils.sh", use
```bash
bash runStencils.sh
```
to execute them.
    
    