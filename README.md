# Parallelizing a 2D advection solver on both CPU and GPU

**Author: Zhiyuan Huang u6656110**

**Supervisor: Prof. Peter Strazdins**


## Project Description

This is the project repo for COMP8755 individual project. In this project, we are parallelizing a 2D advection solver using OpenMP and CUDA and investigate their performance.


## Files Description

In this repo, there are three dierctories and one report. The report is the final project report. For more details about this project, please read the report.

The program in the "Original code" directory is the orignal program without any modification. There is a README file in the directory to descripe how to execute this program.

The program in the "OMP parallelization" directory is the code that is parallelized by OpenMP. Again, there is a README file in the directory to specify how to execute.

The programs in the "CUDA parallelization" directory is the code that is parallelized by CUDA. There are four directories each of which is one implementation of CUDA parallelization. Running details are in the directory.
The code in this directory is quite different from code in the other directories because I rewrite the program to make sure it could be compiled and executed by CUDA.
