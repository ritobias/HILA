
# Description 

The Hila framework consists of 

1. the transformer and 
2. a lattice simulation library.

The library is currently found in `programs/plumbing`.

Transformer contains a C++ preprocessing tool and framework for programming lattice field theory simulations: the main method for getting measurements from non-perturbative quantum field theories.  

Lattice field theory simulations involve up to 4 dimensional grids whose points are updated continuously according to some Monte Carlo update algorithm. The update at each grid point depends on the data stored at neighboring lattice points, and the ideal 
update order depends on the parity of the lattice points. Efficient parallel implementations using MPI and GPU's can be quite complicated to implement, modify and debug. Some kind of simplification was clearly needed. 

Transformer aims to make it easier for researchers to implement a broad class of these simulations by abstracting a lot of the technicalities involved, and by bringing the syntax from CUDA kernels and MPI calls to the essentials. The approach given 
here involves new datatypes and a preprocessing tool that converts c++ code with the new simplified syntax for loops and element accessors into working c++ code and gpu kernels. 

# Instructions

## Compiling the preprocessing tool and using it on c++ code

In short, the framework can be used in these steps: 

1. Write c++ code using the syntax and datatypes laid out below
2. Use the transformer excecutable to convert this code into .cpt code 
3. Compile the new .cpt code into the final excecutable 

To compile the transformer, first create a build directory inside the main directory if it doesn't exist. 
Then, compile the transformer by typing `make` in the main folder.
This will create an executable called `transformer` in the build folder.

You can then use it to compile an extended C++ file into standard C++ using
~~~ bash
build/transformer path/to/program.cpp
~~~
This will create a `cpt` file written in standard C++.

The `cpt` can be compiled with any c++ compiler, but must be linked against the headers and c++ files in the plumbing directory.

Check the example programs in the programs folder. You can use almost any standard C++ code, by there are a couple of new reserved names: the variable `X` and the function `onsites()`. In addition the framework defines a global `lattice` variable, which you should not overwrite.

In order to use the additional features for field type variables, you should inlude `plumbing/field.h` in you program. You can also include one or more of the files in the `datatypes` folder, which contains predefined datatypes that can be used to construct a field.

After this you need to define an output stream. To use `stdout`, add the line
~~~ C++
std::ostream &hila::output = std::cout;
~~~
Next, we need to initialize a lattice. The following constructs a lattice and sets it as default
~~~ C++
lattice_struct my_lattice;
lattice_struct * lattice = & my_lattice;
~~~


### Compiling on Puhti

There is a separate makefile for compiling transformer on Puhti.
To use it, run
~~~
module load gcc
make -f Makefile_puhti
~~~

This will link against the llvm installation in the hila development project folder.


## Syntax - What works

### Single line statements

You can operate on fields using statements like
~~~ C++
my_field[ALL] = my_other_field[X] + my_third_field[X];
~~~
On the left-hand side of the statement you should specify
either `[ALL]` lattice sites, `[EVEN]` sites or `[ODD]` sites.
The statement will apply only to this collection of sites.
On the right hand side, use `[X]` to refer to this collection
of sites.

You can refer to neighbouring sites by adding a direction (`XUP`, `XDOWN`, `YUP`, `YDOWN`, `ZUP`, `ZDOWN`, `TUP`, `TDOWN`, ...):
~~~ C++
my_field[EVEN] = my_field[X+YUP];
~~~

You can also operate on fields directly,
~~~ C++
my_field = my_field + 1;
~~~
This will operate on all sites and is equivalent to 
~~~ C++
my_field[ALL] = my_field[X] + 1;
~~~


### General loops 
Loops over all sites or a parity:
~~~ C++
forsites(ALL){}
forsites(EVEN){}
forsites(ODD){}
~~~
Inside the loop, refer to the sites using X:
~~~ C++
forsites(ALL){
    my_field[X] = 1;
}
~~~

As before, you can refer to neighbouring sites by adding a direction:
~~~ C++
forsites(EVEN){
    my_field[X] = my_field[X+YUP];
}
~~~



## What doesn't work (as expected)

Direction loops in forsites environments:
~~~ C++
forsites(ALL){
    foralldir(d){
        my_field[X] += my_other_field[X+d];
    }
}
~~~
## Testing

In the `programs/test_cases` folder you can find a collection of simple test programs. To test whether the translations to cpu code work, type:

~~~ bash
make test_baseline
~~~
This calls transformer without any parameters on the test programs, and outputs the exit status of each excecutable produced from them. To test whether the GPU translations work, type:

~~~ bash
make test_gpu
~~~
Or test all types of translations one after the other with:
~~~
make test
~~~




# Goals

 1. Profile and optimize the CUDA version
     * Optimize also cpu code to be fair
 1. Write a comprehensive test suite for existing and new features
 1. Compile and run and OpenACC version
 1. Extend both to support:
     * Fourier transform of field variable
     * If statements in a loop
     * Reduction on dimension (so f(t) = sum_x g(x,t))
 1. Array-of-Struct-of-Arrays layout
 1. MPI
 1. Get rid of NDIM
 1. Extend field to allow lattice as input
 1. Document the library