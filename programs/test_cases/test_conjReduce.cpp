#include "test.h"
#include<string>
/////////////////////
/// test_case 1
/// Coverage:
/// - reduction + onsites env.
/// - EVEN + ODD accessors / neighbor access
/// - field with matrix elements
/// - NOTE: Assumes periodic boundary conditions
/////////////////////

//TODO: rng definition that works for MPI, GPU and CPU

int main(int argc, char **argv){
    seed_mersenne(4l);
    double dsum = 0;
    int isum = 0;
    test_setup(argc, argv);

    // Test input class - should be possible to call before or after setup
    input a;
    a.add_essential("nx");
    a.add_essential("beta", 1.0);

    a.import("params.txt");
    int nx = a.get("nx");
    std::string out = a.get("output");

    field<matrix<2,2,double> > matrices;
    field<int> coordinate, nb_coordinate1, nb_coordinate2;
    
    //check that fieldstruct allocated only after assignment
    assert(matrices.fs==nullptr);

    // Test that neighbours are fetched correctly
    // nd is not available on device. It should be
    foralldir(dir){
        onsites(ALL){
            int nd[4] = { 20, 10, 10, 4 };
            location l = coordinates(X);
            coordinate[X] = l[dir];
            nb_coordinate1[X] = (l[dir] + 1) % nd[dir];
        }

        nb_coordinate2[ALL] = coordinate[X+dir];

        onsites(ALL){
            int diff = nb_coordinate1[X]-nb_coordinate2[X];
            isum += diff*diff;
        }
        assert(isum==0); // Value fetched from neighbour is correct
    }    

    // If MPI is defined, check that gathers are counted correctly
    #ifdef MPI
    coordinate[ALL] = 1;
    lattices[0]->n_gather_done = 0;
    lattices[0]->n_gather_avoided = 0;

    nb_coordinate2[ALL] = coordinate[X+XDOWN];
    assert(lattices[0]->n_gather_done==1);
    assert(lattices[0]->n_gather_avoided==0);
    nb_coordinate2[ALL] = coordinate[X+XDOWN];
    assert(lattices[0]->n_gather_done==1);
    assert(lattices[0]->n_gather_avoided==1);
    #endif
    
    

    // Test matrix multiplication and neighbour fetches
    // Calculates M(X) * M.congugate(X+dir)
    onsites(ALL){
        matrix<2,2,double> a;
        double theta = 2.0*M_PI*hila_random(); //make a random rotation matrix at each even site
        a.c[0][0] =  cos(theta);
        a.c[0][1] = -sin(theta);
        a.c[1][0] =  sin(theta);
        a.c[1][1] =  cos(theta);
        matrices[X] = a;
    }
    assert(matrices.fs!=nullptr);

    matrices[ODD] = matrices[X + XDOWN].conjugate(); //odd matrices are the conjugate of the previous even site in the X direction

    onsites(EVEN){
        matrices[X]*=matrices[X + XUP]; //multiply the even sites with the matrices stored in the neighboring odd site in X direction
    }
    onsites(ODD){
        matrices[X]*=matrices[X].conjugate(); //multiply odd sites by their conjugate
    }

    //now all sites should be unit matrices

    dsum=0;
    onsites(ALL){
        dsum += matrices[X].trace();
    }

    assert(((int) dsum) == lattice->volume()*2);
    
    finishrun();
}
