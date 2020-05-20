#include "test.h"


#include "../plumbing/fermion/staggered.h"
#include "../plumbing/fermion/wilson.h"
#include "../plumbing/algorithms/conjugate_gradients.h"


#define N 3




void test_gamma_matrices(){
  Wilson_vector<N, double> w1, w2, w3;
  SU<N> U; U.random();
  w1.gaussian();

#if NDIM == 4
  w2 = w1-gamma5*(gamma5*w1);
  assert(w2.norm_sq() < 0.0001 && "g5*g5 = 1");
#endif

  foralldir(d){
    half_Wilson_vector<N, double> h1;
    w2 = w1-gamma_matrix[d]*(gamma_matrix[d]*w1);
    assert(w2.norm_sq() < 0.0001 && "gamma_d*gamma_d = 1");

    w2 = w1 + gamma_matrix[d]*w1;
    h1 = half_Wilson_vector(w1,d,1);
    double diff = w2.norm_sq() - h1.norm_sq();
    assert(diff*diff < 0.0001 && "half_Wilson_vector projection norm to direction XUP");

    w3 = (U*h1).expand(d,1) - U*w2;
    assert(w3.norm_sq() < 0.0001 && "half_wilson_vector expand");


    w2 = w1 - gamma_matrix[d]*w1;
    h1 = half_Wilson_vector(w1,d,-1);
    diff = w2.norm_sq() - h1.norm_sq();
    assert(diff*diff < 0.0001 && "half_Wilson_vector projection norm to direction XUP");

    w3 = (U*h1).expand(d,-1) - U*w2;
    assert(w3.norm_sq() < 0.0001 && "half_wilson_vector expand");

  }

}




int main(int argc, char **argv){

  #if NDIM==1
  lattice->setup( 64, argc, argv );
  #elif NDIM==2
  lattice->setup( 32, 8, argc, argv );
  #elif NDIM==3
  lattice->setup( 16, 8, 8, argc, argv );
  #elif NDIM==4
  lattice->setup( 8, 8, 8, 8, argc, argv );
  #endif
  seed_random(2);

  test_gamma_matrices();

  field<SU<N>> U[NDIM];
  foralldir(d) {
    onsites(ALL){
      U[d][X] = 1;
    }
  }


  // Check conjugate of the staggered Dirac operator
  {
    using dirac = dirac_staggered<SU_vector<N, double>, SU<N>>;
    dirac D(0.1, U);
    field<SU_vector<N, double>> a, b, Db, Ddaggera, DdaggerDb;
    field<SU_vector<N, double>> sol;
    onsites(ALL){
      a[X].gaussian();
      b[X].gaussian();
      sol[X] = 0;
    }

    double diffre = 0, diffim = 0;
    D.apply(b, Db);
    D.dagger(a, Ddaggera);
    onsites(ALL){
      diffre += a[X].dot(Db[X]).re - Ddaggera[X].dot(b[X]).re;
      diffim += a[X].dot(Db[X]).im - Ddaggera[X].dot(b[X]).im;
    }
  
    assert(diffre*diffre < 1e-16 && "test dirac_stagggered_dagger");
    assert(diffim*diffim < 1e-16 && "test dirac_stagggered_dagger");
    
    // Now run CG on Ddaggera. b=1/D a -> Db = a 
    CG<dirac> inverse(D);
    b[ALL] = 0;
    inverse.apply(Ddaggera, b);
    D.apply(b, Db);

    onsites(EVEN){
      diffre += norm_squared(a[X]-Db[X]);
    }
    assert(diffre*diffre < 1e-8 && "test CG");
  }

  // Check conjugate of the wilson Dirac operator
  {
    using VEC = SU_vector<N, double>;
    using dirac = dirac_wilson<N, double, SU<N, double>>;
    dirac D(0.1, U);
    field<Wilson_vector<N, double>> a, b, Db, Ddaggera, DdaggerDb;
    field<Wilson_vector<N, double>> sol;
    onsites(ALL){
      a[X].gaussian();
      b[X].gaussian();
      sol[X] = 0;
    }

    double diffre = 0, diffim = 0;
    D.apply(b, Db);
    D.dagger(a, Ddaggera);
    onsites(ALL){
      diffre += a[X].dot(Db[X]).re - Ddaggera[X].dot(b[X]).re;
      diffim += a[X].dot(Db[X]).im - Ddaggera[X].dot(b[X]).im;
    }
  
    assert(diffre*diffre < 1e-16 && "test dirac_stagggered_dagger");
    assert(diffim*diffim < 1e-16 && "test dirac_stagggered_dagger");
  
    // Now run CG on Ddaggera. b=1/D a -> Db = a 
    CG<dirac> inverse(D);
    b[ALL] = 0;
    inverse.apply(Ddaggera, b);
    D.apply(b, Db);

    onsites(EVEN){
      diffre += norm_squared(a[X]-Db[X]);
    }
    assert(diffre*diffre < 1e-8 && "test CG");
  }

  // Check conjugate of the even-odd preconditioned staggered Dirac operator
  {
    using dirac = dirac_staggered_evenodd<SU_vector<N, double>, SU<N>>;
    dirac D(0.1, U);
    field<SU_vector<N, double>> a, b, Db, Ddaggera, DdaggerDb;
    field<SU_vector<N, double>> sol;
    onsites(ALL){
      a[X].gaussian();
      b[X].gaussian();
      sol[X] = 0;
    }

    double diffre = 0, diffim = 0;
    D.apply(b, Db);
    D.dagger(a, Ddaggera);
    onsites(ALL){
      diffre += a[X].dot(Db[X]).re - Ddaggera[X].dot(b[X]).re;
      diffim += a[X].dot(Db[X]).im - Ddaggera[X].dot(b[X]).im;
    }

    assert(diffre*diffre < 1e-16 && "test dirac_stagggered_dagger");
    assert(diffim*diffim < 1e-16 && "test dirac_stagggered_dagger");
    
    // Now run CG on Ddaggera. b=1/D a -> Db = a 
    CG<dirac> inverse(D);
    b[ALL] = 0;
    inverse.apply(Ddaggera, b);
    D.apply(b, Db);

    onsites(EVEN){
      diffre += norm_squared(a[X]-Db[X]);
    }
    assert(diffre*diffre < 1e-8 && "test CG");
  }

  // Check conjugate of the even-odd preconditioned wilson Dirac operator
  {
    using VEC = SU_vector<N, double>;
    using dirac = Dirac_Wilson_evenodd<N, double, SU<N>>;
    dirac D(0.1, U);
    field<Wilson_vector<N, double>> a, b, Db, Ddaggera, DdaggerDb;
    field<Wilson_vector<N, double>> sol;

    a[ODD] = 0; b[ODD] = 0;
    onsites(EVEN){
      a[X].gaussian();
      b[X].gaussian();
      sol[X] = 0;
    }

    double diffre = 0, diffim = 0;
    D.apply(b, Db);
    D.dagger(a, Ddaggera);
    onsites(EVEN){
      diffre += a[X].dot(Db[X]).re - Ddaggera[X].dot(b[X]).re;
      diffim += a[X].dot(Db[X]).im - Ddaggera[X].dot(b[X]).im;
    }
  
    assert(diffre*diffre < 1e-16 && "test dirac_stagggered_dagger");
    assert(diffim*diffim < 1e-16 && "test dirac_stagggered_dagger");
  
    // Now run CG on Ddaggera. b=1/D a -> Db = a 
    CG<dirac> inverse(D);
    b[ALL] = 0;
    inverse.apply(Ddaggera, b);
    D.apply(b, Db);

    onsites(EVEN){
      diffre += norm_squared(a[X]-Db[X]);
    }
    assert(diffre*diffre < 1e-8 && "test CG");
  }


  finishrun();
}


