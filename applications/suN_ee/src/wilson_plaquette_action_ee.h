/** @file wilson_plaquette_action_ee.h */

#ifndef WILSON_PLAQUETTE_ACTION_EE_H_
#define WILSON_PLAQUETTE_ACTION_EE_H_

#include "hila.h"
#include "plaquettefield.h"

// functions for Wilson's plaquette action -S_{plaq}=\beta/N * \sum_{plaq} ReTr(plaq)

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void rstaplesum(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, atype alpha,
                out_only Field<T> &staples, Direction d1) {

    Field<T> lower[2];
    lower[1].set_nn_topo(1);

    onsites(ALL) staples[X] = 0;

    foralldir(d2) if (d2 != d1) {

        U[1][d2].start_gather(d1, ALL);
        U[0][d2].start_gather(d1, ALL);
        U[1][d1].start_gather(d2, ALL);
        U[0][d1].start_gather(d2, ALL);

        // calculate first lower 'u' of the staple sum
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                lower[1][X] = (U[1][d1][X] * U[1][d2][X + d1]).dagger() * U[1][d2][X];
            } else if(ip == 2) {
                lower[1][X] = (U[1][d1][X] * U[1][d2][X + d1]).dagger() * U[1][d2][X];
                lower[0][X] = (U[0][d1][X] * U[0][d2][X + d1]).dagger() * U[0][d2][X];
            } else {
                lower[0][X] = (U[0][d1][X] * U[0][d2][X + d1]).dagger() * U[0][d2][X];
            }
        }

        plaq_tbc_mode[d1][d2].start_gather(-d2, ALL);
        lower[1].start_gather(-d2, ALL);
        lower[0].start_gather(-d2, ALL);

        // calculate then the upper 'n'
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                staples[X] += U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger();
            } else if (ip == 2) {
                staples[X] +=
                    alpha * (U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger()) +
                    (1.0 - alpha) * (U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger());
            } else {
                staples[X] += U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger();
            }
        }

        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X - d2];
            if (ip == 1) {
                staples[X] += lower[1][X - d2];
            } else if (ip == 2) {
                staples[X] += alpha * lower[1][X - d2] + (1.0 - alpha) * lower[0][X - d2];
            } else {
                staples[X] += lower[0][X - d2];
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void drstaplesum(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, atype alpha,
                out_only Field<T> &staples, Direction d1) {

    Field<T> lower[2];
    lower[1].set_nn_topo(1);

    onsites(ALL) staples[X] = 0;

    foralldir(d2) if (d2 != d1) {

        U[1][d2].start_gather(d1, ALL);
        U[0][d2].start_gather(d1, ALL);
        U[1][d1].start_gather(d2, ALL);
        U[0][d1].start_gather(d2, ALL);

        // calculate first lower 'u' of the staple sum
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                lower[1][X] = 0;
            } else if (ip == 2) {
                lower[1][X] = (U[1][d1][X] * U[1][d2][X + d1]).dagger() * U[1][d2][X];
                lower[0][X] = (U[0][d1][X] * U[0][d2][X + d1]).dagger() * U[0][d2][X];
            } else {
                lower[0][X] = 0;
            }
        }

        plaq_tbc_mode[d1][d2].start_gather(-d2, ALL);
        lower[1].start_gather(-d2, ALL);
        lower[0].start_gather(-d2, ALL);

        // calculate then the upper 'n'
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 2) {
                staples[X] += U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger() -
                              U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger();
            }
        }

        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X - d2];
            if (ip == 2) {
                staples[X] += lower[1][X - d2] - lower[0][X - d2];
            } 
        }
    }
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void rstaplesum(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                Field<T> (out_only &staples)[2], Direction d1) {

    Field<T> lower[2];
    lower[1].set_nn_topo(1);

    onsites(ALL) {
        staples[0][X] = 0;
        staples[1][X] = 0;
    }

    foralldir(d2) if (d2 != d1) {

        U[1][d2].start_gather(d1, ALL);
        U[0][d2].start_gather(d1, ALL);

        U[1][d1].start_gather(d2, ALL);
        U[0][d1].start_gather(d2, ALL);

        // calculate first lower 'u' of the staple sum
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                lower[1][X] = (U[1][d1][X] * U[1][d2][X + d1]).dagger() * U[1][d2][X];
            } else if (ip == 2) {
                lower[1][X] = (U[1][d1][X] * U[1][d2][X + d1]).dagger() * U[1][d2][X];
                lower[0][X] = (U[0][d1][X] * U[0][d2][X + d1]).dagger() * U[0][d2][X];
            } else {
                lower[0][X] = (U[0][d1][X] * U[0][d2][X + d1]).dagger() * U[0][d2][X];
            }
        }

        plaq_tbc_mode[d1][d2].start_gather(-d2, ALL);
        lower[1].start_gather(-d2, ALL);
        lower[0].start_gather(-d2, ALL);

        // calculate then the upper 'n'
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                T tstaple = U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger();
                staples[1][X] += tstaple;
                staples[0][X] += tstaple;
            } else if (ip == 2) {
                staples[1][X] += (U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger());
                staples[0][X] += (U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger());
            } else {
                T tstaple = U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger();
                staples[1][X] += tstaple;
                staples[0][X] += tstaple;
            }
        }

        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X - d2];
            if (ip == 1) {
                T tstaple = lower[1][X - d2];
                staples[1][X] += tstaple;
                staples[0][X] += tstaple;
            } else if (ip == 2) {
                staples[1][X] += lower[1][X - d2];
                staples[0][X] += lower[0][X - d2];
            } else {
                T tstaple = lower[0][X - d2];
                staples[1][X] += tstaple;
                staples[0][X] += tstaple;
            }
        }
    }
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_s_wplaq(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                       atype alpha) {
    // measure the total Wilson plaquette action for the gauge field U
    Reduction<double> plaq = 0;
    plaq.allreduce(false).delayed(true);
    foralldir(d1) foralldir(d2) if (d1 < d2) {
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if(ip == 1) {
                plaq += 1.0 - real(trace(U[1][d1][X] * U[1][d2][X + d1] *
                                         (U[1][d2][X] * U[1][d1][X + d2]).dagger())) /
                                  T::size();
            } else if (ip == 2) {
                plaq +=
                    1.0 - (alpha * real(trace(U[1][d1][X] * U[1][d2][X + d1] *
                                              (U[1][d2][X] * U[1][d1][X + d2]).dagger())) +
                           (1.0 - alpha) * real(trace(U[0][d1][X] * U[0][d2][X + d1] *
                                                      (U[0][d2][X] * U[0][d1][X + d2]).dagger()))) /
                              T::size();
            } else {
                plaq += 1.0 - real(trace(U[0][d1][X] * U[0][d2][X + d1] *
                                         (U[0][d2][X] * U[0][d1][X + d2]).dagger())) /
                                  T::size();
            }
        }
    }
    return plaq.value();
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_s_wplaq(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                       atype alpha, out_only atype &max_plaq) {
    // measure the total and maximal Wilson plaquette action for the gauge field U
    Reduction<double> plaq = 0;
    max_plaq = -1.0;
    Field<atype> P;
    plaq.allreduce(false).delayed(true);
    foralldir(d1) foralldir(d2) if (d1 < d2) {
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                P[X] = 1.0 - real(trace(U[1][d1][X] * U[1][d2][X + d1] *
                                         (U[1][d2][X] * U[1][d1][X + d2]).dagger())) /
                                  T::size();
            } else if (ip == 2) {
                P[X] =
                    1.0 - (alpha * real(trace(U[1][d1][X] * U[1][d2][X + d1] *
                                              (U[1][d2][X] * U[1][d1][X + d2]).dagger())) +
                           (1.0 - alpha) * real(trace(U[0][d1][X] * U[0][d2][X + d1] *
                                                      (U[0][d2][X] * U[0][d1][X + d2]).dagger()))) /
                              T::size();
            } else {
                P[X] = 1.0 - real(trace(U[0][d1][X] * U[0][d2][X + d1] *
                                         (U[0][d2][X] * U[0][d1][X + d2]).dagger())) /
                                  T::size();
            }
            plaq += (double)P[X];
        }
        atype tmax_plaq = P.max();
        if (tmax_plaq > max_plaq) {
            max_plaq = tmax_plaq;
        }
    }
    return plaq.value();
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_ds_wplaq_dbcms(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode) {
    // measure plaquette action difference for the two different tbcs for plaquettes with
    // plaq_tbc_mode=2
    Reduction<double> dplaq = 0;
    dplaq.allreduce(true).delayed(true);
    Direction d2 = e_t;
    foralldir(d1) if (d1 < d2) {
        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 2) {
                dplaq += (real(trace(U[0][d1][X] * U[0][d2][X + d1] *
                                     (U[0][d2][X] * U[0][d1][X + d2]).dagger())) -
                          real(trace(U[1][d1][X] * U[1][d2][X + d1] *
                                     (U[1][d2][X] * U[1][d1][X + d2]).dagger()))) /
                         T::size();
            }
        }
    }
    return dplaq.value();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void get_force_wplaq_add(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                         atype alpha, VectorField<Algebra<T>> &K, atype eps = 1.0) {
    // compute the force for the plaquette action and write it to K
    Field<T> staple;
    foralldir(d) {
        rstaplesum(U, plaq_tbc_mode, alpha, staple, d);
        onsites(ALL) {
            K[d][X] -= (U[0][d][X] * staple[X]).project_to_algebra_scaled(eps);
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void get_force_wplaq(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                     atype alpha, out_only VectorField<Algebra<T>> &K, atype eps = 1.0) {
    // compute the force for the plaquette action and write it to K
    Field<T> staple;
    foralldir(d) {
        rstaplesum(U, plaq_tbc_mode, alpha, staple, d);
        onsites(ALL) {
            K[d][X] = (U[0][d][X] * staple[X]).project_to_algebra_scaled(-eps);
        }
    }
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void get_force_wplaq_add(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                         VectorField<Algebra<T>> (&K)[2], atype eps = 1.0) {
    // compute the force for the plaquette action and write it to K
    Field<T> staple[2];
    foralldir(d) {
        rstaplesum(U, plaq_tbc_mode, staple, d);
        onsites(ALL) {
            K[0][d][X] -= (U[0][d][X] * staple[0][X]).project_to_algebra_scaled(eps);
        }
        onsites(ALL) {
            K[1][d][X] -= (U[1][d][X] * staple[1][X]).project_to_algebra_scaled(eps);
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void get_force_wplaq(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                     VectorField<Algebra<T>>(out_only &K)[2], atype eps = 1.0) {
    // compute the force for the plaquette action and write it to K
    Field<T> staple[2];
    foralldir(d) {
        rstaplesum(U, plaq_tbc_mode, staple, d);
        onsites(ALL) {
            K[0][d][X] = (U[0][d][X] * staple[0][X]).project_to_algebra_scaled(-eps);
        }
        onsites(ALL) {
            K[1][d][X] = (U[1][d][X] * staple[1][X]).project_to_algebra_scaled(-eps);
        }
    }
}

#endif