/** @file sun_hmc_ee.h */

#ifndef SUN_HMC_EE_H_
#define SUN_HMC_EE_H_

#include "hila.h"
#include "tools/string_format.h"
#include "tools/floating_point_epsilon.h"
#include "wilson_plaquette_action_ee.h"

using ftype = double;
// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct hmc_parameters {
    ftype beta;         // inverse gauge coupling
    ftype dt;           // HMC time step
    int trajlen;        // number of HMC time steps per trajectory
};

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_s(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                 atype alpha) {
    return measure_s_wplaq(U, plaq_tbc_mode, alpha);
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void update_E(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, atype alpha,
              VectorField<Algebra<T>> &E, atype delta) {
    // compute the force for the chosen action and use it to evolve E
    atype eps = delta / T::size();
    get_force_wplaq_add(U, plaq_tbc_mode, alpha, E, eps);
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void update_E2(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, int intp_dir,
              VectorField<Algebra<T>> &E, atype delta) {
    // compute the force for the chosen action and use it to evolve E
    atype eps = delta / T::size();
    VectorField<Algebra<T>> K[2];
    get_force_wplaq(U, plaq_tbc_mode, K, eps);

    atype isqrt2 = 1.0 / sqrt(2.0);
    Field<Algebra<T>> P;
    if (intp_dir>0) {
        foralldir(d) {
            onsites(ALL) {
                E[d][X] += K[0][d][X];
                P[X] = K[0][d][X] / (isqrt2 * K[0][d][X].norm());
                K[1][d][X] -= P[X].dot(K[1][d][X]) * P[X];
                E[d][X] += K[1][d][X];
            }
        }
    } else {
        foralldir(d) {
            onsites(ALL) {
                E[d][X] += K[1][d][X];
                P[X] = K[1][d][X] / (isqrt2 * K[1][d][X].norm());
                K[0][d][X] -= P[X].dot(K[0][d][X]) * P[X];
                E[d][X] += K[0][d][X];
            }
        }
    }
}

template <typename T, typename atype = hila::arithmetic_type<T>>
void update_U(GaugeField<T> &U, const VectorField<Algebra<T>> &E, atype delta) {
    // evolve U with momentum E over time step delta
    foralldir(d) {
        onsites(ALL) U[d][X] = chexp(E[d][X].expand_scaled(delta)) * U[d][X];
    }
}

template <typename T>
double measure_e2(const VectorField<Algebra<T>> &E) {
    // compute gauge kinetic energy from momentum field E
    Reduction<double> e2 = 0;
    e2.allreduce(false).delayed(true);
    foralldir(d) {
        onsites(ALL) e2 += E[d][X].squarenorm();
    }
    return e2.value() / 2;
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_action(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                      atype alpha, const VectorField<Algebra<T>> &E, const hmc_parameters &p,
                      out_only double &plaq, out_only double &e2) {
    // measure the total action, consisting of plaquette and momentum term
    plaq = p.beta * measure_s(U, plaq_tbc_mode, alpha);
    e2 = measure_e2(E) / 2;
    return plaq + e2;
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_action(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                      atype alpha, const VectorField<Algebra<T>> &E, const hmc_parameters &p) {
    // measure the total action, consisting of plaquette and momentum term
    double plaq, e2;
    return measure_action(U, plaq_tbc_mode, alpha, E, p, plaq, e2);
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_interp_hmc_trajectory(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                       VectorField<Algebra<T>> &E, const hmc_parameters &p, int interp_dir, out_only atype &ds) {
    // leap frog integration for interpolating action
    atype dalpha = 1.0 / p.trajlen; // step size of interpolating parameter
    atype alpha0 = 0;
    if(interp_dir == 0) {
        dalpha = 0;
    } else if (interp_dir < 0) {
        dalpha = -dalpha;
        alpha0 = 1.0;
    }
    // start trajectory: advance U by half a time step
    update_U(U[0], E, p.dt / 2);
    atype alpha = alpha0 + 0.5 * dalpha;
    ds = measure_ds_wplaq_dbcms(U, plaq_tbc_mode); // work done during the interpolation

    // main trajectory integration:
    for (int n = 1; n < p.trajlen; ++n) {
        update_E(U, plaq_tbc_mode, alpha, E, p.beta * p.dt);
        update_U(U[0], E, p.dt);
        alpha = alpha0 + (0.5 + (atype)n) * dalpha;
        ds += measure_ds_wplaq_dbcms(U, plaq_tbc_mode);
    }

    // end trajectory: bring U and E to the same time
    update_E(U, plaq_tbc_mode, alpha, E, p.beta * p.dt);
    update_U(U[0], E, p.dt / 2);
    ds *= p.beta * dalpha;
    U[0].reunitarize_gauge();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_interp_hmc_trajectory2(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                              VectorField<Algebra<T>> &E, const hmc_parameters &p,
                              int interp_dir, out_only atype &ds) {
    // leap frog integration for interpolating action
    atype dalpha = 1.0 / p.trajlen; // step size of interpolating parameter
    atype alpha0 = 0;
    if (interp_dir == 0) {
        dalpha = 0;
    } else if (interp_dir < 0) {
        dalpha = -dalpha;
        alpha0 = 1.0;
    }
    // start trajectory: advance U by half a time step
    update_U(U[0], E, p.dt / 2);
    atype alpha = alpha0 + 0.5 * dalpha;
    ds = measure_ds_wplaq_dbcms(U, plaq_tbc_mode); // work done during the interpolation

    // main trajectory integration:
    for (int n = 1; n < p.trajlen; ++n) {
        update_E2(U, plaq_tbc_mode, interp_dir, E, p.beta * p.dt);
        update_U(U[0], E, p.dt);
        alpha = alpha0 + (0.5 + (atype)n) * dalpha;
        ds += measure_ds_wplaq_dbcms(U, plaq_tbc_mode);
    }

    // end trajectory: bring U and E to the same time
    update_E2(U, plaq_tbc_mode, interp_dir, E, p.beta * p.dt);
    update_U(U[0], E, p.dt / 2);
    ds *= p.beta * dalpha;
    U[0].reunitarize_gauge();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_hmc_trajectory(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                       atype alpha, VectorField<Algebra<T>> &E, const hmc_parameters &p) {
    // leap frog integration for interpolating action at constant alpha

    // start trajectory: advance U by half a time step
    update_U(U[0], E, p.dt / 2);
    // main trajectory integration:
    for (int n = 0; n < p.trajlen - 1; ++n) {
        update_E(U, plaq_tbc_mode, alpha, E, p.beta * p.dt);
        update_U(U[0], E, p.dt);
    }
    // end trajectory: bring U and E to the same time
    update_E(U, plaq_tbc_mode, alpha, E, p.beta * p.dt);
    update_U(U[0], E, p.dt / 2);

    U[0].reunitarize_gauge();
}

#endif