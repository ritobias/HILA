/** @file gradient_flow_ee.h */

#ifndef GRADIENT_FLOW_EE_H_
#define GRADIENT_FLOW_EE_H_

#include "hila.h"
#include "wilson_plaquette_action_ee.h"
#include "tools/string_format.h"


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void get_gf_force(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, atype alpha,
                  out_only VectorField<Algebra<T>> &E) {
    // wrapper for force computation routine to be used for gradient flow

    atype eps = 1.0; // in principle need factor 2.0 here to switch from unoriented to oriented
                     // plaquettes (factor is usually absorbed in \beta, but gradient flow force
                     // is computed from action term with \beta-factor stripped off)
                     // however; it seems that in practice factor 1.0 is used.
                     // note: when switching to factor 2.0, remember to change also the stability
                     // limit in the do_gradient_flow_adapt() below


    //get_force_wplaq(U, plaq_tbc_mode, alpha, E, eps);
    VectorField<Algebra<T>> K[2];
    get_force_wplaq(U, plaq_tbc_mode, K, eps);

    atype isqrt2 = 1.0 / sqrt(2.0);
    Field<Algebra<T>> P;
    if(alpha == 0) {
        foralldir(d) {
            onsites(ALL) {
                E[d][X] = K[1][d][X];
                P[X] = K[0][d][X] / (isqrt2 * K[0][d][X].norm());
                E[d][X] -= P[X].dot(E[d][X]) * P[X];
            }
        }
    } else {
        foralldir(d) {
            onsites(ALL) {
                E[d][X] = K[0][d][X];
                P[X] = K[1][d][X] / (isqrt2 * K[1][d][X].norm());
                E[d][X] -= P[X].dot(E[d][X]) * P[X];
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
atype measure_gf_s(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                   atype alpha) {
    // wrapper for gauge action computation routine
    atype res = measure_s_wplaq(U, plaq_tbc_mode, alpha);
    return res;
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
atype measure_dE_wplaq_dt(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                          atype alpha) {
    Reduction<double> de = 0;
    de.allreduce(false).delayed(true);
    VectorField<Algebra<T>> K;
    get_gf_force(U, plaq_tbc_mode, alpha, K);
    foralldir(d) {
        onsites(ALL) {
            de += -2.0 * K[d][X].dot(K[d][X]);
        }
    }
    return (atype)de.value();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void measure_topo_charge_and_energy_clover(const GaugeField<T> (&U)[2],
                                           const PlaquetteField<pT> &plaq_tbc_mode, atype alpha,
                                           out_only atype &qtopo_out, out_only atype &energy_out) {
    // measure topological charge and field strength energy of the gauge field, using the
    // clover matrices as components of the field strength tensor

    Reduction<double> qtopo = 0;
    Reduction<double> energy = 0;
    qtopo.allreduce(false).delayed(true);
    energy.allreduce(false).delayed(true);

#if NDIM == 4
    Field<T> F[6];
    // F[0]: F[0][1], F[1]: F[0][2], F[2]: F[0][3],
    // F[3]: F[1][2], F[4]: F[1][3], F[5]: F[2][3]

    Field<T> tF0;

    Field<T> tF1[2];
    tF1[1].set_nn_topo(1);

    int k = 0;
    foralldir(d1) foralldir(d2) if (d1 < d2) {
        onsites(ALL) {
            // d1-d2-plaquette that starts and ends at X; corresponds to F[d1][d2]
            // at center location X+d1/2+d2/2 of plaquette:
            int ip = plaq_tbc_mode[d1][d2][X];
            T tF;
            if (ip == 1) {
                tF = U[1][d1][X] * U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger();
                // project to Lie-algebra (anti-hermitian trace-free)
                tF -= tF.dagger();
                tF *= 0.5;
                tF -= trace(tF) / T::size();
                tF0[X] = tF;
                // parallel transport to X+d2
                tF1[1][X] = U[1][d2][X].dagger() * tF * U[1][d2][X];
            } else if (ip == 2) {
                tF = U[1][d1][X] * U[1][d2][X + d1] * (U[1][d2][X] * U[1][d1][X + d2]).dagger();
                // project to Lie-algebra (anti-hermitian trace-free)
                tF -= tF.dagger();
                tF *= 0.5;
                tF -= trace(tF) / T::size();
                tF0[X] = alpha * tF;
                // prepare parallel transport to X+d2
                tF1[1][X] = U[1][d2][X].dagger() * tF * U[1][d2][X];

                tF = U[0][d1][X] * U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger();
                // project to Lie-algebra (anti-hermitian trace-free)
                tF -= tF.dagger();
                tF *= 0.5;
                tF -= trace(tF) / T::size();
                tF0[X] += (1.0 - alpha) * tF;
                // prepare parallel transport to X+d2
                tF1[0][X] = U[0][d2][X].dagger() * tF * U[0][d2][X];

            } else {
                tF = U[0][d1][X] * U[0][d2][X + d1] * (U[0][d2][X] * U[0][d1][X + d2]).dagger();
                // project to Lie-algebra (anti-hermitian trace-free)
                tF -= tF.dagger();
                tF *= 0.5;
                tF -= trace(tF) / T::size();
                tF0[X] = tF;
                // prepare parallel transport to X+d2
                tF1[0][X] = U[0][d2][X].dagger() * tF * U[0][d2][X];
            }
        }

        onsites(ALL) {
            int ip = plaq_tbc_mode[d1][d2][X - d2];
            if (ip == 1) {
                tF0[X] += tF1[1][X - d2];
            } else if (ip == 2) {
                tF0[X] += alpha * tF1[1][X - d2] + (1.0 - alpha) * tF1[0][X - d2];
            } else {
                tF0[X] += tF1[0][X - d2];
            }
        }

        onsites(ALL) {
            // get F[d1][d2] at X from average of the (parallel transported) F[d1][d2] from
            // the centers of all d1-d2-plaquettes that touch X :
            // (note: d1 is always spatial, temporal boundary conditions therefore don't matter)
            F[k][X] =
                (tF0[X] + U[0][d1][X - d1].dagger() * tF0[X - d1] * U[0][d1][X - d1]) * 0.25;
        }
        ++k;
    }

    onsites(ALL) {
        qtopo += real(mul_trace(F[0][X], F[5][X]));
        qtopo += -real(mul_trace(F[1][X], F[4][X]));
        qtopo += real(mul_trace(F[2][X], F[3][X]));

        energy += F[0][X].squarenorm();
        energy += F[1][X].squarenorm();
        energy += F[2][X].squarenorm();
        energy += F[3][X].squarenorm();
        energy += F[4][X].squarenorm();
        energy += F[5][X].squarenorm();
    }
#endif
    qtopo_out = (atype)qtopo.value() / (4.0 * M_PI * M_PI);
    energy_out = (atype)energy.value();
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void measure_gradient_flow_stuff(const GaugeField<T> (&V)[2],
                                 const PlaquetteField<pT> &plaq_tbc_mode, atype alpha, atype flow_l,
                                 atype t_step, int s) {
    // perform measurements on flowed gauge configuration V at flow scale flow_l
    // [t_step is the flow time integration step size used in last gradient flow step]
    // and print results in formatted form to standard output
    static bool first = true;
    if (first) {
        // print info about used flow action
        hila::out0 << "GFINFO using Wilson's plaquette action\n";
        // print legend for flow measurement output
        hila::out0 << "LGFLMEAS  l(ambda)        S-plaq       dS-plaq        E_plaq    dE_plaq/dl  "
                      "       E_clv     Qtopo_clv   [t_step_size]   [max_S-plaq]\n";
        first = false;
    }
    atype max_plaq = 0;
    atype plaq = measure_s_wplaq(V, plaq_tbc_mode, alpha, max_plaq) /
                 (lattice.volume() * NDIM * (NDIM - 1) / 2); // average wilson plaquette action
    atype dplaqs = measure_ds_wplaq_dbcms(V, plaq_tbc_mode) / (s * (NDIM - 1));

    atype eplaq = plaq * NDIM * (NDIM - 1) * T::size(); // naive energy density (plaq. action)

    // average energy density and toplogical charge from
    // clover definition of field strength tensor :
    atype qtopocl, ecl;
    measure_topo_charge_and_energy_clover(V, plaq_tbc_mode, alpha, qtopocl, ecl);
    ecl /= lattice.volume();


    // derivative of plaquette energy density w.r.t. to flow time :
    atype deplaqdt = measure_dE_wplaq_dt(V, plaq_tbc_mode, alpha) / lattice.volume();

    // print formatted results to standard output :
    hila::out0 << string_format(
        "GFLMEAS  % 9.3f % 0.6e % 0.6e % 0.6e % 0.6e % 0.6e % 0.6e     [%0.3e]    [%0.3e]\n",
        flow_l, plaq, dplaqs, eplaq, 0.25 * flow_l * deplaqdt, ecl, qtopocl, t_step, max_plaq);
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
atype do_gradient_flow_adapt(GaugeField<T> (&V)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                             atype alpha, atype l_start, atype l_end, atype atol = 1.0e-6,
                             atype rtol = 1.0e-4, atype tstep = 0.0) {
    // wilson flow integration from flow scale l_start to l_end using 3rd order
    // 3-step Runge-Kutta (RK3) from arXiv:1006.4518 (cf. appendix C of
    // arXiv:2101.05320 for derivation of this Runge-Kutta method)
    // and embedded RK2 for adaptive step size

    atype esp = 3.0; // expected single step error scaling power: err ~ step^(esp)
                     //   - for RK3 with embedded RK2: esp \approx 3.0
    atype iesp = 1.0 / esp; // inverse single step error scaling power

    atype stepmf = 1.0;
    atype maxstepmf = 10.0;  // max. growth factor of adaptive step size
    atype minstepmf = 0.1; // min. growth factor of adaptive step size

    // translate flow scale interval [l_start,l_end] to corresponding
    // flow time interval [t,tmax] :
    atype t = l_start * l_start / 8.0;
    atype tmax = l_end * l_end / 8.0;

    atype ubstep = (tmax - t) / 2.0; // max. allowed time step

    atype tatol = atol * sqrt(2.0);

    // hila::out0<<"t: "<<t<<" , tmax: "<<tmax<<" , step: "<<tstep<<" , minmaxreldiff:
    // "<<minmaxreldiff<<"\n";

    // temporary variables :
    VectorField<Algebra<T>> k1, k2, tk;
    GaugeField<T> V2, V0;
    Field<atype> reldiff;

    // RK3 coefficients from arXiv:1006.4518 :
    // correspond to standard RK3 with Butcher-tableau
    // (cf. arXiv:2101.05320, Appendix C)
    //  0  |   0     0     0
    //  #  |  1/4    0     0
    //  #  | -2/9   8/9    0
    // -------------------------
    //     |  1/4    0    3/4
    //
    atype a11 = 0.25;
    atype a21 = -17.0 / 36.0, a22 = 8.0 / 9.0;
    atype a33 = 0.75;

    // RK2 coefficients :
    // cf. Alg. 6 and Eqn. (13)-(14) in arXiv:2101.05320 to see
    // how these are obtained from standard RK2 with Butcher-tableau
    //  0  |   0     0
    //  #  |  1/4    0
    // -----------------
    //     |  -1     2
    //
    atype b21 = -1.25, b22 = 2.0;

    atype step = min(tstep, ubstep); // initial step size

    if (t == 0 || step == 0) {
        // when using a gauge action for gradient flow that is different from
        // the one used to sample the gauge cofingurations, the initial force
        // can be huge. Therefore, if no t_step is provided as input, the inital
        // value for step is here adjustet so that
        // step * <largest local force> = maxstk
        atype maxstk = 1.0e-1;

        // get max. local gauge force:
        get_gf_force(V, plaq_tbc_mode, alpha, tk);
        atype maxtk = 0.0;
        foralldir(d) {
            onsites(ALL) {
                reldiff[X] = (tk[d][X].squarenorm());
            }
            atype tmaxtk = reldiff.max();
            if(tmaxtk>maxtk) {
                maxtk = tmaxtk;
            }
        }
        maxtk = sqrt(0.5 * maxtk);

        if (step == 0) {
            if (maxtk > maxstk) {
                step = min(maxstk / maxtk,
                           ubstep); // adjust initial step size based on max. force magnitude
                hila::out0 << "GFINFO using max. gauge force (max_X |F(X)|=" << maxtk
                           << ") to set initial flow time step size: " << step << "\n";
            } else {
                step = min((atype)1.0, ubstep);
            }
        } else if (step * maxtk > maxstk) {
            step = min(maxstk / maxtk,
                       ubstep); // adjust initial step size based on max. force magnitude
            hila::out0 << "GFINFO using max. gauge force (max_X |F(X)|=" << maxtk
                       << ") to set initial flow time step size: " << step << "\n";
        }
    }


    V0 = V[0];
    bool stop = false;
    while (t < tmax && !stop) {
        tstep = step;
        if (t + step >= tmax) {
            step = tmax - t;
            stop = true;
        }

        get_gf_force(V, plaq_tbc_mode, alpha, k1);
        foralldir(d) onsites(ALL) {
            // first steps of RK3 and RK2 are the same :
            V[0][d][X] = chexp(k1[d][X] * (step * a11)) * V[0][d][X];
        }

        get_gf_force(V, plaq_tbc_mode, alpha, k2);
        foralldir(d) onsites(ALL) {
            // second step of RK2 :
            // (tk[d][X] will be used for rel. error computation)
            tk[d][X] = k2[d][X];
            tk[d][X] *= (step * b22);
            tk[d][X] += k1[d][X] * (step * b21);
            V2[d][X] = chexp(tk[d][X]) * V[0][d][X];

            // second step of RK3 :
            k2[d][X] *= (step * a22);
            k2[d][X] += k1[d][X] * (step * a21);
            V[0][d][X] = chexp(k2[d][X]) * V[0][d][X];
        }

        get_gf_force(V, plaq_tbc_mode, alpha, k1);
        foralldir(d) onsites(ALL) {
            // third step of RK3 :
            k1[d][X] *= (step * a33);
            k1[d][X] -= k2[d][X];
            V[0][d][X] = chexp(k1[d][X]) * V[0][d][X];
        }

        // determine maximum difference between RK3 and RK2,
        // relative to desired accuracy :
        atype relerr = 0.0;
        foralldir(d) {
            onsites(ALL) {
                reldiff[X] = (V2[d][X] * V[0][d][X].dagger()).project_to_algebra().norm() /
                             (tatol + rtol * tk[d][X].norm() / step);
                // note: we divide tk.norm() by step to have consistent leading stepsize dependency  
                // no mather whether relative or absolute error tollerance dominates
            }
            atype trelerr = reldiff.max();
            if (trelerr > relerr) {
                relerr = trelerr;
            }
        }

        if (relerr < 1.0) {
            // proceed to next iteration
            t += step;
            V[0].reunitarize_gauge();
            V0 = V[0];
        } else {
            // repeat current iteration if single step error was too large
            V[0] = V0;
            stop = false;
        }

        // determine step size to achieve desired accuracy goal :
        stepmf = pow(relerr, -iesp);
        if (stepmf <= minstepmf) {
            stepmf = minstepmf;
        } else if (stepmf >= maxstepmf) {
            stepmf = maxstepmf;
        } 

        // adjust step size :
        step = min((atype)0.9 * stepmf * step, ubstep);
    }

    return tstep;
}

#endif