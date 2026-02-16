#include "hila.h"
#include "tools/string_format.h"
#include "tools/floating_point_epsilon.h"
#include <algorithm>

#ifndef NCOLOR
#define NCOLOR 4
#endif

using ftype = double;
using fT = Vector<NCOLOR, ftype>;
using fTf = Vector<NCOLOR, float>;

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    ftype kappa;        // hopping parameter
    ftype lambda;       // quartic coupling (lattice definition)
    fT source;          // source terms
    int s;              // number of replicas
    ftype l;            // momentum space entangling region slab width l
    ftype lc;           // momentum space entangling region slab width lc
    ftype alpha;        // interpolation parameter between l and lc
    ftype dalpha;
    ftype bcms;
    int n_traj;         // number of trajectories to generate
    int n_therm;        // number of thermalization trajectories (counts only accepted traj.)
    int n_heatbath;     // number of heat-bath sweeps per "trajectory"
    int n_multhits;     // number of corrected heat-bath hits per update
    int n_interp_steps; // number of interpolation steps to change alpha from 0 to 1
    int n_dump_corr;    // number of trajectories between correlator and condensate dumps
    int n_dump_mom_corr;// number of trajectories between momentum mode correlator dumps
    int n_dump_config;  // number of trajectories between config. dumps
    int n_save;         // number of trajectories between config. check point
    std::string config_file;
    ftype time_offset;
};


template <typename T, typename iT, typename atype = hila::arithmetic_type<T>>
void bwrite_to_file(std::string fname, const std::vector<T> &dat, parameters &p, iT idump, bool first) {

    if (hila::myrank() == 0) {
        std::ofstream ofile;

        if (first) {
            first = false;
            if (filesys_ns::exists(fname)) {
                // if file exists: make sure 
                std::string fname_temp = fname + "_temp";
                filesys_ns::rename(fname, fname_temp);
                std::ifstream ifile;
                ifile.open(fname_temp, std::ios::in | std::ios::binary);
                ofile.open(fname, std::ios::out | std::ios::binary);

                // header info:
                // NCOLOR,
                // NDIM,
                // lattice.size(0), ..., lattice.size(NDIM - 1),
                // sizeof(double),
                // dat.size(),
                // sizeof(T),
                // sizeof(atype),
                // p.s,
                // p.kappa,
                // p.lambda,
                // p.source(0), ..., p.source(NCOLOR -1),
                // p.l,
                // p.lc,
                // p.alpha,
                // p.dalpha
                //
                int64_t *ibuff = (int64_t *)memalloc((NDIM + 7) * sizeof(int64_t));
                ifile.read((char *)ibuff, (NDIM + 7) * sizeof(int64_t));
                ofile.write((char *)ibuff, (NDIM + 7) * sizeof(int64_t));

                double *fbuff = (double *)memalloc((NCOLOR + 6) * sizeof(double));
                ifile.read((char *)fbuff, (NCOLOR + 6) * sizeof(double));
                ofile.write((char *)fbuff, (NCOLOR + 6) * sizeof(double));

                int buffsize = dat.size() * sizeof(T);
                atype *buffer = (atype *)memalloc(buffsize);

                while (ifile.good()) {
                    ifile.read((char *)ibuff, sizeof(int64_t));
                    int64_t tidump = ibuff[0];
                    if (ifile.good() && tidump < idump) {
                        ofile.write((char *)&(tidump), sizeof(int64_t));
                        ifile.read((char *)buffer, buffsize);
                        ofile.write((char *)buffer, buffsize);
                    } else {
                        break;
                    }
                }
                ifile.close();
                ofile.close();
                free(fbuff);
                free(ibuff);
                free(buffer);
                filesys_ns::remove(fname_temp);
            } else {
                // header info:
                // NCOLOR,
                // NDIM,
                // lattice.size(0), ..., lattice.size(NDIM - 1),
                // sizeof(double),
                // dat.size(),
                // sizeof(T),
                // sizeof(atype),
                // p.s,
                // p.kappa,
                // p.lambda,
                // p.source(0), ..., p.source(NCOLOR -1),
                // p.l,
                // p.lc,
                // p.alpha,
                // p.dalpha
                //
                ofile.open(fname, std::ios::out | std::ios::binary);
                int64_t *ibuff = (int64_t *)memalloc((NDIM + 7) * sizeof(int64_t));
                ibuff[0] = NCOLOR;
                ibuff[1] = NDIM;
                foralldir(d) {
                    ibuff[2 + (int)d] = lattice.size(d);
                }
                ibuff[NDIM + 2] = sizeof(double);
                ibuff[NDIM + 3] = dat.size();
                ibuff[NDIM + 4] = sizeof(T);
                ibuff[NDIM + 5] = sizeof(atype);
                ibuff[NDIM + 6] = p.s;
                ofile.write((char *)ibuff, (NDIM + 7) * sizeof(int64_t));

                double *fbuff = (double *)memalloc((NCOLOR + 6) * sizeof(double));
                fbuff[0] = p.kappa;
                fbuff[1] = p.lambda;
                for (int ic = 0; ic < NCOLOR; ++ic) {
                    fbuff[2 + ic] = p.source[ic];
                }
                fbuff[NCOLOR + 2] = p.l;
                fbuff[NCOLOR + 3] = p.lc;
                fbuff[NCOLOR + 4] = p.alpha;
                fbuff[NCOLOR + 5] = p.dalpha;
                ofile.write((char *)fbuff, (NCOLOR + 6) * sizeof(double));


                ofile.close();
                free(fbuff);
                free(ibuff);
            }
        }


        ofile.open(fname, std::ios::out | std::ios_base::app | std::ios::binary);
        int64_t tidump = (int64_t)idump;
        ofile.write((char *)&(tidump), sizeof(int64_t));

        for (int ic = 0; ic < dat.size(); ++ic) {
            ofile.write((char *)&(dat[ic]), sizeof(T));
        }

        ofile.close();

    }
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered_p(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d, const Parity &par,
                     bool both_dirs, out_only Field<T> &Sd, const parameters &p) {
    // pull fields from neighboring sites in d-direction, using different nn-topologies depending on local value of field bcmsid[X]
    // (was used only for testing)
    if (both_dirs) {
        S[1].start_gather(-d, par);
        S[0].start_gather(-d, par);
    }
    onsites(par) {
        if (bcmsid[X] <= p.bcms) {
            Sd[X] = S[1][X + d];
        } else {
            Sd[X] = S[0][X + d];
        }
    }
    if (both_dirs) {
        onsites(par) {
            if (bcmsid[X] <= p.bcms) {
                Sd[X] += S[1][X - d];
            } else {
                Sd[X] += S[0][X - d];
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered_k(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d, const Parity &par, bool both_dirs,
                   out_only Field<T> &Sd, const parameters &p) {
    // pull fields from neighboring sites in d-direction, using different nn-topologies for
    // different Fourier modes, depending on the magnitude |k_s| of their spatial k-vector k_s,
    // where |k_s| is encded in bcmsid[X].

    // define direction in which Fourier transform should be taken
    CoordinateVector fftdirs;
    foralldir(td) if(td != d) fftdirs[td] = 1;
    fftdirs[d] = 0;

    Field<Complex<atype>> tS, tSK;
    Field<Complex<atype>> SK[2];
    SK[0] = 0;
    SK[1].make_ref_to(SK[0], 1);
    double svol = lattice.volume() / lattice.size(d);
    Sd = 0;
    for (int inc = 0; inc < T::size(); inc += 2) {
        onsites(ALL) {
            T tvec = S[0][X];
            if (inc + 1 < T::size()) {
                tS[X] = Complex<atype>(tvec[inc], tvec[inc + 1]);
            } else {
                tS[X] = Complex<atype>(tvec[inc], 0);
            }
        }
        tS.FFT(fftdirs, SK[0]);
        if(both_dirs) {
            SK[1].start_gather(-d);
            SK[0].start_gather(-d);
        }
        onsites(ALL) {
            if (bcmsid[X] <= p.l) {
                tSK[X] = SK[1][X + d];
            } else {
                if (bcmsid[X] <= p.lc) {
                    tSK[X] = p.alpha * SK[1][X + d] + (1.0 - p.alpha) * SK[0][X + d];
                } else {
                    tSK[X] = SK[0][X + d];
                }
            }
        }
        if(both_dirs) {
            onsites(ALL) {
                if (bcmsid[X] <= p.l) {
                    tSK[X] += SK[1][X - d];
                } else {
                    if (bcmsid[X] <= p.lc) {
                        tSK[X] += p.alpha * SK[1][X - d] + (1.0 - p.alpha) * SK[0][X - d];
                    } else {
                        tSK[X] += SK[0][X - d];
                    }
                }
            }
        }
        tSK.FFT(fftdirs, tS, fft_direction::back);

        onsites(par) {
            Complex<atype> tc = tS[X] / svol;
            Sd[X][inc] = tc.real();
            if (inc + 1 < T::size()) {
                Sd[X][inc + 1] = tc.imag();
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d,
                   const Parity &par, bool both_dirs, out_only Field<T> &Sd, const parameters &p) {
    if (p.s > 1) {
        move_filtered_k(S, bcmsid, d, par, both_dirs, Sd, p);
    } else {
        if (both_dirs) {
            S[0].start_gather(-d, par);
        }

        onsites(par) Sd[X] = S[0][X + d];

        if (both_dirs) {
            onsites(par) Sd[X] += S[0][X - d];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////
// O(N) action
// action: S[\phi] = \sum_{x} ( -\kappa/2 \sum_{\nu}( \phi(x).\phi(x+\hat{\nu}) +
// \phi(x).\phi(x-\hat{\nu}) ) + \phi(x).\phi(x) + \lambda (\phi(x).\phi(x) - 1)^2 )
//
///////////////////////////////////////////////////////////////////////////////////
// heat-bath functions

template <typename T, typename atype = hila::arithmetic_type<T>>
void ON_heatbath(T &S, const T &nnsum, atype kappa, atype lambda, const T &source) {
    T Sn = 0;
    atype vari = 1.0 / sqrt(2.0);
    Sn.gaussian_random(vari);
    Sn += 0.5 * (kappa * nnsum + source);
    atype rSnsq = Sn.squarenorm() - 1.0;
    atype rSsq = S.squarenorm() - 1.0;
    atype texp = lambda * (rSnsq * rSnsq - rSsq * rSsq);
    if (texp<=0 || hila::random() < exp(-texp)) {
        S = Sn;
    }
}


/**
 * @brief Wrapper function to updated O(N) scalar field per paraity
 * @details --
 *
 * @tparam T field type
 * @tparam pT type of bcmsid
 * @param S[2] field for the two different nn-topologies
 * @param bcmsid field specifying nn-topology to be used to compute nn terms
 * @param p parameters
 * @param par Parity specifies parity of links to be updated
 * @param tpar temporal parity
 */
template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void hb_update_parity(Field<T>(&S)[2], const Field<pT> &bcmsid, const parameters &p, Parity par, int tpar) {

    static hila::timer hb_timer("Heatbath");

    Field<T> Sd, nnsum = 0;

    foralldir(d) {
        if (d < NDIM - 1) {
            S[0].start_gather(-d);
            onsites(par) nnsum[X] += S[0][X + d];

            onsites(par) nnsum[X] += S[0][X - d];
        } else {
            move_filtered(S, bcmsid, d, par, true, Sd, p);
            onsites(par) nnsum[X] += Sd[X];
        }
    }

    hb_timer.start();
    Direction td = Direction(NDIM - 1);
    onsites(par) {
        if(X.coordinate(td) % 2 == tpar) {
            for (int i = 0; i < p.n_multhits; ++i) {
                ON_heatbath(S[0][X], nnsum[X], p.kappa, p.lambda, p.source);
            }
        }
    }

    hb_timer.stop();
}

/**
 * @brief Wrapper update function
 * @details field update sweep with randomly chosen spatial and temporal parities 
 *
 * @tparam T field type
 * @tparam pT type of bcmsid
 * @param S[2] field for the two different nn-topologies
 * @param bcmsid field specifying nn-topology to be used to compute nn terms
 * @param p parameters
 */
template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void hb_update(Field<T> (&S)[2], const Field<pT> &bcmsid, const parameters &p) {
    std::array<int, 4> rnarr;
    for (int i = 0; i < 4; ++i) {
        // randomly choose a spatial and a time slice parity:
        rnarr[i] = (int)(hila::random() * 4);
    }
    hila::broadcast(rnarr);

    for (int i = 0; i < 4; ++i) {
        int tdp = rnarr[i];

        int ttpar = tdp / 2;
        int tspar = 1 + (tdp % 2);

        // perform the selected updates:
        hb_update_parity(S, bcmsid, p, Parity(tspar), ttpar);
    }
}

/**
 * @brief Evolve field
 * @details Evolution happens by means of a metropolis corrected heatbath algorithm. We do on average
 * p.n_heatbath heatbath updates on each site per sweep.
 *
 * @tparam T field type
 * @tparam pT type of plaq_tbc_mode
 * @param S[2] Field for the two different nn-topologies
 * @param bcmsid Field specifying nn-topology to be used to compute nn terms
 * @param p parameters
 */
template <typename T, typename pT>
void do_hb_trajectory(Field<T> (&S)[2], const Field<pT> &bcmsid, const parameters &p) {
    for (int n = 0; n < p.n_heatbath; n++) {
        hb_update(S, bcmsid, p);
    }
}


// heat-bath functions
///////////////////////////////////////////////////////////////////////////////////
// measurement functions

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_s(const Field<T> (&S)[2], const Field<pT> &bcmsid, const parameters &p, double &s_kin) {
    Reduction<double> s = 0, s_pot = 0;
    s.allreduce(false).delayed(true);
    Field<T> Sd;
    // hopping terms in all directions:
    foralldir(d) {
        if (d < NDIM - 1) {
            onsites(ALL) {
                s += -p.kappa * S[0][X].dot(S[0][X + d]);
            }
        } else {
            move_filtered(S, bcmsid, d, ALL, false, Sd, p);
            onsites(ALL) {
                s += -p.kappa * S[0][X].dot(Sd[X]);
            }
        }
    }
    // potential term:
    onsites(ALL) {
        atype S2 = S[0][X].squarenorm();
        s_pot += S2 + p.lambda * (S2 - 1.0) * (S2 - 1.0) - S[0][X].dot(p.source);
    }
    s_kin = s.value();
    return s_kin + s_pot.value();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_ds_dalpha(const Field<T> (&S)[2], const Field<pT> &bcmsid, const parameters &p) {
    // define direction in which Fourier transform should be taken
    CoordinateVector fftdirs;
    foralldir(d) if (d < NDIM - 1) fftdirs[d] = 1;
    fftdirs[NDIM - 1] = 0;

    Reduction<double> ds = 0;
    ds.allreduce(false).delayed(true);
    Field<T> Sd;

    Field<Complex<atype>> tS, tSK;
    Field<Complex<atype>> SK[2];
    SK[0] = 0;
    SK[1].make_ref_to(SK[0], 1);
    double svol = lattice.volume() / lattice.size(NDIM - 1);
    Sd = 0;

    // hopping terms in direction d
    Direction d = Direction(NDIM - 1);

    for (int inc = 0; inc < T::size(); inc += 2) {
        onsites(ALL) {
            T tvec = S[0][X];
            if (inc + 1 < T::size()) {
                tS[X] = Complex<atype>(tvec[inc], tvec[inc + 1]);
            } else {
                tS[X] = Complex<atype>(tvec[inc], 0);
            }
        }
        tS.FFT(fftdirs, SK[0]);
        onsites(ALL) {
            if (bcmsid[X] > p.l && bcmsid[X] <= p.lc) {
                tSK[X] = SK[1][X + d] - SK[0][X + d];
            } else {
                tSK[X] = 0;
            }
        }
        tSK.FFT(fftdirs, tS, fft_direction::back);
        onsites(ALL) {
            Complex<atype> tc = tS[X] / svol;
            Sd[X][inc] = tc.real();
            if (inc + 1 < T::size()) {
                Sd[X][inc + 1] = tc.imag();
            }
        }
    }

    onsites(ALL) {
        ds += -p.kappa * S[0][X].dot(Sd[X]);
    }

    return ds.value();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void measure_wall_wall_corr_vs_dist(const Field<T> (&S)[2], const Field<pT> &bcmsid, Direction d,
                                    const parameters &p,
                                    std::vector<Matrix<NCOLOR, NCOLOR, atype>> &res) {

    Field<Matrix<NCOLOR, NCOLOR, atype>> fS;
    Field<fT> tS[2][2];
    tS[0][0] = 0;
    tS[0][1].make_ref_to(tS[0][0], 1);
    tS[1][0] = 0;
    tS[1][1].make_ref_to(tS[1][0], 1);
    int nt = lattice.size(NDIM - 1);
    int nd = lattice.size(d);
    if (hila::myrank() == 0) {
        res.resize(nt * nd);
    }

    ReductionVector<Matrix<NCOLOR, NCOLOR, double>> corr(nt, 0);
    corr.allreduce(false).delayed(true);

    int ip = 0;
    onsites(ALL) {
        tS[ip][0][X] = S[0][X];
    }
    onsites(ALL) {
        fS[X] = S[0][X].outer_product(tS[ip][0][X]);
    }
    onsites(ALL) {
        corr[X.coordinate(Direction(NDIM - 1))] += fS[X];
    }
    corr.reduce();
    for (int it = 0; it < nt; ++it) {
        if (hila::myrank() == 0) {
            res[it * nd] = corr[it] / (lattice.volume() / nt);
        }
        corr[it] = 0;
    }

    for (int ir = 1; ir < nd; ++ir) {
        if(d < NDIM - 1) {
            onsites(ALL) {
                tS[1 - ip][0][X] = tS[ip][0][X + d];
            }
        } else {
            move_filtered(tS[ip], bcmsid, d, ALL, false, tS[1 - ip][0], p);
        }
        ip = 1 - ip;

        onsites(ALL) {
            fS[X] = S[0][X].outer_product(tS[ip][0][X]);
        }
        onsites(ALL) {
            corr[X.coordinate(Direction(NDIM - 1))] += fS[X];
        }
        corr.reduce();
        for (int it = 0; it < nt; ++it) {
            if (hila::myrank() == 0) {
                res[it * nd + ir] = corr[it] / (lattice.volume() / nt);
            }
            corr[it] = 0;
        }
    }
}

template <typename T, typename atype = hila::arithmetic_type<T>>
void measure_wall_wall_corr(const Field<T> &S, Direction d,
                            std::vector<Matrix<NCOLOR, NCOLOR, atype>> &res) {

    Field<Matrix<NCOLOR, NCOLOR, atype>> fS;
    Field<fT> tS[2];
    tS[0].set_nn_topo(1);
    tS[1].set_nn_topo(1);
    int nt = lattice.size(NDIM - 1);
    int nd = lattice.size(d);
    if(hila::myrank() == 0) {
        res.resize(nt * nd);
    }

    ReductionVector<Matrix<NCOLOR, NCOLOR, double>> corr(nt, 0);
    corr.allreduce(false).delayed(true);

    int ip = 0;
    onsites(ALL) {
        tS[ip][X] = S[X];
    }
    onsites(ALL) {
        fS[X] = S[X].outer_product(tS[ip][X]);
    }
    onsites(ALL) {
        corr[X.coordinate(Direction(NDIM - 1))] += fS[X];
    }
    corr.reduce();
    for (int it = 0; it < nt; ++it) {
        if (hila::myrank() == 0) {
            res[it * nd] = corr[it] / (lattice.volume() / nt);
        }
        corr[it] = 0;
    }

    for (int ir = 1; ir < nd; ++ir) {
        onsites(ALL) {
            tS[1 - ip][X] = tS[ip][X + d];
        }
        ip = 1 - ip;

        onsites(ALL) {
            fS[X] = S[X].outer_product(tS[ip][X]);
        }
        onsites(ALL) {
            corr[X.coordinate(Direction(NDIM - 1))] += fS[X];
        }
        corr.reduce();
        for (int it = 0; it < nt; ++it) {
            if (hila::myrank() == 0) {
                res[it * nd + ir] = corr[it] / (lattice.volume() / nt);
            }
            corr[it] = 0;
        }
    }
}

template <typename T, typename atype = hila::arithmetic_type<T>>
void measure_spat_mom_corr(const Field<T> &S, Direction d1, Direction d2,
                                           std::vector<Matrix<T::size(), T::size(), Complex<atype>>> &res) {

    Field<Matrix<T::size(), T::size(), Complex<atype>>> fS;
    CoordinateVector fftdirs;
    foralldir(d) if (d < NDIM - 1) fftdirs[d] = 1;
    fftdirs[NDIM - 1] = 0;

    Direction dt = Direction(NDIM - 1);

    Field<Complex<atype>> tS, tSK;
    Field<Vector<T::size(), Complex<atype>>> SK;
    SK.set_nn_topo(1);
    double svol = lattice.volume() / lattice.size(dt);
    for (int inc = 0; inc < T::size(); inc += 2) {
        onsites(ALL) {
            T tvec = S[X];
            if (inc + 1 < T::size()) {
                tS[X] = Complex<atype>(tvec[inc], tvec[inc + 1]);
            } else {
                tS[X] = Complex<atype>(tvec[inc], 0);
            }
        }
        tS.FFT(fftdirs, tSK);

        onsites(ALL) {
            Complex<atype> tc = tSK[X] / svol;
            SK[X][inc] = tc.real();
            if (inc + 1 < T::size()) {
                SK[X][inc + 1] = tc.imag();
            }
        }
    }

    int nt = lattice.size(dt);
    int nd1 = lattice.size(d1) / 2;
    int nd2 = lattice.size(d2);
    if (hila::myrank() == 0) {
        res.resize((nt * (nt + 1)) / 2 * nd1 * nd2);
    }
    CoordinateVector slice1(0);
    CoordinateVector slice2(0);

    slice1[d1] = -1;
    slice2[d2] = -1;

    std::vector<Vector<T::size(), Complex<atype>>> ls1, ls2;
    int iit = 0;
    for (int it1 = 0; it1 < nt; ++it1) {
        slice1[NDIM - 1] = it1;
        ls1 = SK.get_slice(slice1);
        for (int it2 = it1; it2 < nt; ++it2) {
            slice2[NDIM - 1] = it2;
            ls2 = SK.get_slice(slice2);
            if(hila::myrank() == 0) {
                for (int k1 = 0; k1 < nd1; ++k1) {
                    for (int k2 = 0; k2 < nd2; ++k2) {
                        res[(iit * nd1 + k1) * nd2 + k2] = ls1[k1].outer_product(ls2[k2].conj());
                    }
                }
            }
            ++iit;
        }
    }
}


template <typename T, typename atype = hila::arithmetic_type<T>>
void measure_cond(const Field<T> &S, std::vector<T> &res) {

    int nt = lattice.size(NDIM - 1);
    res.resize(nt);

    ReductionVector<Vector<NCOLOR, double>> cond(nt, 0);
    cond.allreduce(false).delayed(true);

    onsites(ALL) {
        cond[X.coordinate(Direction(NDIM - 1))] += S[X];
    }
    cond.reduce();
    for (int it = 0; it < nt; ++it) {
        res[it] = cond[it] / (lattice.volume() / nt);
    }
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_interp_hb_trajectory(Field<T> (&S)[2], const Field<pT> &bcmsid, parameters &p,
                             int interp_dir, out_only atype &ds) {


    static hila::timer hbbc_timer("HBBC (update)");
    hbbc_timer.start();

    atype dalpha = p.dalpha / p.n_interp_steps; // step size of interpolating parameter
    atype alpha0 = p.alpha;
    if (interp_dir == 0) {
        dalpha = 0;
    } else if (interp_dir < 0) {
        dalpha = -dalpha;
    }

    ds = 0;
    for (int n = 1; n < p.n_interp_steps; ++n) {
        ds += measure_ds_dalpha(S, bcmsid, p); // work done during n-th interpolation step
        p.alpha = alpha0 + n * dalpha;

        hb_update(S, bcmsid, p);

    }
    ds += measure_ds_dalpha(S, bcmsid, p); // work done during last interpolation step


    ds *= dalpha;

    hbbc_timer.stop();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_hbbc_measure(Field<T> (&S)[2], const Field<pT> &bcmsid, parameters &p, std::vector<atype> &ds, bool output = false) {

    auto alpha = p.alpha;


    ds.resize(2);
    ds[0] = 0;
    ds[1] = 0;

    Field<T> S_old = S[0];

    static bool first = true;
    if (first) {
        if(output) {
            // print legend for measurement output
            hila::out0 << "LHBBC:         DIR              dS_EXT          TIME\n";
        }
        first = false;
    }

    ftype ttime = hila::gettime();
    int ipdir;
    if (p.alpha == 0) {
        ipdir = 1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds[0]);
        if (output) {
            hila::out0 << string_format("HBBC            % 1d % 0.12e    % 10.5f\n", ipdir, ds[0],
                                        hila::gettime() - ttime);
        }
    } else if (p.alpha == 1) {
        ipdir = -1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds[1]);
        if (output) {
            hila::out0 << string_format("HBBC            % 1d % 0.12e    % 10.5f\n", ipdir, ds[1],
                                        hila::gettime() - ttime);
        }
    } else {
        p.alpha = 0.5;
        ipdir = 1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds[0]);
        ftype tttime = hila::gettime();
        if (output) {
            hila::out0 << string_format("HBBC            % 1d % 0.12e    % 10.5f\n", ipdir, ds[0],
                                        tttime - ttime);
        }
        ttime = tttime;
        S[0] = S_old;
        p.alpha = 0.5;
        ipdir = -1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds[1]);
        if (output) {
            hila::out0 << string_format("HBBC            % 1d % 0.12e    % 10.5f\n", ipdir, ds[1],
                                        hila::gettime() - ttime);
        }
    }

    S[0] = S_old;
    p.alpha = alpha;
}

template <typename T, typename pT>
void measure_stuff(const Field<T> (&S)[2], const Field<pT> &bcmsid, const parameters &p) {

    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LMEAS:         s_kin               s                  dS\n";
        first = false;
    }
    double s_kin = 0;
    auto s = measure_s(S, bcmsid, p, s_kin) / lattice.volume();
    s_kin /= lattice.volume();
    auto ds = measure_ds_dalpha(S, bcmsid, p);

    hila::out0 << string_format("MEAS % 0.8e % 0.8e % 0.12e\n", s_kin, s, ds);
}

// end measurement functions
///////////////////////////////////////////////////////////////////////////////////
// load/save config functions

template <typename T>
void checkpoint(const Field<T> &S, int trajectory, const parameters &p) {
    double t = hila::gettime();
    // name of config with extra suffix
    std::string config_file =
        p.config_file + "_" + std::to_string(abs((trajectory + 1) / p.n_save) % 2);
    // save config
    S.config_write(config_file);
    // write run_status file
    if (hila::myrank() == 0) {
        std::ofstream outf;
        outf.open("run_status", std::ios::out | std::ios::trunc);
        outf << "trajectory  " << trajectory + 1 << '\n';
        //outf << "alpha       " << p.alpha << "\n";
        outf << "seed        " << static_cast<uint64_t>(hila::random() * (1UL << 61)) << '\n';
        outf << "time        " << hila::gettime() << '\n';
        // write config name to status file:
        outf << "config name  " << config_file << '\n';
        outf.close();
    }
    std::stringstream msg;
    msg << "Checkpointing, time " << hila::gettime() - t;
    hila::timestamp(msg.str().c_str());
}

template <typename T>
bool restore_checkpoint(Field<T> &S, int &trajectory, parameters &p) {
    uint64_t seed;
    bool ok = true;
    p.time_offset = 0;
    hila::input status;
    if (status.open("run_status", false, false)) {
        hila::out0 << "RESTORING FROM CHECKPOINT:\n";
        trajectory = status.get("trajectory");
        //p.alpha = status.get("alpha");
        seed = status.get("seed");
        p.time_offset = status.get("time");
        // get config name with suffix from status file:
        std::string config_file = status.get("config name");
        status.close();
        hila::seed_random(seed);
        S.config_read(config_file);
        ok = true;
    } else {
        std::ifstream in;
        in.open(p.config_file, std::ios::in | std::ios::binary);
        if (in.is_open()) {
            in.close();
            hila::out0 << "READING initial config\n";
            S.config_read(p.config_file);
            ok = true;
        } else {
            ok = false;
        }
    }
    return ok;
}

// end load/save config functions
///////////////////////////////////////////////////////////////////////////////////

// class to specify user defined nearest neighbor connectivity 
class u_nn_map_t : public nn_map_t {
public:
    int s;
    int Nts;
    u_nn_map_t(const CoordinateVector &tlsize, int ts) : nn_map_t(tlsize), s(ts) {
        hila::out0 << "using user-defined nn-mapping with s=" << s << " replica";
        if(s > 0 && lsize[NDIM - 1] % s == 0) { 
            Nts = lsize[NDIM - 1] / s;
            hila::out0 << " of temporal extent Nts=" << Nts << "\n";
        } else {
            s = 1;
            Nts = lsize[NDIM - 1];
            hila::out0 << "  error: Nt=" << lsize[NDIM - 1] << " not divisible by s=" << s << "\n";
            hila::terminate(1);
        }
    }
    ~u_nn_map_t() {

    }
    CoordinateVector operator()(const CoordinateVector &c, const Direction &d) {
        CoordinateVector cn;
        cn = c + d;
        if(d == NDIM - 1) {
            for (int ts = s; ts > 0; --ts) {
                if (cn[NDIM - 1] == ts * Nts) {
                    cn[NDIM - 1] = (ts - 1) * Nts;
                    break;
                }
            }
        } else if(-d == NDIM - 1) {
            for (int ts = 0; ts < s; ++ts) {
                if (c[NDIM - 1] == ts * Nts) {
                    cn[NDIM - 1] = (ts + 1) * Nts - 1;
                    break;
                }
            }
        }

        if (cn[e_y] >= lsize[e_y]) {
            //cn[e_z] += 1;
            cn[e_y] = 0;
        }
        if (cn[e_y] < 0) {
            //cn[e_z] -= 1;
            cn[e_y] = lsize[e_y] - 1;
        }
        if(NDIM > 2) {
            if (cn[e_z] >= lsize[e_z]) {
                //cn[e_x] += 1;
                cn[e_z] = 0;
            }
            if (cn[e_z] < 0) {
                //cn[e_x] -= 1;
                cn[e_z] = lsize[e_z] - 1;
            }
        }
        if (cn[e_x] >= lsize[e_x]) {
            cn[e_x] = 0;
        }
        if (cn[e_x] < 0) {
            cn[e_x] = lsize[e_x] - 1;
        }

        return cn;
    }
};

template <typename bT>
CoordinateVector bcms_coordinates(bT bcms, const Field<bT> &bcmsid) {
    Reduction<int> c = 0;
    c.allreduce(true).delayed(true);
    CoordinateVector cres;
    for (Direction d = Direction(0); d < NDIM - 1; ++d) {
        c = 0;
        onsites(ALL) {
            if (bcmsid[X] == bcms && X.coordinate(Direction(NDIM - 1)) == lattice.size(NDIM - 1) - 1) {
                c += X.coordinate(d);
            }
        }
        cres[d] = c.value();
    }
    cres[NDIM - 1] = 0;
    return cres;
}


int main(int argc, char **argv) {

    // hila::initialize should be called as early as possible
    hila::initialize(argc, argv);

    hila::out0 << "O(" << fT::size()
               << ") scalar field simulation using metropolis corrected heat bath updates\n";

    hila::out0 << "Using floating point epsilon: " << fp<ftype>::epsilon << "\n";

    // hila provides an input class hila::input, which is
    // a convenient way to read in parameters from input files.
    // parameters are presented as key - value pairs, as an example
    //  " lattice size  64, 64, 64, 64"
    // is read below.
    //
    // Values are broadcast to all MPI nodes.
    //
    // .get() -method can read many different input types,
    // see file "input.h" for documentation

    parameters p;

    hila::input par("parameters");

    CoordinateVector lsize;
    // reads NDIM numbers
    lsize = par.get("lattice size");
    // hopping parameter coupling
    p.kappa = par.get("kappa");
    // phi^4 potential (lattice definition)
    p.lambda = par.get("lambda");
    // source terms
    p.source = par.get("source terms");
    // number of replicas
    p.s = par.get("replica number");
    // momentum space entangling region (sphere) radius for alpha=0
    p.l = par.get("momentum scale l");
    // momentum space entangling region (sphere) radius for alpha=1
    p.lc = par.get("momentum scale lc");
    // interpolation parameter
    p.alpha = par.get("alpha");
    // interpolation parameter
    p.dalpha = par.get("dalpha");
    p.bcms = -1;
    // number of trajectories
    p.n_traj = par.get("number of trajectories");
    // number of heat-bath (HB) sweeps per trajectory
    p.n_heatbath = par.get("heatbath updates");
    // number of multi-hits per HB update
    p.n_multhits = par.get("number of hb hits");
    // number of thermalization trajectories
    p.n_therm = par.get("thermalization trajs");
    // number of alpha-interpolation steps
    p.n_interp_steps = par.get("interpolation steps");
    // number of trajectories per correlator dump
    p.n_dump_corr = par.get("trajs/corr dump");
    // number of trajectories per momentum correlator dump
    p.n_dump_mom_corr = par.get("trajs/mom corr dump");
    // number of trajectories per configuration dump
    p.n_dump_config = par.get("trajs/config dump");
    // random seed = 0 -> get seed from time
    long seed = par.get("random seed");
    // save config and checkpoint
    p.n_save = par.get("trajs/saved");
    // measure surface properties and print "profile"
    p.config_file = par.get("config name");

    par.close(); // file is closed also when par goes out of scope

    if(p.s<1) {
        hila::out0 << "Error: replica number s must be non-zero. Current value is s=" << p.s << "\n";
        hila::terminate(1);
    } else if(p.s==1) {
        hila::out0 << "Warning: replica number is set to s=1.\n";
    }

    // specify nearest neighbor topologies (needs to be done before calling lattice.setup())
    u_nn_map_t nn_map0(lsize, p.s);
    lattice.nn_map.push_back(&nn_map0);
    u_nn_map_t nn_map1(lsize, 1);
    lattice.nn_map.push_back(&nn_map1);

    // set up the lattice
    lattice.setup(lsize);

    // We need random number here
    hila::seed_random(seed);

    // use a field bcmsid as look-up table for the magnitude of the spatial momentum vector in
    // Fourier space:
    Field<ftype> bcmsid = 0;
    bcmsid.set_nn_topo(1); // full e_t-periodicity (doesn't really matter, thought, since bcmsid is
                           // the same on all time slices)
    onsites(ALL) {
        auto k = X.coordinates().convert_to_k();
        ftype sknorm = 0;
        for (int d = 0; d < NDIM - 1; ++d) {
            sknorm += (ftype)k[d] * (ftype)k[d];
        }
        sknorm = sqrt(sknorm);
        bcmsid[X] = sknorm;
    }

    std::string output_fname = string_format("O%d", NCOLOR);
    output_fname += string_format("_d%d", NDIM);
    foralldir(d) {
        output_fname += string_format("_%d", lattice.size(d));
    }
    output_fname += string_format("_h%.6f", p.kappa);
    output_fname += string_format("_lm%.6f", p.lambda);
    output_fname += string_format("_s%.6f", p.source[0]);
    for (int ic = 1; ic < NCOLOR; ++ic) {
        output_fname += string_format("_%.6f", p.source[ic]);
    }
    output_fname += string_format("_r%d", p.s);
    output_fname += string_format("_l%.6f", p.l);
    output_fname += string_format("_lc%.6f", p.lc);

    std::string ds_output_fname = output_fname + "_ds.bout";

    std::string cond_output_fname;
    std::vector<fT> av_cond;
    cond_output_fname = output_fname + "_cond.bout";
    av_cond.resize(lattice.size(NDIM - 1));
    for (int ic = 0; ic < av_cond.size(); ++ic) {
        av_cond[ic] = 0;
    }
    
    std::string corr_output_fname[NDIM];
    std::vector<Matrix<NCOLOR, NCOLOR, ftype>> av_corr[NDIM];
    foralldir(d) {
        corr_output_fname[d] = output_fname + string_format("_wwcorr_d%d.bout", (int)d);
        if(hila::myrank() == 0) {
            av_corr[d].resize(lattice.size(NDIM - 1) * lattice.size(d));
            for (int ic = 0; ic < av_corr[d].size(); ++ic) {
                av_corr[d][ic] = 0;
            }
        }
    }

    std::string corr_vs_dist_output_fname;
    std::vector<Matrix<NCOLOR, NCOLOR, ftype>> av_corr_vs_dist;
    corr_vs_dist_output_fname = output_fname + string_format("_wwcorr_vs_dist_d%d.bout", NDIM - 1);
    if (hila::myrank() == 0) {
        av_corr_vs_dist.resize(lattice.size(NDIM - 1) * lattice.size(NDIM - 1));
        for (int ic = 0; ic < av_corr_vs_dist.size(); ++ic) {
            av_corr_vs_dist[ic] = 0;
        }
    }

    std::string mom_corr_output_fname[NDIM][NDIM];
    std::vector<Matrix<NCOLOR, NCOLOR, Complex<ftype>>> av_mom_corr[NDIM][NDIM];
    foralldir(d1) foralldir(d2) if (d1 <= d2 && d2 < NDIM - 1) {
        mom_corr_output_fname[d1][d2] = output_fname + string_format("_mom_corr_d%d%d.bout", (int)d1, (int)d2);
        if (hila::myrank() == 0) {
            av_mom_corr[d1][d2].resize((lattice.size(NDIM - 1) * (lattice.size(NDIM - 1) + 1)) / 2 *
                                       lattice.size(d1) / 2 * lattice.size(d2));
            for (int ic = 0; ic < av_mom_corr[d1][d2].size(); ++ic) {
                av_mom_corr[d1][d2][ic] = 0;
            }
        }
    }

    // use negative trajectory for thermal
    int start_traj = -p.n_therm;

    // Alloc field (S) 
    Field<fT> S[2];
    Field<fT> S_back = 0;

    if (!restore_checkpoint(S[0], start_traj, p)) {
        S[0] = 0;
    }
    S[1].make_ref_to(S[0], 1);

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");

    double act_old, act_new, s_old, s_new;
    bool first = true;
    bool first_corr = true;
    bool first_mom_corr = true;
    for (int trajectory = start_traj; trajectory <= p.n_traj; ++trajectory) {

        ftype ttime = hila::gettime();

        update_timer.start();

        do_hb_trajectory(S, bcmsid, p);

        // put sync here in order to get approx gpu timing
        hila::synchronize_threads();
        update_timer.stop();

        measure_timer.start();

        hila::out0 << "Measure_start " << trajectory << '\n';

        measure_stuff(S, bcmsid, p);

        if (p.s > 1 && trajectory >= 0) {
            if (p.n_interp_steps > 0) {
                std::vector<double> ds(2, 0);
                do_hbbc_measure(S, bcmsid, p, ds);
                bwrite_to_file(ds_output_fname, ds, p, trajectory, first);
                first = false;
            }
            if (p.n_dump_corr) {
                std::vector<fT> cond;
                // hila::out0 << "measure_cond " << '\n';
                measure_cond(S[1], cond);
                for (int ic = 0; ic < av_cond.size(); ++ic) {
                    av_cond[ic] += cond[ic];
                }
                foralldir(d) {
                    std::vector<Matrix<NCOLOR, NCOLOR, ftype>> corr;
                    //hila::out0 << "measure_wall_wall_corr in " << d << "-dir"  << '\n';
                    measure_wall_wall_corr(S[1], d, corr);
                    for (int ic = 0; ic < av_corr[d].size(); ++ ic) {
                        av_corr[d][ic] += corr[ic];
                    }
                }
                {
                    std::vector<Matrix<NCOLOR, NCOLOR, ftype>> corr;
                    //hila::out0 << "measure_wall_wall_corr_vs_dist in " << d << "-dir" << '\n';
                    measure_wall_wall_corr_vs_dist(S, bcmsid, Direction(NDIM - 1), p, corr);
                    for (int ic = 0; ic < av_corr_vs_dist.size(); ++ic) {
                        av_corr_vs_dist[ic] += corr[ic];
                    }
                }
                if((trajectory + 1) % p.n_dump_corr == 0) {
                    int icdump = (trajectory + 1) / p.n_dump_corr;

                    for (int ic = 0; ic < av_cond.size(); ++ic) {
                        av_cond[ic] /= p.n_dump_corr;
                    }
                    bwrite_to_file(cond_output_fname, av_cond, p, icdump, first_corr);
                    for (int ic = 0; ic < av_cond.size(); ++ic) {
                        av_cond[ic] = 0;
                    }

                    foralldir(d) {
                        for (int ic = 0; ic < av_corr[d].size(); ++ic) {
                            av_corr[d][ic] /= p.n_dump_corr;
                        }
                        bwrite_to_file(corr_output_fname[d], av_corr[d], p, icdump, first_corr);
                        for (int ic = 0; ic < av_corr[d].size(); ++ic) {
                            av_corr[d][ic] = 0;
                        }
                    }
                    for (int ic = 0; ic < av_corr_vs_dist.size(); ++ic) {
                        av_corr_vs_dist[ic] /= p.n_dump_corr;
                    }
                    bwrite_to_file(corr_vs_dist_output_fname, av_corr_vs_dist, p, icdump, first_corr);
                    for (int ic = 0; ic < av_corr_vs_dist.size(); ++ic) {
                        av_corr_vs_dist[ic] = 0;
                    }
                    first_corr = false;
                }
            }
            if (p.n_dump_mom_corr) {
                foralldir(d1) foralldir(d2) if(d1 <= d2 && d2 < NDIM - 1) {
                    std::vector<Matrix<NCOLOR, NCOLOR, Complex<ftype>>> corr;
                    // hila::out0 << "measure_wall_wall_corr in " << d << "-dir"  << '\n';
                    measure_spat_mom_corr(S[0], d1, d2, corr);
                    for (int ic = 0; ic < av_mom_corr[d1][d2].size(); ++ic) {
                        av_mom_corr[d1][d2][ic] += corr[ic];
                    }
                }
                if ((trajectory + 1) % p.n_dump_mom_corr == 0) {
                    int icdump = (trajectory + 1) / p.n_dump_mom_corr;
                    foralldir(d1) foralldir(d2) if(d1 <= d2 && d2 < NDIM -1) {
                        for (int ic = 0; ic < av_mom_corr[d1][d2].size(); ++ic) {
                            av_mom_corr[d1][d2][ic] /= p.n_dump_mom_corr;
                        }
                        bwrite_to_file(mom_corr_output_fname[d1][d2], av_mom_corr[d1][d2], p, icdump, first_mom_corr);
                        for (int ic = 0; ic < av_mom_corr[d1][d2].size(); ++ic) {
                            av_mom_corr[d1][d2][ic] = 0;
                        }
                    }
                    first_mom_corr = false;
                }
            }
            
        }

        hila::out0 << "Measure_end " << trajectory << '\n';

        measure_timer.stop();

        if (p.n_dump_config && (trajectory + 1) % p.n_dump_config == 0) {
            Field<fTf> fS;
            onsites(ALL) {
                fS[X] = S[0][X];
            }

            int icdump = (trajectory + 1) / p.n_dump_config;
            std::string cdump_file = output_fname + string_format("_conf%04d.bout", icdump);
            fS.config_write(cdump_file);
        }

        if (p.n_save > 0 && (trajectory + 1) % p.n_save == 0) {
            checkpoint(S[0], trajectory, p);
        }
    }

    hila::finishrun();
}
