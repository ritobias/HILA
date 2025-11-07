#include "hila.h"
#include "tools/string_format.h"
#include "tools/floating_point_epsilon.h"
#include <algorithm>

#ifndef NCOLOR
#define NCOLOR 4
#endif

using ftype = double;
using myfield = Vector<NCOLOR, ftype>;

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    ftype kappa;        // hopping parameter
    ftype lambda;       // phi^4 coupling (lattice definition)
    int s;              // number of replicas
    ftype l;            // momentum space entangling region slab width l
    ftype lc;           // momentum space entangling region slab width lc
    ftype alpha;        // interpolation parameter between l and lc
    ftype dalpha;
    ftype bcms;
    int n_traj;         // number of trajectories to generate
    int n_therm;        // number of thermalization trajectories (counts only accepted traj.)
    int n_heatbath;     // number of heat-bath sweeps per "trajectory"
    int n_interp_steps; // number of interpolation steps to change alpha from 0 to 1
    int n_save;         // number of trajectories between config. check point
    std::string config_file;
    ftype time_offset;
};


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered_p(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d, bool both_dirs, out_only Field<T> &Sd, const parameters &p) {
    // pull fields from neighboring sites in d-direction, using different nn-topologies depending on local value of field bcmsid[X]
    // (was used only for testing)
    if (both_dirs) {
        S[1].start_gather(-d);
        S[0].start_gather(-d);
    }
    onsites(ALL) {
        if (bcmsid[X] <= p.bcms) {
            Sd[X] = S[1][X + d];
        } else {
            Sd[X] = S[0][X + d];
        }
    }
    if (both_dirs) {
        onsites(ALL) {
            if (bcmsid[X] <= p.bcms) {
                Sd[X] += S[1][X - d];
            } else {
                Sd[X] += S[0][X - d];
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered_k(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d, bool both_dirs,
                   out_only Field<T> &Sd, const parameters &p) {
    // pull fields from neighboring sites in d-direction, using different nn-topologies for
    // different Fourier modes, depending on the magnitude |k_s| of their spatial k-vector k_s,
    // where |k_s| is encded in bcmsid[X].

    // define direction in which Fourier transform should be taken
    CoordinateVector fftdirs;
    foralldir(d) if(d < NDIM - 1) fftdirs[d] = 1;
    fftdirs[NDIM - 1] = 0;

    Field<Complex<atype>> tS, tSK;
    Field<Complex<atype>> SK[2];
    SK[0] = 0;
    SK[1].make_ref_to(SK[0], 1);
    double svol = lattice.volume() / lattice.size(NDIM - 1);
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
        onsites(ALL) {
            Complex<atype> tc = tS[X] / svol;
            Sd[X][inc] = tc.real();
            if (inc + 1 < T::size()) {
                Sd[X][inc + 1] = tc.imag();
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d, bool both_dirs,
                     out_only Field<T> &Sd, const parameters &p) {

    move_filtered_k(S, bcmsid, d, both_dirs, Sd, p);
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_s(const Field<T>(&S)[2], const Field<pT> &bcmsid, const parameters &p) {
    Reduction<double> s = 0;
    s.allreduce(false).delayed(true);
    Field<T> Sd;
    // hopping terms in all directions:
    foralldir(d) {
        if(d < NDIM - 1) {
            onsites(ALL) {
                s += -p.kappa * S[0][X].dot(S[0][X + d]);
            }
        } else {
            move_filtered(S, bcmsid, d, false, Sd, p);
            onsites(ALL) {
                s += -p.kappa * S[0][X].dot(Sd[X]);
            }
        }
    }
    // potential term:
    onsites(ALL) {
        atype S2 = S[0][X].squarenorm();
        s += S2 + p.lambda * (S2 - 1.0) * (S2 - 1.0);
    }

    return s.value();
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

///////////////////////////////////////////////////////////////////////////////////
// O(N) action
// action: S[\phi] = \sum_{x} ( -\kappa/2 \sum_{\nu}( \phi(x).\phi(x+\hat{\nu}) +
// \phi(x).\phi(x-\hat{\nu}) ) + \phi(x).\phi(x) + \lambda (\phi(x).\phi(x) - 1)^2 )
//
///////////////////////////////////////////////////////////////////////////////////
// heat-bath functions

template <typename T, typename atype = hila::arithmetic_type<T>>
void ON_heatbath(T &S, const T &nnsum, atype kappa, atype lambda) {
    T Sn = 0;
    atype vari = 1.0 / sqrt(2.0);
    Sn.gaussian_random(vari);
    Sn += 0.5 * kappa * nnsum;
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
            move_filtered(S, bcmsid, d, true, Sd, p);
            onsites(par) nnsum[X] += Sd[X];
        }
    }

    hb_timer.start();
    Direction td = Direction(NDIM - 1);
    onsites(par) {
        if(X.coordinate(td) % 2 == tpar) {
            ON_heatbath(S[0][X], nnsum[X], p.kappa, p.lambda);
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
void do_hbbc_measure(Field<T> (&S)[2], const Field<pT> &bcmsid, parameters &p) {

    auto alpha = p.alpha;

    atype ds = 0;
    Field<T> S_old = S[0];

    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LHBBC:         DIR              dS_EXT          TIME\n";
        first = false;
    }

    ftype ttime = hila::gettime();
    ftype tttime;

    int ipdir;
    if (p.alpha == 0) {
        ipdir = 1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds);
    } else if (p.alpha == 1) {
        ipdir = -1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds);
    } else {
        p.alpha = 0.5;
        ipdir = 1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds);
        tttime = hila::gettime();
        hila::out0 << string_format("HBBC            % 1d % 0.12e    % 10.5f\n", ipdir, ds,
                                    tttime - ttime);
        ttime = tttime;
        ds = 0;
        S[0] = S_old;
        p.alpha = 0.5;
        ipdir = -1;
        do_interp_hb_trajectory(S, bcmsid, p, ipdir, ds);
    }

    hila::out0 << string_format("HBBC            % 1d % 0.12e    % 10.5f\n", ipdir, ds,
                                hila::gettime() - ttime);


    S[0] = S_old;
    p.alpha = alpha;
}

template <typename T, typename pT>
void measure_stuff(const Field<T> (&S)[2], const Field<pT> &bcmsid, const parameters &p) {

    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LMEAS:           s                  dS\n";
        first = false;
    }
    auto s = measure_s(S, bcmsid, p) / lattice.volume();
    auto ds = measure_ds_dalpha(S, bcmsid, p);

    hila::out0 << string_format("MEAS % 0.6e % 0.12e\n", s, ds);
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

    hila::out0 << "O(" << myfield::size() << ") scalar field simulation using HMC\n";

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
    // number of thermalization trajectories
    p.n_therm = par.get("thermalization trajs");
    // number of alpha-interpolation steps
    p.n_interp_steps = par.get("interpolation steps");
    // random seed = 0 -> get seed from time
    long seed = par.get("random seed");
    // save config and checkpoint
    p.n_save = par.get("trajs/saved");
    // measure surface properties and print "profile"
    p.config_file = par.get("config name");

    par.close(); // file is closed also when par goes out of scope

    if(p.s<=1) {
        hila::out0 << "Error: replica number s must be larger than 1. Current value is s=" << p.s << "\n";
        hila::terminate(1);
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


    // use negative trajectory for thermal
    int start_traj = -p.n_therm;

    // Alloc field (S) and momentum field (E)
    Field<myfield> S[2];
    Field<myfield> E, S_back = 0;

    if (!restore_checkpoint(S[0], start_traj, p)) {
        S[0] = 0;
    }
    S[1].make_ref_to(S[0], 1);

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");

    double act_old, act_new, s_old, s_new;

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

        if (trajectory >= 0 && p.n_interp_steps > 0) {
            do_hbbc_measure(S, bcmsid, p);
        }

        hila::out0 << "Measure_end " << trajectory << '\n';

        measure_timer.stop();

        if (p.n_save > 0 && (trajectory + 1) % p.n_save == 0) {
            checkpoint(S[0], trajectory, p);
        }
    }

    hila::finishrun();
}
