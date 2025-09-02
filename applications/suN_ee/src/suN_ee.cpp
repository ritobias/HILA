#include "hila.h"
#include "wilson_plaquette_action_ee.h"
#include "sun_hmc_ee.h"
#include "gauge/sun_heatbath.h"
#include "gauge/sun_overrelax.h"
#include "gradient_flow_ee.h"
#include "tools/string_format.h"
#include "tools/floating_point_epsilon.h"
#include "gauge/sun_max_staple_sum_rot.h"
#include "gauge/polyakov.h"
#include <algorithm>

#ifndef NCOLOR
#define NCOLOR 3
#endif

using ftype = double;
using mygroup = SU<NCOLOR, ftype>;

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    ftype beta;         // inverse gauge coupling
    int s;              // number of replicas
    ftype l;            // entangling region slab width
    ftype alpha;        // interpolation parameter
    int n_traj;         // number of trajectories to generate
    int n_therm;        // number of thermalization trajectories (counts only accepted traj.)
    int n_update;       // number of heat-bath sweeps per "trajectory"
    int n_overrelax;    // number of overrelaxation sweeps per "trajectory"
    int gflow_freq;     // number of trajectories between gflow measurements
    ftype gflow_max_l;  // flow scale at which gradient flow stops
    ftype gflow_l_step; // flow scale interval between flow measurements
    ftype gflow_a_accu; // desired absolute accuracy of gradient flow integration steps
    ftype gflow_r_accu; // desired relative accuracy of gradient flow integration steps
    int n_save;         // number of trajectories between config. check point
    std::string config_file;
    ftype time_offset;
};

///////////////////////////////////////////////////////////////////////////////////
// heat-bath functions


/**
 * @brief Sum the staples of link matrices to direction dir taking into account plaquette weights
 *
 * Naive method is to compute:
 *
 * \code {.cpp}
 * foralldir(d2) if (d2 != d1)
 *     stapes[par] += U[d2][X]*U[d1][X+d2]*U[d2][X+d1].dagger()  +
 *                    U[d2][X-d2].dagger()*U[d1][X-d2]*U[d2][X-d2+d1]
 * \endcode
 *
 * But the method is computed in a slightly more optimized way
 *
 * @tparam T gaug field type
 * @tparam pT type of plaq_tbc_mode
 * @param U[2] GaugeField for the two different nn-topologies
 * @param plaq_tbc_mode PlaquetteField specifying nn-topology to be used to compute plaquettes
 * @param staples Filed to compute staplesum into at each lattice point
 * @param d1 Direction to compute staplesum for
 * @param par Parity to compute staplesum for
 * @param alpha \in [0,1] bcmode interpolation parameter for plaquettes with plaq_tbc_mode = 2 
 */
template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void staplesum(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, atype alpha,
               out_only Field<T> &staples, Direction d1, Parity par = ALL) {

    Field<T> lower[2];
    lower[1].set_nn_topo(1);

    onsites(par) staples[X] = 0;

    foralldir(d2) if (d2 != d1) {

        U[1][d2].start_gather(d1, ALL);
        U[0][d2].start_gather(d1, ALL);
        U[1][d1].start_gather(d2, par);
        U[0][d1].start_gather(d2, par);

        // calculate first lower 'u' of the staple sum
        // do it on opp parity
        onsites(opp_parity(par)) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if(ip == 1) {
                lower[1][X] = U[1][d2][X].dagger() * U[1][d1][X] * U[1][d2][X + d1];
            } else if(ip == 2) {
                lower[1][X] = U[1][d2][X].dagger() * U[1][d1][X] * U[1][d2][X + d1];
                lower[0][X] = U[0][d2][X].dagger() * U[0][d1][X] * U[0][d2][X + d1];
            } else {
                lower[0][X] = U[0][d2][X].dagger() * U[0][d1][X] * U[0][d2][X + d1];
            }
        }

        plaq_tbc_mode[d1][d2].start_gather(-d2, par);
        lower[1].start_gather(-d2, par);
        lower[0].start_gather(-d2, par);

        // calculate then the upper 'n', and add the lower
        // lower could also be added on a separate loop
        onsites(par) {
            int ip = plaq_tbc_mode[d1][d2][X];
            if (ip == 1) {
                staples[X] += (U[1][d2][X] * U[1][d1][X + d2] * U[1][d2][X + d1].dagger());
            } else if (ip == 2) {
                staples[X] +=
                    alpha * (U[1][d2][X] * U[1][d1][X + d2] * U[1][d2][X + d1].dagger()) +
                    (1.0 - alpha) * (U[0][d2][X] * U[0][d1][X + d2] * U[0][d2][X + d1].dagger());
            } else {
                staples[X] += (U[0][d2][X] * U[0][d1][X + d2] * U[0][d2][X + d1].dagger());
            }
        }
        onsites(par) {
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

/**
 * @brief Wrapper function to updated GaugeField per direction
 * @details Computes first staplesum, then uses computed result to evolve GaugeField either with
 * overrelaxation or heatbath
 *
 * @tparam T gaug field type
 * @tparam pT type of plaq_tbc_mode
 * @param U[2] GaugeField for the two different nn-topologies
 * @param plaq_tbc_mode PlaquetteField specifying nn-topology to be used to compute plaquettes
 * @param p parameters
 * @param d Direction specifies to update links in direction d
 * @param par Parity specifies parity of links to be updated
 * @param relax bool specifies whether
 * to update with overrelaxation (or heatbath)
 */
template <typename T, typename pT>
void hb_update_parity_dir(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                       const parameters &p, Direction d, Parity par, bool relax) {

    static hila::timer hb_timer("Heatbath");
    static hila::timer or_timer("Overrelax");
    static hila::timer staples_timer("Staplesum");

    Field<T> staples;

    staples_timer.start();

    staplesum(U, plaq_tbc_mode, p.alpha, staples, d, par);

    staples_timer.stop();

    if (relax) {

        or_timer.start();

        onsites(par) {
#ifdef SUN_OVERRELAX_dFJ
            suN_overrelax_dFJ(U[0][d][X], staples[X], p.beta);
#else
            //suN_overrelax(U[0][d][X], staples[X]);
            suN_overrelax(U[0][d][X], staples[X], p.beta);
#endif
        }
        or_timer.stop();

    } else {

        hb_timer.start();
        onsites(par) {
            suN_heatbath(U[0][d][X], staples[X], p.beta);
        }
        hb_timer.stop();

    }
}

/**
 * @brief Wrapper update function
 * @details Gauge Field update sweep with randomly chosen parities and directions
 *
 * @tparam T gaug field type
 * @tparam pT type of plaq_tbc_mode
 * @param U[2] GaugeField for the two different nn-topologies
 * @param plaq_tbc_mode PlaquetteField specifying nn-topology to be used to compute plaquettes
 * @param p parameters
 */
template <typename T, typename pT>
void hb_update(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode, const parameters &p) {
    for (int i = 0; i < 2 * NDIM; ++i) {
        // randomly choose a parity and a direction:
        int tdp = hila::broadcast((int)(hila::random() * 2 * NDIM));
        int tdir = tdp / 2;
        int tpar = 1 + (tdp % 2);
        // randomly choose whether to do overrelaxation or heatbath update (with probabilities
        // according to specified p.n_update and p.n_overrelax):
        bool relax =
            hila::broadcast((int)(hila::random() * (p.n_update + p.n_overrelax)) >= p.n_update);
        // perform the selected updates:
        hb_update_parity_dir(U, plaq_tbc_mode, p, Direction(tdir), Parity(tpar), relax);
    }
}

/**
 * @brief Evolve gauge field
 * @details Evolution happens by means of heat bath and overrelaxation. We do on average
 * p.n_update heatbath updates and p.n_overrelax overrelaxation updates on each link per sweep.
 *
 * @tparam T gaug field type
 * @tparam pT type of plaq_tbc_mode
 * @param U[2] GaugeField for the two different nn-topologies
 * @param plaq_tbc_mode PlaquetteField specifying nn-topology to be used to compute plaquettes
 * @param p parameters
 */
template <typename T, typename pT>
void do_hb_trajectory(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                   const parameters &p) {
    for (int n = 0; n < p.n_update + p.n_overrelax; n++) {
        hb_update(U, plaq_tbc_mode, p);
    }
    U[0].reunitarize_gauge();
}

// heat-bath functions
///////////////////////////////////////////////////////////////////////////////////
// boundary update functions

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_hmc_bc_update(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                      VectorField<Algebra<T>> &E, parameters &p, const hmc_parameters &p_hmc) {

    static hila::timer hmc_timer("HMC (update)");
    hmc_timer.start();

    auto alpha = p.alpha;

    atype ds = 0;
    GaugeField<T> U_old = U[0];

    int ipdir = 0;

    foralldir(d) onsites(ALL) E[d][X].gaussian_random();

    double act_old, act_new, g_act_old, g_act_new, e_act_old, e_act_new;

    act_old = measure_action(U, plaq_tbc_mode, alpha, E, p_hmc, g_act_old, e_act_old);

    ftype ttime = hila::gettime();

    if (alpha == 0) {
        ipdir = 1;
        do_interp_hmc_trajectory(U, plaq_tbc_mode, E, p_hmc, ipdir, ds);
        alpha = 1.0;
    } else {
        ipdir = -1;
        do_interp_hmc_trajectory(U, plaq_tbc_mode, E, p_hmc, ipdir, ds);
        alpha = 0;
    }

    act_new = measure_action(U, plaq_tbc_mode, alpha, E, p_hmc, g_act_new, e_act_new);


    bool reject = hila::broadcast(exp(-(act_new - act_old)) < hila::random());

    hila::out0 << string_format(
        "HMCUD  DIR: % 1d  S_TOT_START: % 0.6e  dS_TOT: % 0.6e  dS_GAUGE: % 0.6e  dS_MOM: % 0.6e  "
        "dS_EXT: % 0.6e  ACCEPT: % 1d  TIME: %0.5f\n",
        ipdir, act_old, act_new - act_old, g_act_new - g_act_old, e_act_new - e_act_old, ds,
        (int)!reject, hila::gettime() - ttime);


    if (reject) {
        U[0] = U_old;
    } else {
        p.alpha = alpha;
    }

    hmc_timer.stop();
}

// boundary update functions
///////////////////////////////////////////////////////////////////////////////////
// measurement functions

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void do_hmc_measure(GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                   parameters &p, const hmc_parameters &p_hmc) {

    static hila::timer hmc_ms_timer("HMC (measure)");
    hmc_ms_timer.start();

    auto alpha = p.alpha;

    atype ds = 0;
    GaugeField<T> U_old = U[0];

    VectorField<Algebra<T>> E;

    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LHMC: DIR       S_TOT_S        dS_TOT       dS_PLAQ        dE_KIN        dS_EXT"
                      "    SUCC       TIME\n";
        first = false;
    }

    int ipdir=0;

    foralldir(d) onsites(ALL) E[d][X].gaussian_random();

    double g_act_old, g_act_new, e_act_old, e_act_new;

    auto act_old = measure_action(U, plaq_tbc_mode, alpha, E, p_hmc, g_act_old, e_act_old);

    ftype ttime = hila::gettime();

    //do_hmc_trajectory(U, plaq_tbc_mode, alpha, E, p_hmc);
    if (alpha == 0) {
        ipdir = 1;
        do_interp_hmc_trajectory(U, plaq_tbc_mode, E, p_hmc, ipdir, ds);
        alpha = 1.0;
    } else {
        ipdir = -1;
        do_interp_hmc_trajectory(U, plaq_tbc_mode, E, p_hmc, ipdir, ds);
        alpha = 0;
    }
    //do_hmc_trajectory(U, plaq_tbc_mode, alpha, E, p_hmc);

    auto act_new = measure_action(U, plaq_tbc_mode, alpha, E, p_hmc, g_act_new, e_act_new);


    //bool reject = hila::broadcast(exp(-(act_new - act_old)) < hila::random());
    bool reject = true;


    hila::out0 << string_format("HMC    % 1d % 0.6e % 0.6e % 0.6e % 0.6e % 0.6e      % 1d    %0.5f\n", ipdir,
                                act_old, act_new - act_old, g_act_new - g_act_old,
                                e_act_new - e_act_old, ds, (int)!reject, hila::gettime() - ttime);


    if (reject) {
        U[0] = U_old;
    } else {
        p.alpha = alpha;
    }

    hmc_ms_timer.stop();
}


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void measure_stuff(const GaugeField<T> (&U)[2], const PlaquetteField<pT> &plaq_tbc_mode,
                   parameters &p) {
    // perform measurements on current gauge field and print results in formatted form to standard
    // output
    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LMEAS:        plaq         dplaq            dS\n";
        first = false;
    }

    auto plaqs =
        measure_s_wplaq(U, plaq_tbc_mode, p.alpha) / (lattice.volume() * NDIM * (NDIM - 1) / 2);
    auto dplaqs = measure_ds_wplaq_dbcms(U, plaq_tbc_mode) / (p.s * (NDIM - 1));
    auto ds = dplaqs * (p.s * (NDIM - 1)) * p.beta;

    hila::out0 << string_format("MEAS % 0.6e % 0.6e % 0.6e\n", plaqs, dplaqs, ds);
}

// end measurement functions
///////////////////////////////////////////////////////////////////////////////////
// load/save config functions

template <typename T>
void checkpoint(const GaugeField<T> &U, int trajectory, const parameters &p) {
    double t = hila::gettime();
    // name of config with extra suffix
    std::string config_file =
        p.config_file + "_" + std::to_string(abs((trajectory + 1) / p.n_save) % 2);
    // save config
    U.config_write(config_file);
    // write run_status file
    if (hila::myrank() == 0) {
        std::ofstream outf;
        outf.open("run_status", std::ios::out | std::ios::trunc);
        outf << "trajectory  " << trajectory + 1 << '\n';
        outf << "alpha       " << p.alpha << "\n";
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
bool restore_checkpoint(GaugeField<T> &U, int &trajectory, parameters &p) {
    uint64_t seed;
    bool ok = true;
    p.time_offset = 0;
    hila::input status;
    if (status.open("run_status", false, false)) {
        hila::out0 << "RESTORING FROM CHECKPOINT:\n";
        trajectory = status.get("trajectory");
        p.alpha = status.get("alpha");
        seed = status.get("seed");
        p.time_offset = status.get("time");
        // get config name with suffix from status file:
        std::string config_file = status.get("config name");
        status.close();
        hila::seed_random(seed);
        U.config_read(config_file);
        ok = true;
    } else {
        std::ifstream in;
        in.open(p.config_file, std::ios::in | std::ios::binary);
        if (in.is_open()) {
            in.close();
            hila::out0 << "READING initial config\n";
            U.config_read(p.config_file);
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
        if(s > 0 && lsize[e_t] % s == 0) { 
            Nts = lsize[e_t] / s;
            hila::out0 << " of temporal extent Nts=" << Nts << "\n";
        } else {
            s = 1;
            Nts = lsize[e_t];
            hila::out0 << "  error: Nt=" << lsize[e_t] << " not divisible by s=" << s << "\n";
            hila::terminate(1);
        }
    }
    ~u_nn_map_t() {

    }
    CoordinateVector operator()(const CoordinateVector &c, const Direction &d) {
        CoordinateVector cn;
        cn = c + d;
        if(d==e_t) {
            for (int ts = s; ts > 0; --ts) {
                if (cn[e_t] == ts * Nts) {
                    cn[e_t] = (ts - 1) * Nts;
                    break;
                }
            }
        } else if(d==e_t_down) {
            for (int ts = 0; ts < s; ++ts) {
                if (c[e_t] == ts * Nts) {
                    cn[e_t] = (ts + 1) * Nts - 1;
                    break;
                }
            }
        }

        if (cn[e_y] >= lsize[e_y]) {
            cn[e_z] += 1;
            cn[e_y] = 0;
        }
        if (cn[e_y] < 0) {
            cn[e_z] -= 1;
            cn[e_y] = lsize[e_y] - 1;
        }
        if (cn[e_z] >= lsize[e_z]) {
            cn[e_x] += 1;
            cn[e_z] = 0;
        }
        if (cn[e_z] < 0) {
            cn[e_x] -= 1;
            cn[e_z] = lsize[e_z] - 1;
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
    for (Direction d = e_x; d < NDIM - 1; ++d) {
        c = 0;
        onsites(ALL) {
            if (bcmsid[X] == bcms && X.t() == lattice.size(e_t) - 1) {
                c += X.coordinate(d);
            }
        }
        cres[d] = c.value();
    }
    cres[e_t] = 0;
    return cres;
}


int main(int argc, char **argv) {

    // hila::initialize should be called as early as possible
    hila::initialize(argc, argv);

    hila::out0 << "SU(" << mygroup::size() << ") heat-bath with overrelaxation\n";

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
    hmc_parameters p_hmc;

    hila::input par("parameters");

    CoordinateVector lsize;
    // reads NDIM numbers
    lsize = par.get("lattice size");
    // inverse gauge coupling
    p.beta = par.get("beta");
    p_hmc.beta = p.beta;
    // number of replicas
    p.s = par.get("replica number");
    // entangling region slab width
    p.l = par.get("slab width");
    // number of trajectories
    p.n_traj = par.get("number of trajectories");
    // number of heat-bath (HB) sweeps per trajectory
    p.n_update = par.get("updates in trajectory");
    // number of overrelaxation sweeps petween HB sweeps
    p.n_overrelax = par.get("overrelax steps");
    // number of thermalization trajectories
    p.n_therm = par.get("thermalization trajs");
    // hmc trajectory length
    p_hmc.trajlen = par.get("hmc trajlen");
    p_hmc.dt = 1.0 / p_hmc.trajlen;
    // wilson flow frequency (number of traj. between w. flow measurement)
    p.gflow_freq = par.get("gflow freq");
    // wilson flow max. flow-distance
    p.gflow_max_l = par.get("gflow max lambda");
    // wilson flow flow-distance step size
    p.gflow_l_step = par.get("gflow lambda step");
    // wilson flow absolute accuracy (per integration step)
    p.gflow_a_accu = par.get("gflow abs. accuracy");
    // wilson flow relative accuracy (per integration step)
    p.gflow_r_accu = par.get("gflow rel. accuracy");
    // random seed = 0 -> get seed from time
    long seed = par.get("random seed");
    // save config and checkpoint
    p.n_save = par.get("trajs/saved");
    // measure surface properties and print "profile"
    p.config_file = par.get("config name");

    par.close(); // file is closed also when par goes out of scope

    p.alpha = 0;

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

    size_t Vs = lattice.volume()/lattice.size(3); // spatial lattice volume
    size_t Ae = Vs / lattice.size(0); // area of y-z-plane (~area of connected component of entangling surface)

    size_t bcms = (size_t)(p.l * (ftype)Ae);

    Field<size_t> bcmsid = 0;
    bcmsid.set_nn_topo(1); // full e_t-periodicity to distribute spatial bcmsid to all time slices
    {
        // assigne spatial bcmsid to each spatial site at e_t=0:
        CoordinateVector c = {0, 0, 0, 0};
        for (size_t i = 0; i < Vs; ++i) {
            bcmsid[c] = i;
            c = nn_map1(c, e_y);
        }

        // copy bcmsid from e_t=0 to remaining time slices:
        for (size_t i = 1; i < lattice.size(3); ++i) {
#pragma hila safe_access(bcmsid)
            onsites(ALL) {
                if(X.t() == i) {
                    bcmsid[X] = bcmsid[X - e_t];
                }
            }
        }
    }

    // get coordinates of spatial site with bcmsid = bcms:
    auto bcmsc = bcms_coordinates(bcms, bcmsid);
    //hila::out << "rank " << hila::myrank() << "  : bcmsc=(" << bcmsc << ")\n";

    // plaquette field indicating whether a plaquette belongs to region A (value 1) 
    // or region B (value 0)
    PlaquetteField<int> plaq_tbc_mode;
    plaq_tbc_mode.set_nn_topo(1);
    foralldir(d1) {
        onsites(ALL) {
            int bid0 = bcmsid[X];
            int bid1 = bcmsid[X + d1];
            if (bid0 <= bcms && bid1 <= bcms) {
                // link belongs to region A
                plaq_tbc_mode[d1][d1][X] = 1;
            } else {
                // link belongs to region B
                plaq_tbc_mode[d1][d1][X] = 0;
            }
        }
        foralldir(d2) if(d2 != d1) {
            onsites(ALL) {
                int bid2 = bcmsid[X + d2];
                if (plaq_tbc_mode[d1][d1][X] == 1 && bid2 <= bcms) {
                    // plaquette belongs to region A
                    plaq_tbc_mode[d1][d2][X] = 1;
                } else {
                    // plaquette belongs to region B
                    plaq_tbc_mode[d1][d2][X] = 0;
                }
            }
        }
    }
    {
        // mark the temporal plaquettes (value 2) that would be affected when the spatial site with
        // bcmsid equal to bcms + 1 were moved from region B to region A:
        CoordinateVector c, cc;
        for (int it = 0; it < lattice.size(e_t); ++it) {
            c = nn_map1({bcmsc[0], bcmsc[1], bcmsc[2], it}, e_y);
            if (nn_map1(c, e_t) != nn_map0(c, e_t)) {
                foralldir(d1) if (d1 != e_t) {
                    cc = nn_map0(c, opp_dir(d1));
                    plaq_tbc_mode[d1][e_t][cc] = 2;
                    plaq_tbc_mode[e_t][d1][cc] = 2;
                }
            }
        }
    }
    /*
    if(0) {
        foralldir(d1) {
            foralldir(d2) if(d1 != d2) {
                onsites(ALL) {
                    int ip = plaq_tbc_mode[d1][d2][X];
                    if (ip == 2) {
                        hila::out << "X=(" << X.coordinates() << "), d1=" << d1 << ", d2=" << d2
                                  << ", X+" << d2 << ": " << plaq_tbc_mode[d1][d2][X + d2] << ", X-"
                                  << d2 << ": " << plaq_tbc_mode[d1][d2][X - d2] << "\n";
                    } else if(ip == -1) {
                        hila::out << "error: " << "X=(" << X.coordinates() << "), d1=" << d1
                                  << ", d2=" << d2 << " plaq_tbc_mode = 0\n";
                    }
                }
            }
        }
    }
    */


    // use negative trajectory for thermal
    int start_traj = -p.n_therm;

    // Alloc gauge field (U) and gauge momentum field (E)
    GaugeField<mygroup> U[2];
    if (!restore_checkpoint(U[0], start_traj, p)) {
        U[0] = 1;
    }
    U[1].make_ref_to(U[0], 1);

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");
    hila::timer gf_timer("Gradient Flow");

    ftype t_step0 = 0;
    for (int trajectory = start_traj; trajectory <= p.n_traj; ++trajectory) {

        ftype ttime = hila::gettime();

        update_timer.start();

        do_hb_trajectory(U, plaq_tbc_mode, p);

        // put sync here in order to get approx gpu timing
        hila::synchronize_threads();
        update_timer.stop();

        measure_timer.start();

        hila::out0 << "Measure_start " << trajectory << '\n';

        measure_stuff(U, plaq_tbc_mode, p);

        do_hmc_measure(U, plaq_tbc_mode, p, p_hmc);

        hila::out0 << "Measure_end " << trajectory << '\n';

        measure_timer.stop();

        if (trajectory >= 0) {

            if (p.gflow_freq > 0 && trajectory % p.gflow_freq == 0) {

                int gtrajectory = trajectory / p.gflow_freq;
                if (p.gflow_l_step > 0) {

                    gf_timer.start();

                    int nflow_steps = (int)(p.gflow_max_l / p.gflow_l_step);
                    ftype gftime = hila::gettime();
                    hila::out0 << "Gflow_start " << gtrajectory << '\n';

                    GaugeField<mygroup> V[2];
                    V[1].make_ref_to(V[0], 1);
                    V[0] = U[0];

                    ftype t_step = t_step0;
                    measure_gradient_flow_stuff(V, plaq_tbc_mode, p.alpha, (ftype)0.0, t_step, p.s);
                    t_step = do_gradient_flow_adapt(V, plaq_tbc_mode, p.alpha, (ftype)0.0, p.gflow_l_step, p.gflow_a_accu,
                                                    p.gflow_r_accu, t_step);
                    measure_gradient_flow_stuff(V, plaq_tbc_mode, p.alpha, p.gflow_l_step, t_step, p.s);
                    t_step0 = t_step;

                    for (int i = 1; i < nflow_steps; ++i) {

                        t_step =
                            do_gradient_flow_adapt(V, plaq_tbc_mode, p.alpha, i * p.gflow_l_step, (i + 1) * p.gflow_l_step,
                                                   p.gflow_a_accu, p.gflow_r_accu, t_step);

                        measure_gradient_flow_stuff(V, plaq_tbc_mode, p.alpha, (i + 1) * p.gflow_l_step, t_step, p.s);
                    }

                    hila::out0 << "Gflow_end " << gtrajectory << "    time " << std::setprecision(3)
                               << hila::gettime() - gftime << '\n';

                    gf_timer.stop();

                }

            }

        }

        if (p.n_save > 0 && (trajectory + 1) % p.n_save == 0) {
            checkpoint(U[0], trajectory, p);
        }
    }

    hila::finishrun();
}
