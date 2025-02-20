#include "hila.h"
#include "tools/string_format.h"
#include "tools/floating_point_epsilon.h"



using ftype = double;

using mygroup = long;

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    ftype beta;         // inverse gauge coupling
    int n_traj;         // number of trajectories to generate
    int n_therm;        // number of thermalization trajectories (counts only accepted traj.)
    int n_update;       // number of heat-bath sweeps per "trajectory"
    int n_ps_update;    // number of plaquette shift (s_{x,\mu\nu}-vars) updates per update
    int n_save; // number of trajectories between config. check point
    std::string config_file;
    ftype time_offset;
};


///////////////////////////////////////////////////////////////////////////////////
// Metropolis update functions

template <typename T>
using sw_t = std::array<std::array<Field<T>, NDIM>, NDIM>;
/**
 * @brief Sum the staples of link variables to direction dir taking into account plaquette
 * orientations and shift weights
 *
 * action of all plaquettes containing h_{x,\mu}:
 * \sum_{\nu} {(h_{x,\mu} + h_{x+\hat{\mu},\nu} - h_{x+\hat{\nu},\mu} - h_{x,\nu} + s_{x,\mu\nu})^2
 * + (h_{x-\hat{\nu},\mu} + h_{x-\hat{\nu}+\hat{\mu},\nu} - h_{x,\mu} - h_{x-\hat{\nu},\nu} +
 * s_{x-\hat{\nu},\mu\nu})^2}
 * the staple sum for h_{x,\mu} is defined by the sum of terms in the above expression which are
 * linear in h_{x,\mu}:
 * 2 (d-1) h_{x,\mu}^2
 *   + 2 h_{x,\mu} \sum_{\nu} {
 *              (h_{x+\hat{\mu},\nu} - h_{x+\hat{\nu},\mu} - h_{x,\nu} 
 *               + s_{x,\mu\nu})
 *            + (-h_{x-\hat{\nu}+\hat{\mu},\nu} - h_{x-\hat{\nu},mu} + h_{x-\hat{\nu},\nu}
 *               - s_{x-\hat{\nu},\mu\nu})
 *     }
 *   + terms independent of h_{x,\mu}
 *
 * @tparam T Z-gauge group type
 * @tparam fT plaquette shift and staplesum type
 * @param H GaugeField to compute staples for
 * @param staplesum Filed to compute staplesum into at each lattice point
 * @param d1 Direction to compute staplesum for
 * @param sw plaquette shift
 * @param par Parity to compute staplesum for
 */
template <typename T, typename fT>
void staplesum(const GaugeField<T> &H, Field<fT> &staples, Direction d1,
               const sw_t<fT> &sw, Parity par = ALL) {

    Field<fT> lower;

    bool first = true;
    foralldir(d2) if (d2 != d1) {

        // anticipate that these are needed
        // not really necessary, but may be faster
        H[d2].start_gather(d1, ALL);
        H[d1].start_gather(d2, par);

        // calculate first lower 'U' of the staple sum
        // do it on opp parity
        onsites(opp_parity(par)) {
            lower[X] = ((fT)(-H[d2][X + d1] - H[d1][X] + H[d2][X]) - sw[d1][d2][X]);
        }

        // calculate then the upper 'n', and add the lower
        // lower could also be added on a separate loop
        if (first) {
            onsites(par) {
                staples[X] =
                    2.0 * ((fT)(H[d2][X + d1] - H[d1][X + d2] - H[d2][X]) + sw[d1][d2][X] + lower[X - d2]);
            }
            first = false;
        } else {
            onsites(par) {
                staples[X] +=
                    2.0 * ((fT)(H[d2][X + d1] - H[d1][X + d2] - H[d2][X]) + sw[d1][d2][X] + lower[X - d2]);
            }
        }
    }
}


/**
 * @brief Z gauge theory metropolis update
 * @details --
 * @tparam T Group element type such as long or int
 * @tparam fT staple type such as double or float
 * @param h link variable to be updated
 * @param stapsum staplesum of plaquettes containing h
 * @param beta coupling constant
 * @return double change in plaquette action
 */
template <typename T, typename fT>
fT z_metropolis(T &h, const fT &stapsum, double beta) {
    fT nstap = (fT)(NDIM - 1) * 2.0; 
    fT si = nstap * h * h + (fT)h * stapsum;
    int he = h + 2 * (int)(hila::random() * 2.0) - 1;
    fT nds = nstap * he * he + (fT)he * stapsum - si;
    if (nds < 0 || hila::random() < exp(-0.5 * beta * nds)) {
        h = he;
        return -nds;
    } else {
        return 0;
    }
}

/**
 * @brief plaquette shift metropolis update
 * @details --
 * @tparam fT shift type such as double or float
 * @tparam T Group element type such as long or int
 * @param sw plaquette shift variable
 * @param tplaq plaquette variable
 * @param beta coupling constant
 * @return double change in plaquette action
 */
template <typename fT, typename T>
fT sw_metropolis(fT &sw,const T &tplaq, double beta) {
    fT si = ((fT)tplaq + sw) * ((fT)tplaq + sw);
    fT nds = ((fT)tplaq - sw) * ((fT)tplaq - sw) - si;
    if (nds < 0 || hila::random() < exp(-0.5 * beta * nds)) {
        sw = -sw;
        return -nds;
    } else {
        return 0;
    }
}

/**
 * @brief Wrapper function to update GaugeField per direction
 * @details Computes first staplesum, then uses computed result to evolve GaugeField with Metropolis
 * updates
 *
 * @tparam T Z-gauge group type
 * @tparam fT plaquette shift type
 * @param H GaugeField to evolve
 * @param p parameter struct
 * @param par Parity
 * @param d Direction to evolve
 * @param sw plaquette shifs
 */
template <typename T, typename fT>
void update_parity_dir(GaugeField<T> &H, const parameters &p, Parity par, Direction d, const sw_t<fT> &sw) {

    static hila::timer me_timer("Metropolis (z)");
    static hila::timer staples_timer("Staplesum");

    Field<fT> staples;

    staples_timer.start();

    staplesum(H, staples, d, sw, par);

    staples_timer.stop();

    me_timer.start();
    onsites(par) {
        z_metropolis(H[d][X], staples[X], p.beta);
    }
    me_timer.stop();

}

/**
 * @brief Wrapper function to update plaquette shift variables
 * @details --
 *
 * @tparam fT shift type such as double or float
 * @tparam T Group element type such as long or int
 * @param sw plaquette shift variable
 * @param p paramaters
 * @param H GaugeField from which to compute the plaquettes
 * @return double change in plaquette action
 */
template <typename fT, typename T>
void update_sw_parity_dir(sw_t<fT> &sw, const parameters &p, Parity par, Direction d1,
                          const GaugeField<T> &H) {
    static hila::timer sw_timer("Metropolis (ps)");

    sw_timer.start();

    foralldir(d2) if (d2 != d1) {
        H[d2].start_gather(d1, par);
        H[d1].start_gather(d2, par);
        onsites(par) {
            T tplaq = H[d1][X] + H[d2][X + d1] - H[d1][X + d2] - H[d2][X];
            sw_metropolis(sw[d1][d2][X], tplaq, p.beta);
            sw[d2][d1][X] = -sw[d1][d2][X];
        }
    }

    sw_timer.stop();
}

/**
 * @brief Wrapper update function
 * @details Gauge Field update sweep with randomly chosen parities and directions
 *
 * @tparam T Z-gauge group type
 * @tparam fT plaquette shift type
 * @param H GaugeField to update
 * @param p Parameter struct
 * @param sw plaquette shifts
 */
template <typename T, typename fT>
void update(GaugeField<T> &H, sw_t<fT> &sw, const parameters &p) {

    for (int i = 0; i < 2 * NDIM; ++i) {
        bool do_sw = hila::broadcast((int)(hila::random() * (1 + p.n_ps_update)) != 0);

        int tdp = hila::broadcast((int)(hila::random() * 2 * NDIM));
        int tdir = tdp / 2;
        int tpar = 1 + (tdp % 2);
        // hila::out0 << "   " << Parity(tpar) << " -- " << Direction(tdir);
        if (do_sw) {
            // update plaquette shift variables for plaquettes spanned by (x,\mu\nu) = (x_{tpar},tdir,d2), d2!=tdir 
            update_sw_parity_dir(sw, p, Parity(tpar), Direction(tdir), H);
        } else {
            // update Z-gauge variable on links (x,\mu) = (x_{tpar},tdir)
            update_parity_dir(H, p, Parity(tpar), Direction(tdir), sw);
        }
    }
    // hila::out0 << "\n";
}

/**
 * @brief Evolve gauge field
 * @details Evolution happens by means of metropolis. 
 * 
 * @tparam T Z-gauge group type
 * @tparam fT plaquette shift type
 * @param H GaugeField to evolve
 * @param sw plaquette shifs
 * @param p parameter struct
 */
template <typename T, typename fT>
void do_trajectory(GaugeField<T> &H, sw_t<fT> &sw, const parameters &p) {
    for (int n = 0; n < p.n_update; n++) {
        for (int i = 0; i <= p.n_ps_update; i++) {
            update(H, sw, p);
        }
    }
}

// Metropolis update functions
///////////////////////////////////////////////////////////////////////////////////
// measurement functions

template <typename T, typename fT>
double measure_s_plaq(const GaugeField<T> &H, const sw_t<fT> &sw) {
    // measure the total plaquette action for the gauge field H
    Reduction<double> plaq = 0;
    plaq.allreduce(false).delayed(true);
    foralldir(d1) foralldir(d2) if (d1 < d2) {
        H[d2].start_gather(d1, ALL);
        H[d1].start_gather(d2, ALL);
        onsites(ALL) {
            plaq += pow(((double)(H[d1][X] + H[d2][X + d1] - H[d1][X + d2] - H[d2][X]) +
                              (double)sw[d1][d2][X]),
                             2.0);
        }
    }
    return plaq.value();
}

template <typename T>
double measure_ns_plaq(const GaugeField<T> &H) {
    // measure the total plaquette action for the gauge field H
    Reduction<double> plaq = 0;
    plaq.allreduce(false).delayed(true);
    foralldir(d1) foralldir(d2) if (d1 < d2) {
        H[d2].start_gather(d1, ALL);
        H[d1].start_gather(d2, ALL);
        onsites(ALL) {
            plaq += pow((double)(H[d1][X] + H[d2][X + d1] - H[d1][X + d2] - H[d2][X]),
                        2.0);
        }
    }
    return plaq.value();
}

template <typename T>
double measure_hsq(const GaugeField<T> &H, double(out_only &h_per_par_dir)[2][NDIM]) {
    // measure the total plaquette action for the gauge field H
    Reduction<double> thsq = 0;
    thsq.allreduce(false).delayed(true);
    ReductionVector<double> h_per_p_d(2 * NDIM);
    h_per_p_d = 0.0;
    h_per_p_d.allreduce(false).delayed(true);
    foralldir(d1) {
        onsites(ALL) {
            thsq += pow(H[d1][X], 2.0);
            h_per_p_d[((int)X.parity() - 1) * NDIM + d1] += H[d1][X];
        }
    }
    h_per_p_d.reduce();
    for (int par = 0; par < 2; ++par) {
        for (int dir = 0; dir < NDIM; ++dir) {
            h_per_par_dir[par][dir] = h_per_p_d[par*NDIM+dir];
        }
    }
    return thsq.value();
}

template <typename T, typename fT>
void measure_stuff(const GaugeField<T> &H, const sw_t<fT> &sw) {
    // perform measurements on current gauge field H and
    // print results in formatted form to standard output
    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LMEAS:       splaq        nsplaq           hsq";
        for (int par = 0; par < 2; ++par) {
            for (int dir = 0; dir < NDIM; ++dir) {
                hila::out0 << "          p" << par << "d" << dir;
            }
        }
        hila::out0 << "\n";
        first = false;
    }
    auto splaq = measure_s_plaq(H, sw) / (lattice.volume() * NDIM * (NDIM - 1) / 2);
    auto nsplaq = measure_ns_plaq(H) / (lattice.volume() * NDIM * (NDIM - 1) / 2);
    double h_per_par_dir[2][NDIM];
    auto hsq = measure_hsq(H, h_per_par_dir) / (lattice.volume() * NDIM);
    hila::out0 << string_format("MEAS % 0.6e % 0.6e % 0.6e", splaq, nsplaq, hsq);
    for (int par = 0; par < 2; ++par) {
        for (int dir = 0; dir < NDIM; ++dir) {
            h_per_par_dir[par][dir] /= lattice.volume() / 2;
            hila::out0 << string_format(" % 0.6e", h_per_par_dir[par][dir]);
        }
    }
    hila::out0 << '\n';
}

// end measurement functions
///////////////////////////////////////////////////////////////////////////////////
// load/save config functions

template <typename group>
void checkpoint(const GaugeField<group> &U, int trajectory, const parameters &p) {
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

template <typename group>
bool restore_checkpoint(GaugeField<group> &U, int &trajectory, parameters &p) {
    uint64_t seed;
    bool ok = true;
    p.time_offset = 0;
    hila::input status;
    if (status.open("run_status", false, false)) {
        hila::out0 << "RESTORING FROM CHECKPOINT:\n";
        trajectory = status.get("trajectory");
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


int main(int argc, char **argv) {

    // hila::initialize should be called as early as possible
    hila::initialize(argc, argv);

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

    hila::out0 << "Z gauge theory heat-bath\n";

    hila::out0 << "Using floating point epsilon: " << fp<ftype>::epsilon << "\n";

    parameters p;

    hila::input par("parameters");

    CoordinateVector lsize;
    // reads NDIM numbers
    lsize = par.get("lattice size");
    // gauge coupling
    p.beta = par.get("beta");
    // number of trajectories
    p.n_traj = par.get("number of trajectories");
    // number of Metropolis sweeps per trajectory
    p.n_update = par.get("updates in trajectory");
    // number of s_{x,\mu\nu} relaxation sweeps per Metropolis sweep
    p.n_ps_update = par.get("plaq shift updates");
    // number of thermalization trajectories
    p.n_therm = par.get("thermalization trajs");
    // random seed = 0 -> get seed from time
    long seed = par.get("random seed");
    // save config and checkpoint
    p.n_save = par.get("trajs/saved");
    // measure surface properties and print "profile"
    p.config_file = par.get("config name");

    par.close(); // file is closed also when par goes out of scope

    // set up the lattice
    lattice.setup(lsize);

    // We need random number here
    hila::seed_random(seed);

    // instantiate the gauge field
    GaugeField<mygroup> H;

    // define the plaquette shifts 
    sw_t<ftype> sw;
    foralldir(d1) {
        foralldir(d2) if(d2>=d1) {
            onsites(ALL) {
                if(d1 == d2) {
                    sw[d1][d2][X] = 0;
                } else {
                    sw[d1][d2][X] = 0.5;
                    if (1) {
                        if (X.parity() == Parity::odd) {
                            sw[d1][d2][X] = -sw[d1][d2][X];
                        }
                        if(1) {
                            if (((int)d1+(int)d2) % 2 == 1) {
                                sw[d1][d2][X] = -sw[d1][d2][X];
                            }
                        }
                    }
                    sw[d2][d1][X] = -sw[d1][d2][X];
                }
            }
        }
    }


    // use negative trajectory numbers for thermalisation
    int start_traj = -p.n_therm;

    if (!restore_checkpoint(H, start_traj, p)) {
        foralldir(d) {
            onsites(ALL) {
                H[d][X] = 0;
            }
        }
    }


    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");

    ftype t_step0 = 0;
    for (int trajectory = start_traj; trajectory < p.n_traj; ++trajectory) {

        ftype ttime = hila::gettime();

        update_timer.start();

        do_trajectory(H, sw, p);

        // put sync here in order to get approx gpu timing
        hila::synchronize_threads();
        update_timer.stop();

        measure_timer.start();

        hila::out0 << "Measure_start " << trajectory << '\n';

        measure_stuff(H, sw);

        hila::out0 << "Measure_end " << trajectory << '\n';

        measure_timer.stop();

        if (p.n_save > 0 && (trajectory + 1) % p.n_save == 0) {
            checkpoint(H, trajectory, p);
        }
    }

    hila::finishrun();
}
