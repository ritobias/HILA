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
    ftype l;            // entangling region slab width
    ftype alpha;        // interpolation parameter
    size_t bcms;
    int n_traj;         // number of trajectories to generate
    int trajlen;        // HMC integration trajectory length
    ftype dt;           // HMC integration time step
    int n_therm;        // number of thermalization trajectories (counts only accepted traj.)
    int n_update;       // number of heat-bath sweeps per "trajectory"
    int n_overrelax;    // number of overrelaxation sweeps per "trajectory"
    int n_save;         // number of trajectories between config. check point
    std::string config_file;
    ftype time_offset;
};

///////////////////////////////////////////////////////////////////////////////////
// HMC functions
// action: S[phi] = \sum_{x} ( -\kappa/2 \sum_{\nu}( \phi(x).\phi(x+\hat{\nu}) + \phi(x).\phi(x-\hat{\nu}) + \phi(x).\phi(x) + \lambda (\phi(x).\phi(x) - 1)^2 )


template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered_p(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d, out_only Field<T> &Sd, const parameters &p) {

    onsites(ALL) {
        if (bcmsid[X] <= p.bcms) {
            Sd[X] = S[1][X + d];
        } else {
            Sd[X] = S[0][X + d];
        }
    }

}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered_k(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d,
                   out_only Field<T> &Sd, const parameters &p) {
    Field<Complex<atype>> tS, tSK;
    Field<Complex<atype>> SK[2];
    SK[0] = 0;
    SK[1].make_ref_to(SK[0], 1);
    double svol = lattice.volume() / lattice.size(e_t);
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
        SK[0] = tS.FFT(e_x + e_y + e_z);
        onsites(ALL) {
            if (bcmsid[X] <= p.bcms) {
                tSK[X] = SK[1][X + d];
            } else {
                tSK[X] = SK[0][X + d];
            }
        }
        tS = tSK.FFT(e_x + e_y + e_z, fft_direction::back) / svol;
        onsites(ALL) {
            Complex<atype> tc = tS[X];
            Sd[X][inc] = tc.real();
            if (inc + 1 < T::size()) {
                Sd[X][inc + 1] = tc.imag();
            }
        }
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void move_filtered(const Field<T> (&S)[2], const Field<pT> &bcmsid, const Direction &d,
                     out_only Field<T> &Sd, const parameters &p) {

    move_filtered_k(S, bcmsid, d, Sd, p);
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_s(const Field<T>(&S)[2], const Field<pT> &bcmsid, const parameters &p) {
    Reduction<double> s = 0;
    s.allreduce(false).delayed(true);
    Field<T> Sd;
    foralldir(d) {
        if(d != e_t) {
            onsites(ALL) {
                s += -p.kappa * S[0][X].dot(S[0][X + d]);
            }
        } else {
            move_filtered(S, bcmsid, d, Sd, p);
            onsites(ALL) {
                s += -p.kappa * S[0][X].dot(Sd[X]);
            }
        }
    }
    onsites(ALL) {
        atype S2 = S[0][X].squarenorm();
        s += S2 + p.lambda * (S2 - 1.0) * (S2 - 1.0);
    }

    return s.value();
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void get_force_add(const Field<T> (&S)[2], const Field<pT> &bcmsid, Field<T> &K, atype eps,
                     const parameters &p) {
    // compute the force
    Field<T> Sd;
    atype th = eps * p.kappa;
    foralldir(d) {
        if (d != e_t) {
            onsites(ALL) K[X] += th * S[0][X + d];

            onsites(ALL) K[X] += th * S[0][X - d];
        } else {
            move_filtered(S, bcmsid, d, Sd, p);
            onsites(ALL) K[X] += th * Sd[X];

            move_filtered(S, bcmsid, -d, Sd, p);
            onsites(ALL) K[X] += th * Sd[X];
        }
    }
    onsites(ALL) {
        atype S2 = S[0][X].squarenorm();
        K[X] += -eps * (2.0 + 4.0 * p.lambda * (S2 - 1.0)) * S[0][X];
    }
}

template <typename T, typename atype = hila::arithmetic_type<T>>
double measure_e2(const Field<T> &E) {
    // compute gauge kinetic energy from momentum field E
    Reduction<double> e2 = 0;
    e2.allreduce(false).delayed(true);
    onsites(ALL) e2 += E[X].squarenorm();
    return e2.value();
}

template <typename T, typename atype = hila::arithmetic_type<T>>
void update_S(Field<T> &S, const Field<T> &E, atype delta) {
    onsites(ALL) {
        S[X] += delta * E[X];
    }
}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
void update_E(const Field<T>(&S)[2], const Field<pT> &bcmsid, Field<T> &E, atype delta, const parameters &p) {
    
    get_force_add(S, bcmsid, E, delta, p);

}

template <typename T, typename pT, typename atype = hila::arithmetic_type<T>>
double measure_action(const Field<T> (&S)[2], const Field<pT> &bcmsid,const Field<T> &E, const parameters &p, atype &s) {
    // measure the total action, consisting of plaquette and momentum term
    s = measure_s(S, bcmsid, p);
    double e2 = measure_e2(E);
    return s + e2 / 2;
}


template <typename T>
void do_hmc_trajectory(Field<T> (&S)[2], const Field<size_t> &bcmsid, Field<T> &E, const parameters &p) {

    // leap frog integration

    // start trajectory: advance U by half a time step
    update_S(S[0], E, p.dt / 2);
    // main trajectory integration:
    for (int n = 0; n < p.trajlen - 1; ++n) {
        update_E(S, bcmsid, E, p.dt, p);
        update_S(S[0], E, p.dt);
    }
    // end trajectory: bring U and E to the same time
    update_E(S, bcmsid, E, p.dt, p);
    update_S(S[0], E, p.dt / 2);

}


// HMC functions
///////////////////////////////////////////////////////////////////////////////////
// measurement functions

template <typename T>
void measure_stuff(Field<T> (&S)[2], const Field<size_t> &bcmsid, const parameters &p) {


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
bool restore_checkpoint(Field<T> &S, int &trajectory, parameters &p) {
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
    // entangling region slab width
    p.l = par.get("slab width");
    // number of trajectories
    p.n_traj = par.get("number of trajectories");
    // hmc trajectory length
    p.trajlen = par.get("hmc trajlen");
    p.dt = par.get("hmc step width");
    // number of thermalization trajectories
    p.n_therm = par.get("thermalization trajs");
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

    p.bcms = (size_t)(p.l * (ftype)Ae);

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
    auto bcmsc = bcms_coordinates(p.bcms, bcmsid);
    //hila::out << "rank " << hila::myrank() << "  : bcmsc=(" << bcmsc << ")\n";



    // use negative trajectory for thermal
    int start_traj = -p.n_therm;

    // Alloc gauge field (S) and gauge momentum field (E)
    Field<myfield> S[2]{0,0};
    Field<myfield> E, S_back;

    if (!restore_checkpoint(S[0], start_traj, p)) {
        S[0] = 0;
    }
    S[1].make_ref_to(S[0], 1);

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");

    onsites(ALL) S_back[X] = S[0][X];

    double act_old, act_new, s_old, s_new;

    s_old = measure_s(S, bcmsid, p);

    for (int trajectory = start_traj; trajectory <= p.n_traj; ++trajectory) {

        ftype ttime = hila::gettime();

        update_timer.start();

        onsites(ALL) E[X].gaussian_random();

        act_old = s_old + measure_e2(E) / 2;

        do_hmc_trajectory(S, bcmsid, E, p);

        act_new = measure_action(S, bcmsid, E, p, s_new);

        bool reject = hila::broadcast(exp(act_old - act_new) < hila::random());

        hila::out0 << std::setprecision(12) << "HMC " << trajectory << " S_TOT_start " << act_old
                   << " dS_TOT " << std::setprecision(6) << act_new - act_old
                   << std::setprecision(12);
        if (reject) {
            hila::out0 << " REJECT" << " --> S " << s_old;
            onsites(ALL) S[0][X] = S_back[X];
        } else {
            hila::out0 << " ACCEPT" << " --> S " << s_new;
            s_old = s_new;
        }

        update_timer.stop();

        hila::out0 << "  time " << std::setprecision(3) << hila::gettime() - ttime << '\n';

        measure_timer.start();

        hila::out0 << "Measure_start " << trajectory << '\n';

        measure_stuff(S, bcmsid, p);

        hila::out0 << "Measure_end " << trajectory << '\n';

        measure_timer.stop();

        if (p.n_save > 0 && (trajectory + 1) % p.n_save == 0) {
            checkpoint(S[0], trajectory, p);
        }
    }

    hila::finishrun();
}
