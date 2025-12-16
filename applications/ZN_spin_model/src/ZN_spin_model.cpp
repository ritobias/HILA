#include "hila.h"
#include "tools/string_format.h"
#include "tools/floating_point_epsilon.h"
#include <algorithm>

#ifndef NCOLOR
#define NCOLOR 3
#endif

using ftype = double;
using fT = int;

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    ftype betac;        // clock model coupling
    ftype betap;        // Potts model coupling
    ftype source;       // source J
    int source_state;   // source state
    int n_traj;         // number of trajectories to generate
    int n_therm;        // number of thermalization trajectories (counts only accepted traj.)
    int n_heatbath;     // number of heat-bath sweeps per "trajectory"
    int n_multhits;     // number of corrected heat-bath hits per update
    int n_save;         // number of trajectories between config. check point
    int n_dump_conf;    // number of trajectories between dumped configs (0 = off)
    int n_profile;
    ftype smear_param;
    std::vector<int> n_smear;
    std::string config_file;
    ftype time_offset;
};


///////////////////////////////////////////////////////////////////////////////////
// Z_N spin model action
// action: S[s] = \sum_{x} ( -\beta / 2 * \sum_{\nu}( s_{x} * s^{*}_{x+\hat{\nu}} +
// s^{*}_{x} * s_{x+\hat{\nu}} ) - J / 2 * (s^{*}_{x} + s_{x}) )
//
///////////////////////////////////////////////////////////////////////////////////
// heat-bath functions

template <typename nnT, typename atype = hila::arithmetic_type<nnT>>
void ZN_heatbath(int &S, const nnT &nnsum, const Vector<NCOLOR,int> &nnlist, ftype betac, ftype betap, ftype source, int source_state) {
    atype nnsumabs = abs(nnsum);
    atype nnsumarg = arg(nnsum);

    atype wl[NCOLOR];
    atype wlsum = 0.0;
    for (int i = 0; i < NCOLOR; ++i) {
        wl[i] = exp(betac * nnsumabs * cos(2.0 * M_PI * (ftype)i / (ftype)NCOLOR - nnsumarg) + betap * nnlist[i]  + source * cos(2.0 * M_PI * (ftype)(i - source_state) / (ftype)NCOLOR));
        wlsum += wl[i];
    }
    atype rand_val = hila::random();
    atype totw = 0.0;
    for (int i = 0; i < NCOLOR; ++i) {
        totw += wl[i] / wlsum;
        if(rand_val < totw) {
            S = i;
            break;
        }
    }
}


/**
 * @brief Wrapper function to updated Z_N scalar field per paraity
 * @details --
 *
 * @tparam T field type
 * @param S field
 * @param shift vectorfield to implement twist
 * @param p parameters
 * @param par Parity specifies parity of sites to be updated
 */
template <typename T>
void hb_update_parity(Field<T> &S, const VectorField<T> &shift, const parameters &p, Parity par) {

    static Complex<ftype> sp_state[NCOLOR];
    static bool first = true;
    if(first) {
        first = false;
        for (int i = 0; i < NCOLOR; ++i) {
            sp_state[i] = Complex<ftype>(cos(2.0 * M_PI * (ftype)i / (ftype)NCOLOR),
                                        sin(2.0 * M_PI * (ftype)i / (ftype)NCOLOR));
        }
    }

    Field<Complex<ftype>> nnsum = 0;
    Field<Vector<NCOLOR, int>> nnlist(0);
    foralldir(d) {
        S.start_gather(-d);
        shift[d].start_gather(-d);
        onsites(par) {
            T tval = S[X + d] - shift[d][X];
            nnsum[X] += sp_state[(tval + NCOLOR) % NCOLOR];
            nnlist[X][tval] += 1;
        }

        onsites(par) {
            T tval = S[X - d] + shift[d][X - d];
            nnsum[X] += sp_state[(tval + NCOLOR) % NCOLOR];
            nnlist[X][tval] += 1;
        }
    }

    onsites(par) {
        for (int i = 0; i < p.n_multhits; ++i) {
            ZN_heatbath(S[X], nnsum[X], nnlist[X], p.betac, p.betap, p.source, p.source_state);
        }
    }
}

/**
 * @brief Wrapper update function
 * @details field update sweep with randomly chosen spatial and temporal parities 
 *
 * @tparam T field type
 * @param S field
 * @param p parameters
 */
template <typename T>
void hb_update(Field<T> &S, const VectorField<T> &shift, const parameters &p) {
    std::array<int, 2> rnarr;
    for (int i = 0; i < 2; ++i) {
        // randomly choose parity:
        rnarr[i] = (int)(hila::random() * 2);
    }
    hila::broadcast(rnarr);

    for (int i = 0; i < 2; ++i) {
        int tpar = 1 + rnarr[i];

        // perform the selected updates:
        hb_update_parity(S, shift, p, Parity(tpar));
    }
}

/**
 * @brief Evolve field
 * @details Evolution happens by means of a metropolis corrected heatbath algorithm. We do on average
 * p.n_heatbath heatbath updates on each site per sweep.
 *
 * @tparam T field type
 * @param S Field
 * @param p parameters
 */
template <typename T>
void do_hb_trajectory(Field<T> &S, const VectorField<T> &shift, const parameters &p) {
    for (int n = 0; n < p.n_heatbath; n++) {
        hb_update(S, shift, p);
    }
}


// heat-bath functions
///////////////////////////////////////////////////////////////////////////////////
// measurement functions

template <typename sT>
void spectraldensity_surface(std::vector<sT> &surf, int size_x, int size_y,
                             std::vector<double> &npow, std::vector<int> &hits) {

    // do fft for the surface
    static bool first = true;

    static Complex<double> *buf;
    static fftw_plan fftwplan;

    int area = size_x * size_y;

    if (first) {
        first = false;

        buf = (Complex<double> *)fftw_malloc(sizeof(Complex<double>) * area);

        // note: we had x as the "fast" dimension, but fftw wants the 2nd dim to be
        // the "fast" one. thus, first y, then x.
        fftwplan = fftw_plan_dft_2d(size_y, size_x, (fftw_complex *)buf, (fftw_complex *)buf,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    }

    for (int i = 0; i < area; i++) {
        buf[i] = surf[i];
    }

    fftw_execute(fftwplan);

    int pow_size = npow.size();

    for (int i = 0; i < area; i++) {
        int x = i % size_x;
        int y = i / size_x;
        x = (x <= size_x / 2) ? x : (size_x - x);
        y = (y <= size_y / 2) ? y : (size_y - y);

        int k = x * x + y * y;
        if (k < pow_size) {
            npow[k] += buf[i].squarenorm() / (area * area);
            hits[k]++;
        }
    }
}

template <typename T>
T proj_to_nrange(T inval) {
    while (inval > (T)NCOLOR / 2) {
        inval -= (T)NCOLOR;
    }
    while (inval <= -(T)(NCOLOR + 1) / 2) {
        inval += (T)NCOLOR;
    }
    return inval;
}

template <typename T>
T proj_to_nrange(T inval, T tol) {
    while (inval >= (T)NCOLOR / 2 + tol) {
        inval -= (T)NCOLOR;
    }
    while (inval <= -((T)NCOLOR / 2 + tol)) {
        inval += (T)NCOLOR;
    }
    return inval;
}


template <typename T, typename sT, typename atype = hila::arithmetic_type<sT>>
void sm_ready_field(const Field<T> &S, out_only Field<sT> &smS, out_only sT &smSmean) {
    Reduction<double> smStot = 0;
    smStot.allreduce(true).delayed(true);
    onsites(ALL) {
        smS[X] = (sT)proj_to_nrange(S[X]);
        smStot += smS[X];
    }
    smSmean = smStot.value() / lattice.volume();
}


template <typename T, typename sT>
void smear_field(Field<sT> &smS, const VectorField<T> &shift, sT smear_param, int n_smear) {
    int ip = 0;
    Field<sT> tsmS[2];
    onsites(ALL) {
        tsmS[ip][X] = smS[X];
    }
    for (int ism = 0; ism < n_smear; ++ism) {
        onsites(ALL) {
            tsmS[1 - ip][X] = tsmS[ip][X] * (1.0 + 2.0 * (sT)NDIM * smear_param);
        }
        foralldir(d) {
            onsites(ALL) {
                tsmS[1 - ip][X] +=
                    smear_param * proj_to_nrange(tsmS[ip][X + d] - (sT)shift[d][X] - tsmS[ip][X], (sT)0.5);

                tsmS[1 - ip][X] += smear_param * proj_to_nrange(tsmS[ip][X - d] +
                                                                (sT)shift[d][X - d] - tsmS[ip][X], (sT)0.5);
            }
        }
        onsites(ALL) {
            tsmS[1 - ip][X] =
                proj_to_nrange(tsmS[1 - ip][X] / (1.0 + 2.0 * (sT)NDIM * smear_param), (sT)0.5);
        }
        ip = 1 - ip;
    }
    onsites(ALL) {
        smS[X] = tsmS[ip][X];
    }
}


template <typename T, typename sT>
void smear_field(Field<sT> &smS, const VectorField<T> &shift, sT smear_param, int n_smear, sT smSmean) {
    int ip = 0;
    Field<sT> tsmS[2];
    onsites(ALL) {
        tsmS[ip][X] = proj_to_nrange(smS[X] - smSmean);
        //tsmS[ip][X] = smS[X] - smSmean;
    }
    for (int ism = 0; ism < n_smear; ++ism) {
        onsites(ALL) {
            tsmS[1 - ip][X] = tsmS[ip][X];
        }
        foralldir(d) {
            onsites(ALL) {
                tsmS[1 - ip][X] +=
                    smear_param * (tsmS[ip][X + d] - (sT)shift[d][X]);

                tsmS[1 - ip][X] +=
                    smear_param * (tsmS[ip][X - d] + (sT)shift[d][X - d]);
            }
        }
        onsites(ALL) {
            tsmS[1 - ip][X] = tsmS[1 - ip][X] / (1.0 + 2.0 * (sT)NDIM * smear_param);
        }
        ip = 1 - ip;
    }
    onsites(ALL) {
        smS[X] = proj_to_nrange(tsmS[ip][X] + smSmean);
        //smS[X] = tsmS[ip][X] + smSmean;
    }
}


template <typename sT>
void measure_profile(const Field<sT> &smS, Direction d, std::vector<sT> &profile) {
    ReductionVector<sT> p(lattice.size(d));
    p.allreduce(false).delayed(true);
    sT area = (sT)(lattice.volume() / lattice.size(d));
    onsites(ALL) {
        p[X.coordinate(d)] += smS[X] / area;
    }
    p.reduce();
    profile = p.vector();
}

int z_ind(int z, Direction dz) {
    return (z + lattice.size(dz)) % lattice.size(dz);
}


template <typename sT>
void measure_interface(const Field<sT> &smS, Direction dz, std::vector<sT> &surf, out_only int &size_x, out_only int &size_y) {

    Direction dx, dy;
    foralldir(td) if (td != dz) {
        dx = td;
        break;
    }
    foralldir(td) if (td != dz && td != dx) {
        dy = td;
        break;
    }

    size_x = lattice.size(dx);
    size_y = lattice.size(dy);

    int area = size_x * size_y;

    if (hila::myrank() == 0) {
        surf.resize(area);
    }

    std::vector<sT> profile;
    measure_profile(smS, dz, profile);

    sT minp = 1.0e8, maxp = -1.0e8;
    int minloc = 0, maxloc = 0;
    for (int i = 0; i < profile.size(); i++) {
        if (minp > profile[i]) {
            minp = profile[i];
            minloc = i;
        }
        if (maxp < profile[i]) {
            maxp = profile[i];
            maxloc = i;
        }
    }

    int startloc = (minloc + maxloc) / 2;

    sT surface_level = 0.5 * (sT)(std::round(minp) + std::round(maxp));


    int ddz = 1;
    if (minloc > maxloc) {
        ddz = -1;
    }

    std::vector<sT> lsmS;
    std::vector<sT> line(lattice.size(dz));


    CoordinateVector yslice;
    foralldir(td) {
        yslice[td] = -1;
    }
    for (int y = 0; y < size_y; ++y) {
        yslice[dy] = y;
        lsmS = smS.get_slice(yslice);
        if (hila::myrank() == 0) {
            for (int x = 0; x < size_x; ++x) {

                for (int z = 0; z < lattice.size(dz); z++) {
                    line[z] = lsmS[x + size_x * z];
                }

                // if (hila::myrank() == 0) {
                // start search of the surface from the center between min and max
                int z = startloc;

                while (line[z_ind(z, dz)] > surface_level && ddz * (startloc - z) < lattice.size(dz) * 0.4) {
                    z -= ddz;
                }

                while (line[z_ind(z + ddz, dz)] <= surface_level && ddz * (z - startloc) < lattice.size(dz) * 0.4) {
                    z += ddz;
                }


                // do linear interpolation
                surf[x + y * size_x] =
                    (sT)z + (sT)ddz * (surface_level - line[z_ind(z, dz)]) / (line[z_ind(z + ddz, dz)] - line[z_ind(z, dz)]);

            }
        }
    }
}


template <typename sT>
void measure_interface_spectrum(Field<sT> &smS, Direction dir_z, int bds_shift, int nsm,
                                int64_t idump, parameters &p, bool first_pow) {
    std::vector<ftype> surf;
    int size_x, size_y;
    measure_interface(smS, dir_z, surf, size_x, size_y);
    if (hila::myrank() == 0) {
        if(false) {
            for (int x = 0; x < size_x; ++x) {
                for (int y = 0; y < size_y; ++y) {
                    hila::out0 << "SURF" << nsm << ' ' << x << ' ' << y << ' ' << surf[x + y * size_x] << '\n';
                }
            }
        }
        constexpr int pow_size = 200;
        std::vector<double> npow(pow_size);
        std::vector<int> hits(pow_size);
        spectraldensity_surface(surf, size_x, size_y, npow, hits);

        std::string prof_file = string_format("profile_nsm%04d", nsm);
        std::ofstream prof_os;

        if (first_pow) {
            int64_t npowmeas = 0;
            for (int ipow = 0; ipow < pow_size; ++ipow) {
                if (hits[ipow] > 0) {
                    ++npowmeas;
                }
            }

            if (filesys_ns::exists(prof_file)) {
                std::string prof_file_temp = prof_file + "_temp";
                filesys_ns::rename(prof_file, prof_file_temp);
                std::ifstream ifile;
                ifile.open(prof_file_temp, std::ios::in | std::ios::binary);
                prof_os.open(prof_file, std::ios::out | std::ios::binary);

                int64_t *ibuff = (int64_t *)memalloc((NDIM + 5) * sizeof(int64_t));
                ifile.read((char *)ibuff, (NDIM + 5) * sizeof(int64_t));
                prof_os.write((char *)ibuff, (NDIM + 5) * sizeof(int64_t));
                int64_t tnpowmeas = ibuff[NDIM + 2];
                double *buffer = nullptr;
                if (tnpowmeas > npowmeas) {
                    buffer = (double *)memalloc(tnpowmeas * sizeof(double));
                } else {
                    buffer = (double *)memalloc(npowmeas * sizeof(double));
                    for (int tib = tnpowmeas; tib < npowmeas; ++tib) {
                        buffer[tib] = 0;
                    }
                }
                ifile.read((char *)buffer, sizeof(double));
                prof_os.write((char *)buffer, sizeof(double));

                ifile.read((char *)buffer, tnpowmeas * sizeof(double));

                npowmeas = 0;
                for (int ipow = 0; ipow < pow_size; ++ipow) {
                    if (hits[ipow] > 0) {
                        buffer[npowmeas] = (double)ipow;
                        ++npowmeas;
                    }
                }
                prof_os.write((char *)buffer, npowmeas * sizeof(double));

                while (ifile.good()) {
                    ifile.read((char *)ibuff, sizeof(int64_t));
                    int64_t tidump = ibuff[0];
                    ifile.read((char *)buffer, tnpowmeas * sizeof(double));
                    if (ifile.good() && tidump < idump) {
                        prof_os.write((char *)&(tidump), sizeof(int64_t));
                        prof_os.write((char *)buffer, npowmeas * sizeof(double));
                    } else {
                        break;
                    }
                }
                ifile.close();
                prof_os.close();
                free(buffer);
                free(ibuff);
                filesys_ns::remove(prof_file_temp);
            } else {
                prof_os.open(prof_file, std::ios::out | std::ios::binary);
                int64_t *ibuff = (int64_t *)memalloc((NDIM + 5) * sizeof(int64_t));
                ibuff[0] = NCOLOR;
                ibuff[1] = NDIM;
                foralldir(d) {
                    ibuff[2 + (int)d] = lattice.size(d);
                }
                ibuff[NDIM + 2] = npowmeas;
                ibuff[NDIM + 3] = nsm;
                ibuff[NDIM + 4] = sizeof(double);
                prof_os.write((char *)ibuff, (NDIM + 5) * sizeof(int64_t));
                double tval = p.smear_param;
                prof_os.write((char *)&(tval), sizeof(double));

                double *buffer = (double *)memalloc(npowmeas * sizeof(double));
                for (int tib = npowmeas; tib < npowmeas; ++tib) {
                    buffer[tib] = 0;
                }
                npowmeas = 0;
                for (int ipow = 0; ipow < pow_size; ++ipow) {
                    if (hits[ipow] > 0) {
                        buffer[npowmeas] = (double)ipow;
                        ++npowmeas;
                    }
                }
                prof_os.write((char *)buffer, npowmeas * sizeof(double));

                prof_os.close();
                free(buffer);
                free(ibuff);
            }
        }

        prof_os.open(prof_file, std::ios::out | std::ios_base::app | std::ios::binary);

        prof_os.write((char *)&(idump), sizeof(int64_t));
        for (int ipow = 0; ipow < pow_size; ++ipow) {
            if (hits[ipow] > 0) {
                double tval = npow[ipow] / hits[ipow];
                prof_os.write((char *)&(tval), sizeof(double));
            }
        }
        prof_os.close();
    }
}


template <typename T>
Complex<T> phase_to_complex(T arg) {
    return Complex<T>(cos(arg), sin(arg));
}


template <typename sT>
void measure_interface_ft(const Field<sT> &smS, Direction dz, int bds_shift, std::vector<sT> &surf,
                          out_only int &size_x, out_only int &size_y) {

    Direction dx, dy;
    foralldir(td) if (td != dz) {
        dx = td;
        break;
    }
    foralldir(td) if (td != dz && td != dx) {
        dy = td;
        break;
    }

    size_x = lattice.size(dx);
    size_y = lattice.size(dy);

    int area = size_x * size_y;

    static std::vector<Complex<ftype>> ft_basis;
    static bool first = true;

    if (first) {
        first = false;
        ft_basis.resize(lattice.size(dz));
        for (int iz = 0; iz < lattice.size(dz); ++iz) {
            ftype ftarg =
                2.0 * M_PI * (ftype)iz / (ftype)lattice.size(dz) * (ftype)bds_shift / (ftype)NCOLOR;
            ft_basis[iz] = phase_to_complex(ftarg);
        }
    }

    if (hila::myrank() == 0) {
        surf.resize(area);
    }

    std::vector<sT> lS;

    CoordinateVector yslice;
    foralldir(td) {
        yslice[td] = -1;
    }
    Complex<sT> ftn;
    for (int y = 0; y < size_y; ++y) {
        yslice[dy] = y;
        lS = smS.get_slice(yslice);
        if (hila::myrank() == 0) {
            for (int x = 0; x < size_x; ++x) {
                ftn = 0;
                for (int z = 0; z < lattice.size(dz); z++) {
                    ftn += phase_to_complex(2.0 * M_PI * (ftype)lS[x + size_x * z] / (ftype)NCOLOR) * ft_basis[z];
                }

                surf[x + y * size_x] = (ftype)arg(ftn) * ((ftype)NCOLOR * (ftype)lattice.size(dz)) /
                                           (2.0 * M_PI * bds_shift) +
                                       (ftype)(lattice.size(dz) + 1) / 2;
            }
        }
    }
}


template <typename sT>
void measure_interface_spectrum_ft(Field<sT> &smS, Direction dir_z, int bds_shift, int nsm,
                                int64_t idump, parameters &p, bool first_pow) {
    std::vector<ftype> surf;
    int size_x, size_y;
    measure_interface_ft(smS, dir_z, bds_shift, surf, size_x, size_y);
    if (hila::myrank() == 0) {
        if (false) {
            for (int x = 0; x < size_x; ++x) {
                for (int y = 0; y < size_y; ++y) {
                    hila::out0 << "SURFFT" << nsm << ' ' << x << ' ' << y << ' '
                               << surf[x + y * size_x] << '\n';
                }
            }
        }
        constexpr int pow_size = 200;
        std::vector<double> npow(pow_size);
        std::vector<int> hits(pow_size);
        spectraldensity_surface(surf, size_x, size_y, npow, hits);

        std::string prof_file = string_format("profile_ft_nsm%04d", nsm);
        std::ofstream prof_os;

        if (first_pow) {
            int64_t npowmeas = 0;
            for (int ipow = 0; ipow < pow_size; ++ipow) {
                if (hits[ipow] > 0) {
                    ++npowmeas;
                }
            }

            if (filesys_ns::exists(prof_file)) {
                std::string prof_file_temp = prof_file + "_temp";
                filesys_ns::rename(prof_file, prof_file_temp);
                std::ifstream ifile;
                ifile.open(prof_file_temp, std::ios::in | std::ios::binary);
                prof_os.open(prof_file, std::ios::out | std::ios::binary);

                int64_t *ibuff = (int64_t *)memalloc((NDIM + 5) * sizeof(int64_t));
                ifile.read((char *)ibuff, (NDIM + 5) * sizeof(int64_t));
                prof_os.write((char *)ibuff, (NDIM + 5) * sizeof(int64_t));
                int64_t tnpowmeas = ibuff[NDIM + 2];
                double *buffer = nullptr;
                if (tnpowmeas > npowmeas) {
                    buffer = (double *)memalloc(tnpowmeas * sizeof(double));
                } else {
                    buffer = (double *)memalloc(npowmeas * sizeof(double));
                    for (int tib = tnpowmeas; tib < npowmeas; ++tib) {
                        buffer[tib] = 0;
                    }
                }
                ifile.read((char *)buffer, sizeof(double));
                prof_os.write((char *)buffer, sizeof(double));

                ifile.read((char *)buffer, tnpowmeas * sizeof(double));

                npowmeas = 0;
                for (int ipow = 0; ipow < pow_size; ++ipow) {
                    if (hits[ipow] > 0) {
                        buffer[npowmeas] = (double)ipow;
                        ++npowmeas;
                    }
                }
                prof_os.write((char *)buffer, npowmeas * sizeof(double));

                while (ifile.good()) {
                    ifile.read((char *)ibuff, sizeof(int64_t));
                    int64_t tidump = ibuff[0];
                    ifile.read((char *)buffer, tnpowmeas * sizeof(double));
                    if (ifile.good() && tidump < idump) {
                        prof_os.write((char *)&(tidump), sizeof(int64_t));
                        prof_os.write((char *)buffer, npowmeas * sizeof(double));
                    } else {
                        break;
                    }
                }
                ifile.close();
                prof_os.close();
                free(buffer);
                free(ibuff);
                filesys_ns::remove(prof_file_temp);
            } else {
                prof_os.open(prof_file, std::ios::out | std::ios::binary);
                int64_t *ibuff = (int64_t *)memalloc((NDIM + 5) * sizeof(int64_t));
                ibuff[0] = NCOLOR;
                ibuff[1] = NDIM;
                foralldir(d) {
                    ibuff[2 + (int)d] = lattice.size(d);
                }
                ibuff[NDIM + 2] = npowmeas;
                ibuff[NDIM + 3] = nsm;
                ibuff[NDIM + 4] = sizeof(double);
                prof_os.write((char *)ibuff, (NDIM + 5) * sizeof(int64_t));
                double tval = p.smear_param;
                prof_os.write((char *)&(tval), sizeof(double));

                double *buffer = (double *)memalloc(npowmeas * sizeof(double));
                for (int tib = npowmeas; tib < npowmeas; ++tib) {
                    buffer[tib] = 0;
                }
                npowmeas = 0;
                for (int ipow = 0; ipow < pow_size; ++ipow) {
                    if (hits[ipow] > 0) {
                        buffer[npowmeas] = (double)ipow;
                        ++npowmeas;
                    }
                }
                prof_os.write((char *)buffer, npowmeas * sizeof(double));

                prof_os.close();
                free(buffer);
                free(ibuff);
            }
        }

        prof_os.open(prof_file, std::ios::out | std::ios_base::app | std::ios::binary);

        prof_os.write((char *)&(idump), sizeof(int64_t));
        for (int ipow = 0; ipow < pow_size; ++ipow) {
            if (hits[ipow] > 0) {
                double tval = npow[ipow] / hits[ipow];
                prof_os.write((char *)&(tval), sizeof(double));
            }
        }
        prof_os.close();
    }
}


template <typename T, typename vfT>
double measure_s(const Field<T> &S, const VectorField<T> &shift, const parameters &p,
                 out_only double &s_kin, out_only vfT &vol_fracs) {
    Reduction<double> s = 0, s_source = 0;
    s.allreduce(false).delayed(true);
    s_source.allreduce(false).delayed(true);
    ReductionVector<double> volfr(NCOLOR, 0);
    volfr.allreduce(false).delayed(true);
    // nn terms in all directions:
    foralldir(d) {
        onsites(ALL) {
            s += -p.betac * cos(2.0 * M_PI * (ftype)(S[X] - S[X + d] + shift[d][X]) / (ftype)NCOLOR);
            s += -p.betap * (S[X] == (S[X + d] - shift[d][X]));
        }
    }
    // potential term:
    onsites(ALL) {
        s_source += -p.source * cos(2.0 * M_PI * (ftype)(S[X] - p.source_state) / (ftype)NCOLOR);
        volfr[S[X]] += 1.0;
    }
    volfr.reduce();
    for (int i = 0; i < NCOLOR; ++i) {
        vol_fracs[i] = volfr[i]/lattice.volume();
    }
    s_kin = s.value() / lattice.volume();
    return s_kin + s_source.value() / lattice.volume();
}

template <typename T>
void measure_stuff(const Field<T> &S, const VectorField<T> &shift, const parameters &p) {

    static bool first = true;
    if (first) {
        // print legend for measurement output
        hila::out0 << "LMEAS:         s_kin             s\n";
        hila::out0 << "LVPST:";
        for (int i = 0; i < NCOLOR; ++i) {
            hila::out0 << string_format("  st_%02d", i);
        }
        hila::out0 << "\n";
        first = false;
    }
    std::vector<double> vol_fracs(NCOLOR);
    double s_kin;
    auto s = measure_s(S, shift, p, s_kin, vol_fracs);

    hila::out0 << string_format("MEAS % 0.8e % 0.8e\n", s_kin, s);
    hila::out0 << "VPST: ";
    for (int i = 0; i < NCOLOR; ++i) {
        hila::out0 << string_format(" % 0.4e", vol_fracs[i]);
    }
    hila::out0 << "\n";
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

int main(int argc, char **argv) {

    // hila::initialize should be called as early as possible
    hila::initialize(argc, argv);

    hila::out0 << "Z_N spin model simulation\n";

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
    // clock model coupling
    p.betac = par.get("betac");
    // Potts model coupling
    p.betap = par.get("betap");
    // source term
    p.source = par.get("source term magnitude");
    p.source_state = par.get("source term direction");
    // boundary shift
    int bds_dir = par.get("bds direction");
    int bds_shift = par.get("bds shift");
    // number of trajectories
    p.n_traj = par.get("number of trajectories");
    // number of heat-bath (HB) sweeps per trajectory
    p.n_heatbath = par.get("heatbath updates");
    // number of multi-hits per HB update
    p.n_multhits = par.get("number of hb hits");
    // number of thermalization trajectories
    p.n_therm = par.get("thermalization trajs");
    // random seed = 0 -> get seed from time
    long seed = par.get("random seed");
    // save config and checkpoint
    p.n_save = par.get("trajs/saved");
    // measure surface properties and print "profile"
    p.config_file = par.get("config name");
    // dump configurations after every n_dump_conf trajectories:
    p.n_profile = par.get("trajs/profile measure");
    p.n_dump_conf = par.get("trajs/config dump");
    p.smear_param = par.get("smearing coeff");
    p.n_smear = par.get("smearing steps");

    par.close(); // file is closed also when par goes out of scope


    // set up the lattice
    lattice.setup(lsize);

    // We need random number here
    if (!hila::is_rng_seeded())
        hila::seed_random(seed);

    // use negative trajectory for thermal
    int start_traj = -p.n_therm;

    Direction dir_z = Direction(bds_dir);


    // Alloc and define boundary shift vector field
    VectorField<fT> shift(0);
    onsites(ALL) {
        if (X.coordinate(dir_z) == lattice.size(dir_z) - 1) {
            shift[dir_z][X] = bds_shift;
        } else {
            shift[dir_z][X] = 0;
        }
    }


    // Alloc field
    Field<fT> S;
    if (!restore_checkpoint(S, start_traj, p)) {
        S = 0;
        onsites(ALL) {
            if (X.coordinate(dir_z) < lattice.size(dir_z) / 2) {
                S[X] = 0;
            } else {
                S[X] = NCOLOR - ((bds_shift + NCOLOR) % NCOLOR);
            }
        }
    }

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");
    bool first_pow = true;
    for (int trajectory = start_traj; trajectory <= p.n_traj; ++trajectory) {

        ftype ttime = hila::gettime();

        update_timer.start();

        do_hb_trajectory(S, shift, p);

        // put sync here in order to get approx gpu timing
        hila::synchronize_threads();
        update_timer.stop();

        measure_timer.start();

        hila::out0 << "Measure_start " << trajectory << '\n';

        measure_stuff(S, shift, p);
        if (trajectory + 1 >= 0) {

            if (p.n_profile && (trajectory + 1) % p.n_profile == 0) {

                int64_t idump = (trajectory + 1) / p.n_profile;

                Field<ftype> smS;
                ftype smSmean;
                sm_ready_field(S, smS, smSmean);
                smSmean = -0.5 * (ftype)bds_shift;

                int n_smear = 0;

                for (int ism = 0; ism < p.n_smear.size(); ++ism) {
                    smear_field(smS, shift, p.smear_param, p.n_smear[ism] - n_smear, smSmean);
                    n_smear = p.n_smear[ism];

                    if (p.n_dump_conf && (trajectory + 1) % p.n_dump_conf == 0) {
                        int icdump = (trajectory + 1) / p.n_dump_conf;
                        std::string dump_file =
                            string_format("fdump_%04d_nsm%04d", icdump, n_smear);
                        smS.config_write(dump_file);
                    }

                    if (p.n_profile && (trajectory + 1) % p.n_profile == 0) {
                        measure_interface_spectrum_ft(smS, dir_z, bds_shift, n_smear, idump, p,
                                                      first_pow);
                        measure_interface_spectrum(smS, dir_z, bds_shift, n_smear, idump, p,
                                                      first_pow);
                    }
                }
                first_pow = false;
            }
        }

        hila::out0 << "Measure_end " << trajectory << '\n';

        measure_timer.stop();

        if (p.n_save > 0 && (trajectory + 1) % p.n_save == 0) {
            checkpoint(S, trajectory, p);
        }
    }

    hila::finishrun();
}
