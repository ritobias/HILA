#include "hila.h"
#include "gauge/staples.h"
#include "gauge/polyakov.h"
#include "gauge/stout_smear.h"

#include "gauge/sun_heatbath.h"
#include "gauge/sun_overrelax.h"
#include "checkpoint.h"
#include "tools/string_format.h"
#include <fftw3.h>

#ifndef NCOLOR
#define NCOLOR 3
#endif

using ftype = float;
using mygroup = SU<NCOLOR, ftype>;

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    double beta;
    int n_overrelax;
    int n_heatbath;
    int n_trajectories;
    int n_thermal;
    int n_save;
    int n_profile;
    int n_surf_spec;
    std::string config_file;
    double time_offset;
    std::vector<int> n_smear;
    double smear_coeff;
    std::vector<int> z_smear;
    int n_surface;
    int n_dump_polyakov;
};

////////////////////////////////////////////////////////////////

/**
 * @brief Wrapper function to updated GaugeField per direction
 * @details Computes first staplesum, then uses computed result to evolve GaugeField either with
 * over relaxation or heat bath
 *
 * @tparam group
 * @param U GaugeField to evolve
 * @param p parameter struct
 * @param par Parity
 * @param d Direction to evolve
 * @param relax If true evolves GaugeField with over relaxation if false then with heat bath
 * @param plaqw plaquette weights
 */
template <typename group, typename wT>
void update_parity_dir(GaugeField<group> &U, const parameters &p, Parity par, Direction d,
                       bool relax, const plaqw_t<wT> &plaqw) {

    static hila::timer hb_timer("Heatbath");
    static hila::timer or_timer("Overrelax");
    static hila::timer staples_timer("Staplesum");

    double accr = 0.0;

    Field<group> staples;

    staples_timer.start();

    staplesum(U, staples, d, plaqw, par);

    staples_timer.stop();

    if (relax) {

        or_timer.start();

        onsites(par) {
#ifdef SUN_OVERRELAX_dFJ
            suN_overrelax_dFJ(U[d][X], staples[X], p.beta);
#else
            suN_overrelax(U[d][X], staples[X]);
            //suN_overrelax(U[d][X], staples[X], p.beta);
#endif
        }
        or_timer.stop();

    } else {

        hb_timer.start();
        onsites(par) {
            suN_heatbath(U[d][X], staples[X], p.beta);
        }
        hb_timer.stop();
    }
}


////////////////////////////////////////////////////////////////


template <typename group, typename wT>
void update(GaugeField<group> &U, const parameters &p, const plaqw_t<wT> &plaqw) {

    std::array<int, 2 * NDIM> rnarr;
    for (int i = 0; i < 2 * NDIM; ++i) {
        // randomly choose a parity and a direction:
        rnarr[i] = (int)(hila::random() * 2 * NDIM);

        // randomly choose whether to do overrelaxation or heatbath update (with probabilities
        // according to specified p.n_heatbath and p.n_overrelax):
        if (hila::random() >= (double)p.n_heatbath / (p.n_heatbath + p.n_overrelax)) {
            rnarr[i] += 2 * NDIM;
        }
    }
    hila::broadcast(rnarr);

    for (int i = 0; i < 2 * NDIM; ++i) {
        bool relax = false;
        int tdp = rnarr[i];
        if (tdp >= 2 * NDIM) {
            tdp -= 2 * NDIM;
            relax = true;
        }
        int tdir = tdp / 2;
        int tpar = 1 + (tdp % 2);

        // perform the selected updates:
        update_parity_dir(U, p, Parity(tpar), Direction(tdir), relax, plaqw);
    }
}


////////////////////////////////////////////////////////////////

template <typename group, typename wT>
void do_trajectory(GaugeField<group> &U, const parameters &p, const plaqw_t<wT> &plaqw) {

    for (int n = 0; n < p.n_heatbath + p.n_overrelax; n++) {
        update(U, p, plaqw);
    }
    U.reunitarize_gauge();
}


template <typename group, typename wT>
double measure_plaq(const GaugeField<group> &U, const plaqw_t<wT> &plaqw) {
    Reduction<double> plaq;
    plaq.allreduce(false);


    foralldir(dir1) foralldir(dir2) if (dir1 < dir2) {
        onsites(ALL) {
            plaq += 1.0 - real(plaqw[dir1][dir2][X] *
                                trace(U[dir1][X] * U[dir2][X + dir1] *
                                        U[dir1][X + dir2].dagger() * U[dir2][X].dagger())) /
                                group::size();
        }
    }

    return plaq.value();
}

template <typename group, typename wT>
double measure_action(const GaugeField<group> &U, const parameters &p, const plaqw_t<wT> &plaqw) {

    auto plaq = measure_plaq(U, p, plaqw);

    return p.beta * plaq;
}

/////////////////////////////////////////////////////////////////////////////
/// Measure polyakov "field"
///

template <typename T, typename pT>
void measure_polyakov_field(const Field<T> &Ut, Field<pT> &pl) {
    Field<T> polyakov = Ut;

    // mult links so that polyakov[X.dir == 0] contains the polyakov loop
    for (int plane = lattice.size(e_t) - 2; plane >= 0; plane--) {

        // safe_access(polyakov) pragma allows the expression below, otherwise
        // hilapp would reject it because X and X+dir can refer to the same
        // site on different "iterations" of the loop.  However, here this
        // is restricted on single dir-plane so it works but we must tell it to hilapp.

#pragma hila safe_access(polyakov)
        onsites(ALL) {
            if (X.coordinate(e_t) == plane) {
                polyakov[X] = Ut[X] * polyakov[X + e_t];
            }
        }
    }

    onsites(ALL) if (X.coordinate(e_t) == 0) {
        pl[X] = (pT)real(trace(polyakov[X]));
    }
}

template <typename T, typename pT=Complex<hila::arithmetic_type<T>>>
void measure_polyakov_field_complex(const Field<T> &Ut, Field<pT> &pl) {
    Field<T> polyakov = Ut;

    // mult links so that polyakov[X.dir == 0] contains the polyakov loop
    for (int plane = lattice.size(e_t) - 2; plane >= 0; plane--) {

        // safe_access(polyakov) pragma allows the expression below, otherwise
        // hilapp would reject it because X and X+dir can refer to the same
        // site on different "iterations" of the loop.  However, here this
        // is restricted on single dir-plane so it works but we must tell it to hilapp.

#pragma hila safe_access(polyakov)
        onsites(ALL) {
            if (X.coordinate(e_t) == plane) {
                polyakov[X] = Ut[X] * polyakov[X + e_t];
            }
        }
    }

    onsites(ALL) if (X.coordinate(e_t) == 0) {
        pl[X] = trace(polyakov[X]);
    }
}

////////////////////////////////////////////////////////////////////

template <typename group, typename pT = Complex<hila::arithmetic_type<group>>>
void measure_polyakov_profile(GaugeField<group> &U, Direction dz) {

    Direction dx, dy;
    foralldir(td) if (td != dz) {
        dx = td;
        break;
    }
    foralldir(td) if (td != dz && td != dx) {
        dy = td;
        break;
    }

    int size_x = lattice.size(dx);
    int size_y = lattice.size(dy);

    int area = size_x * size_y;

    Field<pT> pl;
    measure_polyakov_field_complex(U[e_t], pl);

    ReductionVector<pT> p(lattice.size(dz));
    p.allreduce(false);
    onsites(ALL) if (X.coordinate(e_t) == 0) {
        p[X.z()] += pl[X];
    }

    for (int z = 0; z < lattice.size(dz); z++) {
        hila::out0 << "PPOLY " << z << ' ' << p[z].real() / area << ' ' << p[z].imag() / area << ' '
                   << p[z].arg() << '\n';
    }
}


///////////////////////////////////////////////////////////////////////////////////

template <typename group, typename wT>
void measure_stuff(const GaugeField<group> &U, const parameters &p, const plaqw_t<wT> &plaqw) {

    static bool first = true;

    if (first) {
        hila::out0 << "Legend:";
        hila::out0 << " plaq  P.real  P.imag\n";

        first = false;
    }

    auto poly = measure_polyakov(U);

    auto plaq = measure_plaq(U, plaqw) / (lattice.volume() * NDIM * (NDIM - 1) / 2);

    hila::out0 << "MEAS " << std::setprecision(14);

    hila::out0 << plaq << ' ' << poly << '\n';
}


///////////////////////////////////////////////////////////////////////////////////
// measurement functions


template <typename T>
Complex<T> phase_to_complex(T arg) {
    return Complex<T>(cos(arg), sin(arg));
}


template <typename T>
Complex<T> proj_to_nrange(Complex<T> inval) {
    while (inval.imag() > (T)NCOLOR / 2) {
        inval.imag() -= (T)NCOLOR;
    }
    while (inval.imag() <= -(T)(NCOLOR + 1) / 2) {
        inval.imag() += (T)NCOLOR;
    }
    return inval;
}


template <typename T>
Complex<T> proj_to_nrange(Complex<T> inval, T tol) {
    while (inval.imag() > (T)NCOLOR / 2 + tol) {
        inval.imag() -= (T)NCOLOR;
    }
    while (inval.imag() <= -((T)NCOLOR / 2 + tol)) {
        inval.imag() += (T)NCOLOR;
    }
    return inval;
}


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


template <typename T, typename sT, typename aT>
void smear_spat_field(Field<Complex<T>> &smS, const VectorField<sT> &shift, aT smear_param, int n_smear,
                 Complex<T> smSmean) {
    int ip = 0;
    Field<Complex<T>> tsmS[2];
    onsites(ALL) {
        if(X.coordinate(e_t) == 0) {
            tsmS[ip][X] = proj_to_nrange(smS[X] - smSmean, (T)0);
        }
    }
    for (int ism = 0; ism < n_smear; ++ism) {
        onsites(ALL) {
            if (X.coordinate(e_t) == 0) {
                tsmS[1 - ip][X] = tsmS[ip][X];
            }
        }
        foralldir(d) if(d < e_t) {
            onsites(ALL) {
                if (X.coordinate(e_t) == 0) {
                    tsmS[1 - ip][X] += smear_param * (tsmS[ip][X + d] - Complex<T>(0, shift[d][X]));

                    tsmS[1 - ip][X] +=
                        smear_param * (tsmS[ip][X - d] + Complex<T>(0, shift[d][X - d]));
                }
            }
        }
        onsites(ALL) {
            if (X.coordinate(e_t) == 0) {
                tsmS[1 - ip][X] = tsmS[1 - ip][X] / ((T)1.0 + (T)(2 * (NDIM - 1)) * smear_param);
            }
        }
        ip = 1 - ip;
    }
    onsites(ALL) {
        if (X.coordinate(e_t) == 0) {
            smS[X] = proj_to_nrange(tsmS[ip][X] + smSmean, (T)0);
        }
    }
}

template <typename T, typename sT>
void measure_profile(const Field<Complex<T>> &smS, Direction d, std::vector<sT> &profile) {
    ReductionVector<sT> p(lattice.size(d));
    p.allreduce(false).delayed(true);
    sT area = (sT)(lattice.volume() / lattice.size(d) / lattice.size(e_t));
    onsites(ALL) {
        if (X.coordinate(e_t) == 0) {
            p[X.coordinate(d)] += smS[X].imag() / area;
        }
    }
    p.reduce();
    profile = p.vector();
}

int z_ind(int z, Direction dz) {
    return (z + lattice.size(dz)) % lattice.size(dz);
}


template <typename T, typename sT>
void measure_interface(const Field<Complex<T>> &smS, Direction dz, sT surface_level, std::vector<sT> &surf,
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

    if (hila::myrank() == 0) {
        surf.resize(area);
    }

    std::vector<Complex<T>> lsmS;
    std::vector<sT> line(lattice.size(dz));
    measure_profile(smS, dz, line);

    int startloc = lattice.size(dz) / 2;
    int ddz = 1;

    sT minp = 1.0e8, maxp = -1.0e8;
    int minloc = 0, maxloc = 0;
    if(hila::myrank() == 0) {
        for (int i = 0; i < line.size(); i++) {
            if (minp > line[i]) {
                minp = line[i];
                minloc = i;
            }
            if (maxp < line[i]) {
                maxp = line[i];
                maxloc = i;
            }
        }
        if (minloc > maxloc) {
            ddz = -1;
        }
    }

    hila::broadcast(ddz);


    if (hila::myrank() == 0) {
        int z = startloc;
        while (line[z_ind(z, dz)] > surface_level &&
               ddz * (startloc - z) < lattice.size(dz) * 0.4) {
            z -= ddz;
        }

        while (line[z_ind(z + ddz, dz)] <= surface_level &&
               ddz * (z - startloc) < lattice.size(dz) * 0.4) {
            z += ddz;
        }

        startloc = z_ind(z, dz);
    }

    hila::broadcast(startloc);


    CoordinateVector yslice;
    foralldir(td) {
        if (td != e_t) {
            yslice[td] = -1;
        } else {
            yslice[td] = 0;
        }
    }
    for (int y = 0; y < size_y; ++y) {
        yslice[dy] = y;
        lsmS = smS.get_slice(yslice);
        if (hila::myrank() == 0) {
            for (int x = 0; x < size_x; ++x) {

                for (int z = 0; z < lattice.size(dz); z++) {
                    line[z] = lsmS[x + size_x * z].imag();
                }

                // if (hila::myrank() == 0) {
                // start search of the surface from the center between min and max
                int z = startloc;

                while (line[z_ind(z, dz)] > surface_level &&
                       ddz * (startloc - z) < lattice.size(dz) * 0.4) {
                    z -= ddz;
                }

                while (line[z_ind(z + ddz, dz)] <= surface_level &&
                       ddz * (z - startloc) < lattice.size(dz) * 0.4) {
                    z += ddz;
                }


                // do linear interpolation
                surf[x + y * size_x] = (sT)z + (sT)ddz * (surface_level - line[z_ind(z, dz)]) /
                                                   (line[z_ind(z + ddz, dz)] - line[z_ind(z, dz)]);
            }
        }
    }
}


template <typename T, typename sT>
void measure_interface_spectrum(Field<Complex<T>> &smS, Direction dir_z, sT surface_level,
                                int bds_shift, int nsm, int64_t idump, parameters &p,
                                bool first_pow) {
    std::vector<ftype> surf;
    int size_x, size_y;
    measure_interface(smS, dir_z, surface_level, surf, size_x, size_y);
    if (hila::myrank() == 0) {
        if (false) {
            for (int x = 0; x < size_x; ++x) {
                for (int y = 0; y < size_y; ++y) {
                    hila::out0 << "SURF" << nsm << ' ' << x << ' ' << y << ' '
                               << surf[x + y * size_x] << '\n';
                }
            }
        }
        constexpr int pow_size = 200;
        std::vector<double> npow(pow_size, 0);
        std::vector<int> hits(pow_size, 0);
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
                ifile.read((char *)buffer, 2 * sizeof(double));
                prof_os.write((char *)buffer, 2 * sizeof(double));

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
                double tval = p.beta;
                prof_os.write((char *)&(tval), sizeof(double));
                tval = p.smear_coeff;
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


template <typename T, typename sT>
void measure_interface_ft(const Field<Complex<T>> &smS, Direction dz, int bds_shift, std::vector<sT> &surf,
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

    static std::vector<Complex<sT>> ft_basis;
    static bool first = true;

    if (first) {
        first = false;
        ft_basis.resize(lattice.size(dz));
        sT tsign = (sT)1.0;
        if (bds_shift < 0) {
            tsign = -tsign;
        }
        for (int iz = 0; iz < lattice.size(dz); ++iz) {
            sT ftarg = M_PI * (sT)iz / (sT)lattice.size(dz);
            ft_basis[iz] = tsign * phase_to_complex(ftarg);
        }
    }

    if (hila::myrank() == 0) {
        surf.resize(area);
    }

    std::vector<Complex<T>> lS;

    CoordinateVector yslice;
    foralldir(td) {
        if(td != e_t) {
            yslice[td] = -1;
        } else {
            yslice[td] = 0;
        }
    }
    Complex<sT> ftn;
    for (int y = 0; y < size_y; ++y) {
        yslice[dy] = y;
        lS = smS.get_slice(yslice);
        if (hila::myrank() == 0) {
            for (int x = 0; x < size_x; ++x) {
                ftn = 0;
                for (int z = 0; z < lattice.size(dz); z++) {
                    ftn += (2.0 * (sT)lS[x + size_x * z].imag() + (sT)bds_shift) * ft_basis[z];
                }

                surf[x + y * size_x] =
                    (sT)arg(ftn) * (sT)lattice.size(dz) / M_PI + (sT)(lattice.size(dz) + 1) / 2;
            }
        }
    }
}


template <typename T>
void measure_interface_spectrum_ft(Field<Complex<T>> &smS, Direction dir_z, int bds_shift, int nsm,
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
        std::vector<double> npow(pow_size, 0);
        std::vector<int> hits(pow_size, 0);
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
                ifile.read((char *)buffer, 2 * sizeof(double));
                prof_os.write((char *)buffer, 2 * sizeof(double));

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
                double tval = p.beta;
                prof_os.write((char *)&(tval), sizeof(double));
                tval = p.smear_coeff;
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

    parameters p;

    hila::out0 << "SU(" << mygroup::size() << ") heat bath + overrelax update\n";

    hila::input par("parameters");

    CoordinateVector lsize;
    lsize = par.get("lattice size"); // reads NDIM numbers

    p.beta = par.get("beta");
    // trajectory length in steps
    p.n_overrelax = par.get("overrelax steps");
    p.n_heatbath = par.get("heatbath steps");
    p.n_trajectories = par.get("trajectories");
    p.n_thermal = par.get("thermalization");

    // random seed = 0 -> get seed from time
    uint64_t seed = par.get("random seed");
    // save config and checkpoint
    p.n_save = par.get("traj/saved");
    // measure surface properties and print "profile"
    p.config_file = par.get("config name");


    // boundary shift
    int bds_dir = par.get("bds direction");
    int bds_shift = par.get("bds shift");

    if (par.get_item("updates/profile meas", {"off", "%i"}) == 1) {
        p.n_profile = par.get();
    } else {
        p.n_profile = 0;
    }

    if (par.get_item("updates/surf spec meas", {"off", "%i"}) == 1) {
        p.n_surf_spec = par.get();
    } else {
        p.n_surf_spec = 0;
    }

    if (p.n_surf_spec) {
        p.n_smear = par.get("smearing steps");
        p.smear_coeff = par.get("smear coefficient");
        p.z_smear = par.get("z smearing steps");
        p.n_surface = par.get("traj/surface");
        p.n_dump_polyakov = par.get("traj/polyakov dump");

        if (p.n_smear.size() != p.z_smear.size()) {
            hila::out0 << "Error in input file: number of values in 'smearing steps' != 'z "
                          "smearing steps'\n";
            hila::terminate(0);
        }

    } else {
        p.n_dump_polyakov = 0;
    }

    par.close(); // file is closed also when par goes out of scope

    // setting up the lattice is convenient to do after reading
    // the parameter
    lattice.setup(lsize);

    // We need random number here
    if (!hila::is_rng_seeded())
        hila::seed_random(seed);


    // use negative trajectory number for thermalization
    int start_traj = -p.n_thermal;

    // define direction in which boundary shift should be implemented
    Direction bdt_dir = Direction(bds_dir);


    plaqw_t<Complex<ftype>> plaqw;
    VectorField<ftype> shift;
    ftype ltw_ang = 2.0 * M_PI * (double)bds_shift / (double)NCOLOR;
    foralldir(d1) {
        onsites(ALL) {
            plaqw[d1][d1][X] = 0;
            
            if (d1 == bdt_dir && X.coordinate(d1) == lattice.size(d1) -1) {
                shift[d1][X] = -ltw_ang;
            } else {
                shift[d1][X] = 0;
            }
        }
        foralldir(d2) if (d1 < d2) {
            if (d2 == e_t && d1 == bdt_dir) {
                onsites(ALL) {
                    if (X.coordinate(d2) == 0 && X.coordinate(d1) == lattice.size(d1) - 1) {
                        plaqw[d1][d2][X].polar((ftype)1, -ltw_ang);
                        plaqw[d2][d1][X].polar((ftype)1, ltw_ang);
                    } else {
                        plaqw[d1][d2][X] = (ftype)1;
                        plaqw[d2][d1][X] = (ftype)1;
                    }
                }
            } else {
                onsites(ALL) {
                    plaqw[d1][d2][X] = (ftype)1;
                    plaqw[d2][d1][X] = (ftype)1;
                }
            }
        }
    }

    // define boundary twist matrix from provided shift value


    // Alloc gauge field
    GaugeField<mygroup> U;
    U = 1;

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");

    if(!restore_checkpoint(U, p.config_file, p.n_trajectories, start_traj)) {
        onsites(ALL) {
            U[e_t][X] = 1;
            if (X.coordinate(e_t) == 0 && X.coordinate(bdt_dir) >= lattice.size(bdt_dir) / 2) {
                U[e_t][X] *= Complex<ftype>(cos(ltw_ang), sin(ltw_ang));
            }
        }
    }


    double p_prev = (double)NCOLOR;
    double p_now = measure_polyakov(U).real();

    bool first_pow = true;
    bool run = true;
    for (int trajectory = start_traj; run && trajectory < p.n_trajectories; trajectory++) {

        double ttime = hila::gettime();

        update_timer.start();

        do_trajectory(U, p, plaqw);

        // put sync here in order to get approx gpu timing
        hila::synchronize_threads();
        update_timer.stop();

        measure_timer.start();

        hila::out0 << "Measure_start " << trajectory << '\n';

        measure_stuff(U, p, plaqw);

        // trajectory is negative during thermalization
        if (trajectory >= 0) {
            if (p.n_surf_spec && (trajectory + 1) % p.n_surf_spec == 0) {
                //measure_polyakov_surface(U, p, trajectory);
                int64_t idump = (trajectory + 1) / p.n_surf_spec;

                Field<Complex<double>> smS, tsmS, tsmS2;

                measure_polyakov_field_complex(U[e_t], smS);
                onsites(ALL) if (X.coordinate(e_t) == 0) {
                    Complex<double> tsmSval = smS[X];
                    if(abs(tsmSval) > 0) {
                        smS[X] = Complex<double>(log(abs(tsmSval)), arg(tsmSval));
                    } else {
                        smS[X] = 0;
                    }
                }

                ftype surface_level = (ftype)ltw_ang / 2;

                hila::out0 << string_format("SURFLEV %0.5f\n", surface_level);

                int n_smear = 0;

                Complex<double> cshift = Complex<double>(0, ltw_ang / 2);
                for (int ism = 0; ism < p.n_smear.size(); ++ism) {
                    int smear = p.n_smear[ism];
                    smear_spat_field(smS, shift, p.smear_coeff, smear - n_smear, cshift);
                    n_smear = smear;
                    tsmS = smS;
                    if (p.z_smear[ism] > 0) {
                        for (int j = 0; j < p.z_smear[ism]; j++) {
                            onsites(ALL) if (X.coordinate(e_t) == 0) {
                                tsmS2[X] =
                                    tsmS[X] +
                                    p.smear_coeff *
                                        (tsmS[X + bdt_dir] - Complex<double>(0, shift[bdt_dir][X]) +
                                         tsmS[X - bdt_dir] + Complex<double>(0, shift[bdt_dir][X - bdt_dir]));
                            }
                            onsites(ALL) if (X.coordinate(e_t) == 0) {
                                tsmS[X] = tsmS2[X] / (1 + 2 * p.smear_coeff);
                            }
                        }
                    }
                    if(n_smear > 0) {
                        measure_interface_spectrum(tsmS, bdt_dir, surface_level, bds_shift, n_smear,
                                                   idump, p, first_pow);
                    }
                    measure_interface_spectrum_ft(smS, bdt_dir, bds_shift, n_smear, idump, p,
                                                  first_pow);

                    if (p.n_dump_polyakov && (trajectory + 1) % p.n_dump_polyakov == 0) {
                        Field<float> pl;
                        onsites(ALL) {
                            if (X.coordinate(e_t) == 0) {
                                pl[X] = (float)smS[X].imag();
                            } else {
                                pl[X] = 0;
                            }
                        }

                        int icdump = (trajectory + 1) / p.n_dump_polyakov;
                        std::string dump_file =
                            string_format("poly_dump_%04d_nsm%04d", icdump, n_smear);
                        pl.config_slice_write(dump_file, {-1, -1, -1, 0});
                    }
                }
                first_pow = false;

            }

        }

        if (p.n_profile && (trajectory + 1) % p.n_profile == 0) {
            measure_polyakov_profile(U, bdt_dir);
        }

        hila::out0 << "Measure_end " << trajectory << " time " << hila::gettime() << std::endl;

        measure_timer.stop();
        

        run = !hila::time_to_finish();
        if (!run || (p.n_save > 0 && (trajectory + 1) % p.n_save == 0)) {
            checkpoint(U, p.config_file, p.n_trajectories, trajectory);
        }
    }

    hila::finishrun();
}
