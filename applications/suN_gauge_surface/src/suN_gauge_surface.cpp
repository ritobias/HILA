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

enum class poly_limit { OFF, RANGE, PARABOLIC };

// define a struct to hold the input parameters: this
// makes it simpler to pass the values around
struct parameters {
    double beta;
    double deltab;
    int n_overrelax;
    int n_heatbath;
    int n_trajectories;
    int n_thermal;
    int n_save;
    int n_profile;
    int n_surf_spec;
    std::string config_file;
    double time_offset;
    poly_limit polyakov_pot;
    double poly_min, poly_max, poly_m2;
    std::vector<int> n_smear;
    double smear_coeff;
    std::vector<int> z_smear;
    int n_surface;
    int n_dump_polyakov;
};

template <typename T>
void staplesum_db(const GaugeField<T> &U, Field<T> &staples, Direction d1, Parity par,
                  double deltab) {

    Field<T> lower;

    // zero staples just in case
    staples[par] = 0;

    foralldir(d2) if (d2 != d1) {

        // anticipate that these are needed
        // not really necessary, but may be faster
        U[d2].start_gather(d1, ALL);
        U[d1].start_gather(d2, par);

        // calculate first lower 'U' of the staple sum
        // do it on opp parity
        onsites(opp_parity(par)) {
            double m;
            if (2 * X.z() < lattice.size(e_z))
                m = 1.0 - deltab;
            else
                m = 1.0 + deltab;

            lower[X] = m * U[d2][X].dagger() * U[d1][X] * U[d2][X + d1];
        }

        // calculate then the upper 'n', and add the lower
        // lower could also be added on a separate loop
        onsites(par) {
            double m;
            if (2 * X.z() < lattice.size(e_z))
                m = 1.0 - deltab;
            else
                m = 1.0 + deltab;

            staples[X] += m * U[d2][X] * U[d1][X + d2] * U[d2][X + d1].dagger() + lower[X - d2];
        }
    }
}

template <typename group>
double measure_plaq_db(const GaugeField<group> &U, double db = 0.0) {
    Reduction<double> plaq;
    plaq.allreduce(false);

    foralldir(dir1) foralldir(dir2) if (dir1 < dir2) {
        if (db == 0.0) {
            onsites(ALL) {
                plaq += 1.0 - real(trace(U[dir1][X] * U[dir2][X + dir1] *
                                         U[dir1][X + dir2].dagger() * U[dir2][X].dagger())) /
                                  group::size();
            }
        } else {
            onsites(ALL) {
                double c;
                if (2 * X.z() < lattice.size(e_z))
                    c = 1 - db;
                else
                    c = 1 + db;

                plaq += c * (1.0 - real(trace(U[dir1][X] * U[dir2][X + dir1] *
                                              U[dir1][X + dir2].dagger() * U[dir2][X].dagger())) /
                                       group::size());
            }
        }
    }
    return plaq.value();
}

template <typename group>
double measure_action(const GaugeField<group> &U, const VectorField<Algebra<group>> &E,
                      const parameters &p) {

    auto plaq = measure_plaq_bp(U, p.deltab);

    return p.beta * plaq;
}

/////////////////////////////////////////////////////////////////////////////
/// Measure polyakov "field"
///

template <typename T>
void measure_polyakov_field(const Field<T> &Ut, Field<float> &pl) {
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
        pl[X] = real(trace(polyakov[X]));
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

void smear_polyakov_field(Field<float> &pl, int nsmear, float smear_coeff) {

    if (nsmear > 0) {
        Field<float> pl2 = 0;

        for (int i = 0; i < nsmear; i++) {
            onsites(ALL) if (X.coordinate(e_t) == 0) {
                pl2[X] = pl[X] + smear_coeff * (pl[X + e_x] + pl[X - e_x] + pl[X + e_y] +
                                                pl[X - e_y] + pl[X + e_z] + pl[X - e_z]);
            }

            onsites(ALL) if (X.coordinate(e_t) == 0) {
                pl[X] = pl2[X] / (1 + 6 * smear_coeff);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////

std::vector<float> measure_polyakov_profile(Field<float> &pl, std::vector<float> &pro1) {
    ReductionVector<float> p(lattice.size(e_z)), p1(lattice.size(e_z));
    p.allreduce(false);
    p1.allreduce(false);
    onsites(ALL) if (X.coordinate(e_t) == 0) {
        p[X.z()] += pl[X];
        if (X.x() == 0 && X.y() == 0)
            p1[X.z()] += pl[X];
    }
    pro1 = p1.vector();
    return p.vector();
}


///////////////////////////////////////////////////////////////////////////////////

template <typename group>
void measure_plaq_profile(GaugeField<group> &U) {
    ReductionVector<float> p(lattice.size(e_z));
    p.allreduce(false);
    p.delayed(true);
    foralldir(dir1) foralldir(dir2) if (dir2 > dir1) {
        onsites(ALL) {
            p[X.z()] += 1.0 - real(trace(U[dir1][X] * U[dir2][X + dir1] *
                                         U[dir1][X + dir2].dagger() * U[dir2][X].dagger())) /
                                  group::size();
        }
    }
    p.reduce();

    for (int z = 0; z < lattice.size(e_z); z++) {
        hila::out0 << "PPLAQ " << z << ' '
                   << p[z] / (NDIM * (NDIM - 1) / 2 * lattice.volume() / lattice.size(e_z)) << '\n';
    }
}


///////////////////////////////////////////////////////////////////////////////////

template <typename group>
void measure_stuff(const GaugeField<group> &U, const parameters &p) {

    static bool first = true;

    if (first) {
        hila::out0 << "Legend:";
        if (p.polyakov_pot == poly_limit::PARABOLIC)
            hila::out0 << " -V(polyakov)";
        hila::out0 << " plaq  P.real  P.imag\n";

        first = false;
    }

    auto poly = measure_polyakov(U);

    auto plaq = measure_plaq_db(U) / (lattice.volume() * NDIM * (NDIM - 1) / 2);

    hila::out0 << "MEAS " << std::setprecision(14);

    // write the -(polyakov potential) first, this is used as a weight factor in aa
    if (p.polyakov_pot == poly_limit::PARABOLIC) {
        hila::out0 << -polyakov_potential(p, poly.real()) << ' ';
    }

    hila::out0 << plaq << ' ' << poly << '\n';
}

///////////////////////////////////////////////////////////////////////////////////
template <typename sT>
void spectraldensity_surface(std::vector<sT> &surf, std::vector<double> &npow,
                             std::vector<int> &hits) {

    // do fft for the surface
    static bool first = true;
    static Complex<double> *buf;
    static fftw_plan fftwplan;

    int area = lattice.size(e_x) * lattice.size(e_y);

    if (first) {
        first = false;

        buf = (Complex<double> *)fftw_malloc(sizeof(Complex<double>) * area);

        // note: we had x as the "fast" dimension, but fftw wants the 2nd dim to be
        // the "fast" one. thus, first y, then x.
        fftwplan = fftw_plan_dft_2d(lattice.size(e_y), lattice.size(e_x), (fftw_complex *)buf,
                                    (fftw_complex *)buf, FFTW_FORWARD, FFTW_ESTIMATE);
    }

    for (int i = 0; i < area; i++) {
        buf[i] = (double)surf[i];
    }

    fftw_execute(fftwplan);

    int pow_size = npow.size();

    for (int i = 0; i < area; i++) {
        int x = i % lattice.size(e_x);
        int y = i / lattice.size(e_x);
        x = (x <= lattice.size(e_x) / 2) ? x : (lattice.size(e_x) - x);
        y = (y <= lattice.size(e_y) / 2) ? y : (lattice.size(e_y) - y);

        int k = x * x + y * y;
        if (k < pow_size) {
            npow[k] += buf[i].squarenorm() / (area * area);
            hits[k]++;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////
// helper function to get valid z-coordinate index

int z_ind(int z) {
    return (z + lattice.size(e_z)) % lattice.size(e_z);
}

///////////////////////////////////////////////////////////////////////////////////

template <typename group, typename pT = Complex<hila::arithmetic_type<group>>>
void measure_polyakov_profile(GaugeField<group> &U) {
    Field<pT> pl;
    measure_polyakov_field_complex(U[e_t], pl);

    ReductionVector<pT> p(lattice.size(e_z));
    p.allreduce(false);
    onsites(ALL) if (X.coordinate(e_t) == 0) {
        p[X.z()] += pl[X];
    }

    for (int z = 0; z < lattice.size(e_z); z++) {
        hila::out0 << "PPOLY " << z << ' ' << p[z].real() / (lattice.size(e_x) * lattice.size(e_y)) << ' '
                   << p[z].imag() / (lattice.size(e_x) * lattice.size(e_y)) << '\n';
    }
}


template <typename group>
void measure_polyakov_surface(GaugeField<group> &U, const parameters &p, int traj) {

    Field<float> pl;


    if (0) {
        // this section does local sums of poly lines
        Field<group> staples, Ut;

        Ut = U[e_t];

        for (int s = 0; s < 20; s++) {
            staplesum(U, staples, e_t);
            U[e_t][ALL] = (U[e_t][X] + 0.5 * staples[X]) / (1 + 6 * 0.5);
        }

        measure_polyakov_field(U[e_t], pl);

        U[e_t] = Ut;

    } else {
        // here standard non-link integrated field
        measure_polyakov_field(U[e_t], pl);
    }

    hila::out0 << std::setprecision(7);

    std::vector<float> profile, prof1;

    int prev_smear = 0;
    for (int sl = 0; sl < p.n_smear.size(); sl++) {

        int smear = p.n_smear.at(sl);
        smear_polyakov_field(pl, smear - prev_smear, p.smear_coeff);
        prev_smear = smear;

        Field<float> plz = pl;
        if (p.z_smear.at(sl) > 0) {
            Field<float> pl2;
            for (int j = 0; j < p.z_smear.at(sl); j++) {
                onsites(ALL) if (X.coordinate(e_t) == 0) {
                    pl2[X] = plz[X] + p.smear_coeff * (plz[X + e_z] + plz[X - e_z]);
                }
                onsites(ALL) if (X.coordinate(e_t) == 0) {
                    plz[X] = pl2[X] / (1 + 2 * p.smear_coeff);
                }
            }
        }

        profile = measure_polyakov_profile(plz, prof1);

        double m = 1.0 / (lattice.size(e_x) * lattice.size(e_y));
        for (int i = 0; i < profile.size(); i++) {
            profile[i] *= m;
            hila::out0 << "PRO" << sl << ' ' << i << ' ' << profile[i] << ' ' << prof1[i] << '\n';
        }

        float min = 1e8, max = 0;
        int minloc, maxloc;
        for (int i = 0; i < profile.size(); i++) {
            if (min > profile[i]) {
                min = profile[i];
                minloc = i;
            }
            if (max < profile[i]) {
                max = profile[i];
                maxloc = i;
            }
        }

        // find the surface between minloc and maxloc
        float surface_level = max * 0.5; // assume min is really 0
        int area = lattice.size(e_x) * lattice.size(e_y);

        hila::out0 << "Surface_level" << sl << ' ' << surface_level << '\n';

        int startloc, startloc2;
        if (maxloc > minloc)
            startloc = (maxloc + minloc) / 2;
        else
            startloc = ((maxloc + minloc + lattice.size(e_z)) / 2) % lattice.size(e_z);

        // starting positio for the other surface
        startloc2 = z_ind(startloc + lattice.size(e_z) / 2);

        std::vector<float> surf1, surf2;
        if (hila::myrank() == 0) {
            surf1.resize(area);
            surf2.resize(area);
        }

        hila::out0 << std::setprecision(6);

        std::vector<float> poly;
        std::vector<float> line(lattice.size(e_z));

        // get full xyz-volume t=0 slice to main node
        // poly = plz.get_slice({-1, -1, -1, 0});

        for (int y = 0; y < lattice.size(e_y); y++) {
            // get now full xz-plane polyakov line to main node
            // reduces MPI calls compared with doing line-by-line
            poly = plz.get_slice({-1, y, -1, 0});
            if (hila::myrank() == 0) {
                for (int x = 0; x < lattice.size(e_x); x++) {
                    // line = plz.get_slice({x, y, -1, 0});

                    // copy ploop data to line - x runs fastest
                    for (int z = 0; z < lattice.size(e_z); z++) {
                        line[z] = poly[x + lattice.size(e_x) * (z)];
                    }

                    // if (hila::myrank() == 0) {
                    // start search of the surface from the center between min and max
                    int z = startloc;

                    while (line[z_ind(z)] > surface_level && startloc - z < lattice.size(e_z) * 0.4)
                        z--;

                    while (line[z_ind(z + 1)] <= surface_level &&
                           z - startloc < lattice.size(e_z) * 0.4)
                        z++;


                    // do linear interpolation
                    // surf[x + y * lattice.size(e_x)] = z;
                    surf1[x + y * lattice.size(e_x)] =
                        z +
                        (surface_level - line[z_ind(z)]) / (line[z_ind(z + 1)] - line[z_ind(z)]);

                    if (p.n_surface > 0 && (traj + 1) % p.n_surface == 0) {
                        hila::out0 << "SURF" << sl << ' ' << x << ' ' << y << ' '
                                   << surf1[x + y * lattice.size(e_x)] << '\n';
                    }

                    // and locate the other surface - start from Lz/2 offset

                    z = startloc2;

                    while (line[z_ind(z)] <= surface_level &&
                           startloc2 - z < lattice.size(e_z) * 0.4)
                        z--;

                    while (line[z_ind(z + 1)] > surface_level &&
                           z - startloc2 < lattice.size(e_z) * 0.4)
                        z++;

                    // do linear interpolation
                    // surf[x + y * lattice.size(e_x)] = z;
                    surf2[x + y * lattice.size(e_x)] =
                        z +
                        (surface_level - line[z_ind(z)]) / (line[z_ind(z + 1)] - line[z_ind(z)]);
                }
            }
        }

        if (hila::myrank() == 0) {
            constexpr int pow_size = 200;
            std::vector<double> npow(pow_size, 0);
            std::vector<int> hits(pow_size, 0);

            spectraldensity_surface(surf1, npow, hits);
            spectraldensity_surface(surf2, npow, hits);

            for (int i = 0; i < pow_size; i++) {
                if (hits[i] > 0)
                    hila::out0 << "POW" << sl << ' ' << i << ' ' << npow[i] / hits[i] << ' '
                               << hits[i] << '\n';
            }
        }
    }
}

template <typename sT, typename aT = hila::arithmetic_type<sT>>
void measure_profile(const Field<sT> &smS, Direction d, int area, std::vector<sT> &profile) {
    ReductionVector<sT> p(lattice.size(d));
    p.allreduce(false).delayed(true);
    aT iarea = (aT)1.0 / (aT)area;
    onsites(ALL) {
        p[X.coordinate(d)] += smS[X];
    }
    p.reduce();
    profile = p.vector();
    for (int z = 0; z < lattice.size(d); z++) {
        profile[z] *= iarea;
    }
}

int z_ind(int z, Direction dz) {
    return (z + lattice.size(dz)) % lattice.size(dz);
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

template <typename sT, typename aT>
void measure_interface(const Field<sT> &smS, Direction dz, std::vector<aT> &surf1,
                       std::vector<aT> &surf2, out_only int &size_x, out_only int &size_y) {

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
        surf1.resize(area);
        surf2.resize(area);
    }

    std::vector<aT> profile;
    measure_profile(smS, dz, area, profile);

    aT minp = 1.0e8, maxp = -1.0e8;
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


    aT surface_level = maxp * 0.5;


    int startloc, startloc2;
    if (maxloc > minloc)
        startloc = (maxloc + minloc) / 2;
    else
        startloc = ((maxloc + minloc + lattice.size(dz)) / 2) % lattice.size(dz);

    // starting positio for the other surface
    startloc2 = z_ind(startloc + lattice.size(dz) / 2);

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

                // start search of the surface from the center between min and max
                int z = startloc;

                while (line[z_ind(z, dz)] > surface_level && startloc - z < lattice.size(dz) * 0.4)
                    z--;

                while (line[z_ind(z + 1, dz)] <= surface_level &&
                       z - startloc < lattice.size(dz) * 0.4)
                    z++;

                // do linear interpolation
                surf1[x + y * lattice.size(dx)] =
                    z + (surface_level - line[z_ind(z, dz)]) / (line[z_ind(z + 1, dz)] - line[z_ind(z, dz)]);


                // and locate the other surface - start from Lz/2 offset
                z = startloc2;

                while (line[z_ind(z, dz)] <= surface_level && startloc2 - z < lattice.size(dz) * 0.4)
                    z--;

                while (line[z_ind(z + 1, dz)] > surface_level &&
                       z - startloc2 < lattice.size(dz) * 0.4)
                    z++;

                // do linear interpolation
                surf2[x + y * lattice.size(dx)] =
                    z + (surface_level - line[z_ind(z, dz)]) / (line[z_ind(z + 1, dz)] - line[z_ind(z, dz)]);
                
            }
        }
    }
}


template <typename sT>
void measure_interface_spectrum(Field<sT> &smS, Direction dir_z, int nsm, int64_t idump,
                                parameters &p, bool first_pow) {
    std::vector<sT> surf1, surf2;
    int size_x, size_y;
    measure_interface(smS, dir_z, surf1, surf2, size_x, size_y);
    if (hila::myrank() == 0) {
        if (false) {
            for (int x = 0; x < size_x; ++x) {
                for (int y = 0; y < size_y; ++y) {
                    hila::out0 << "SURF" << nsm << ' ' << x << ' ' << y << ' '
                               << surf1[x + y * size_x] << '\n';
                }
            }
        }
        constexpr int pow_size = 200;
        std::vector<double> npow(pow_size, 0);
        std::vector<int> hits(pow_size, 0);
        spectraldensity_surface(surf1, size_x, size_y, npow, hits);
        spectraldensity_surface(surf2, size_x, size_y, npow, hits);

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


////////////////////////////////////////////////////////////////

template <typename group>
void update_parity_dir(GaugeField<group> &U, const parameters &p, Parity par, Direction d,
                       bool relax) {

    static hila::timer hb_timer("Heatbath");
    static hila::timer or_timer("Overrelax");
    static hila::timer staples_timer("Staplesum");

    Field<group> staples;

    staples_timer.start();
    if (p.deltab != 0)
        staplesum_db(U, staples, d, par, p.deltab);
    else
        staplesum(U, staples, d, par);
    staples_timer.stop();

    if (relax) {

        or_timer.start();
        onsites(par) {
#ifdef SUN_OVERRELAX_dFJ
            suN_overrelax_dFJ(U[d][X], staples[X], p.beta);
#else
            suN_overrelax(U[d][X], staples[X]);
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


template <typename group>
void update(GaugeField<group> &U, const parameters &p) {

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
        update_parity_dir(U, p, Parity(tpar), Direction(tdir), relax);
    }
}


////////////////////////////////////////////////////////////////

template <typename group>
void do_trajectory(GaugeField<group> &U, const parameters &p) {

    for (int n = 0; n < p.n_heatbath + p.n_overrelax; n++) {
        update(U, p);
    }
    U.reunitarize_gauge();
}

////////////////////////////////////////////////////////////////

double polyakov_potential(const parameters &p, const double poly) {

    return p.poly_m2 * (sqr(poly - p.poly_min));
}


////////////////////////////////////////////////////////////////

bool accept_polyakov(const parameters &p, const double p_old, const double p_new) {

    if (p.polyakov_pot == poly_limit::PARABOLIC) {
        double dpot = polyakov_potential(p, p_new) - polyakov_potential(p, p_old);

        bool accept = hila::broadcast(hila::random() < exp(-dpot));

        return accept;
    } else {
        if (p_new >= p.poly_min && p_new <= p.poly_max)
            return true;
        if (p_new > p.poly_max && p_new < p_old)
            return true;
        if (p_new < p.poly_min && p_new > p_old)
            return true;
        return false;
    }
}

////////////////////////////////////////////////////////////////

template <typename group>
double update_once_with_range(GaugeField<group> &U, const parameters &p, double &poly) {

    Field<group> Ut_old;

    Ut_old = U[e_t];

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

    double acc = 0;
    int nacc = 0;
    for (int i = 0; i < 2 * NDIM; ++i) {
        bool relax = false;
        int tdp = rnarr[i];
        if (tdp >= 2 * NDIM) {
            tdp -= 2 * NDIM;
            relax = true;
        }
        int tdir = tdp / 2;
        int tpar = 1 + (tdp % 2);

        Parity par = Parity(tpar);
        // perform the selected updates:
        update_parity_dir(U, p, par, Direction(tdir), relax);
        if(tdir == NDIM - 1) {
            double p_now = measure_polyakov(U).real();

            bool acc_update = accept_polyakov(p, poly, p_now);
            ++nacc;
            if (acc_update) {
                poly = p_now;
                acc += 1.0;
                Ut_old[par] = U[e_t][X];
            } else {
                U[e_t][par] = Ut_old[X]; // restore rejected
            }
        }
    }

    if(nacc > 0) {
        acc /= nacc;
    }

    return acc;
}

////////////////////////////////////////////////////////////////

template <typename group>
double trajectory_with_range(GaugeField<group> &U, const parameters &p, double &poly) {

    double acc = 0;
    double p_now = measure_polyakov(U).real();

    for (int n = 0; n < p.n_heatbath + p.n_overrelax; n++) {

        // OR updates
        acc += update_once_with_range(U, p, p_now);


        U.reunitarize_gauge();
    }
    return acc / (p.n_heatbath + p.n_overrelax);

}

/////////////////////////////////////////////////////////////////


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
    // deltab sets system to different beta on different sides, by beta*(1 +- deltab)
    // use for initial config generation only
    p.deltab = par.get("delta beta fraction");
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

    // if polyakov range is off, do nothing with
    int p_item = par.get_item("polyakov potential", {"off", "min", "range"});
    if (p_item == 0) {
        p.polyakov_pot = poly_limit::OFF;
    } else if (p_item == 1) {
        p.polyakov_pot = poly_limit::PARABOLIC;
        p.poly_min = par.get();
        p.poly_m2 = par.get("mass2");
    } else {
        p.polyakov_pot = poly_limit::RANGE;
        Vector<2, double> r;
        r = par.get();
        p.poly_min = r[0];
        p.poly_max = r[1];
    }

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

    // Alloc gauge field
    GaugeField<mygroup> U;

    U = 1;

    // use negative trajectory for thermal
    int start_traj = -p.n_thermal;

    hila::timer update_timer("Updates");
    hila::timer measure_timer("Measurements");

    restore_checkpoint(U, p.config_file, p.n_trajectories, start_traj);

    // We need random number here
    if (!hila::is_rng_seeded())
        hila::seed_random(seed);

    double p_now = measure_polyakov(U).real();

    bool first_pow = true;
    bool run = true;
    for (int trajectory = start_traj; run && trajectory < p.n_trajectories; trajectory++) {

        double ttime = hila::gettime();

        update_timer.start();

        double acc = 0;
        if (p.polyakov_pot != poly_limit::OFF) {
            acc += trajectory_with_range(U, p, p_now);
        } else {
            do_trajectory(U, p);
        }

        // put sync here in order to get approx gpu timing
        hila::synchronize_threads();
        update_timer.stop();

        // trajectory is negative during thermalization
        if (trajectory >= 0) {
            measure_timer.start();

            hila::out0 << "Measure_start " << trajectory << '\n';

            measure_stuff(U, p);

            if (p.n_surf_spec && (trajectory + 1) % p.n_surf_spec == 0) {
                //measure_polyakov_surface(U, p, trajectory);
                int64_t idump = (trajectory + 1) / p.n_surf_spec;

                Field<float> smS, tsmS, tsmS2;

                measure_polyakov_field(U[e_t], smS);

                int n_smear = 0;

                for (int ism = 0; ism < p.n_smear.size(); ++ism) {
                    int smear = p.n_smear[ism];
                    smear_polyakov_field(smS, smear - n_smear, p.smear_coeff);
                    n_smear = smear;
                    tsmS = smS;
                    if (p.z_smear[ism] > 0) {
                        for (int j = 0; j < p.z_smear[ism]; j++) {
                            onsites(ALL) if (X.coordinate(e_t) == 0) {
                                tsmS2[X] = tsmS[X] + p.smear_coeff * (tsmS[X + e_z] + tsmS[X - e_z]);
                            }
                            onsites(ALL) if (X.coordinate(e_t) == 0) {
                                tsmS[X] = tsmS2[X] / (1 + 2 * p.smear_coeff);
                            }
                        }
                    }
                    measure_interface_spectrum(tsmS, e_z, n_smear, idump, p, first_pow);
                }
                first_pow = false;
            }

            if (p.n_profile && (trajectory + 1) % p.n_profile == 0) {
                measure_polyakov_profile(U);
                //measure_plaq_profile(U);
            }

            if (p.n_dump_polyakov && (trajectory + 1) % p.n_dump_polyakov == 0) {
                Field<float> pl;
                std::ofstream poly;
                if (hila::myrank() == 0) {
                    poly.open("polyakov", std::ios::out | std::ios::app);
                }
                measure_polyakov_field(U[e_t], pl);
                pl.write_slice(poly, {-1, -1, -1, 0});
            }

            if (p.polyakov_pot != poly_limit::OFF) {
                hila::out0 << "ACCP " << acc << '\n';
            }

            hila::out0 << "Measure_end " << trajectory << " time " << hila::gettime() << std::endl;

            measure_timer.stop();
        }

        run = !hila::time_to_finish();
        if (!run || (p.n_save > 0 && (trajectory + 1) % p.n_save == 0)) {
            checkpoint(U, p.config_file, p.n_trajectories, trajectory);
        }
    }

    hila::finishrun();
}
