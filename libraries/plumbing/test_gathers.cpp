//////////////////////////////////////////////////////////////////////////////
/// Test the standard gather here
//////////////////////////////////////////////////////////////////////////////

#include "hila.h"

void gather_test() {

    int64_t s = 0;
    onsites(ALL) {
        s += 1;
    }

    if (s != lattice.volume()) {
        hila::out0 << " Reduction test error!  Sum " << s << " should be "
                << lattice.volume() << '\n';
        hila::terminate(1);
    }



#ifndef GENGATHER
    foralldir(d) {
        CoordinateVector dif1 = 0, dif2 = 0;
        Field<CoordinateVector> f1, f2;

        onsites(ALL) {
            f1[X] = X.coordinates();
            f2[X] = (X.coordinates() + d).mod(lattice.size());
        }

        onsites(ALL) {
            dif1 += abs(f1[X + d] - f2[X]);
            dif2 += abs(f1[X] - f2[X - d]);
        }

        if (dif1.squarenorm() != 0) {
            hila::out0 << " Std up-gather test error! Node " << hila::myrank()
                    << " direction " << (unsigned)d << " dif1 " << dif1 << '\n';
            hila::terminate(1);
        }

        if (dif2.squarenorm() != 0) {
            hila::out0 << " Std down-gather test error! Node " << hila::myrank()
                    << " direction " << (unsigned)d << " dif2 " << dif2 << '\n';
            hila::terminate(1);
        }
#if 0 && defined(SPECIAL_BOUNDARY_CONDITIONS)
        // test antiperiodic b.c. to one direction
        if (next_direction(d) == NDIM) {
            f2.set_boundary_condition(d, hila::bc::ANTIPERIODIC);

            onsites(ALL) {
                if (X.coordinate(d) == lattice.size(d) - 1)
                    f2[X] = -f1[X];
                else
                    f2[X] = f1[X];
            }

            dif1 = 0;
            onsites(ALL) { dif1 += f1[X] - f2[X - d]; }

            if (dif1 != 0) {
                hila::out0 << " Antiperiodic up-gather test error! Node " << hila::myrank()
                        << " direction " << (unsigned)d << '\n';
                hila::terminate(1);
            }
        }
#endif
    }
#else
    //test correct working of generalized nn gathering:
    Field<CoordinateVector> f;
    onsites(ALL) {
        f[X] = X.coordinates();
    }

    for (int inntopo = 0; inntopo < lattice.nn_map.size(); ++inntopo) {
        //hila::out << "rank " << hila::myrank() << " : testing nn_map_" << inntopo << "\n";
        hila::out0 << "testing nn_map_" << inntopo << "\n";
        f.set_nn_topo(inntopo);
        //hila::out << "rank " << hila::myrank() << " : nn-topo switched to " << inntopo << "\n";
        bool terminate = false;
        foralldir(d) {
            {
                Field<CoordinateVector> f1;
                onsites(ALL) {
                    f1[X] = f[X + d];
                }
                // Since (*lattice.nn_map)(l, d) can't be called inside site-loop, we transfer the
                // field to node 0 (should probably be turned off for large lattices)
                auto f1l = f1.get_slice({-1, -1, -1, -1});
                if (hila::myrank() == 0) {
                    CoordinateVector l, ln;
                    CoordinateVector dif1 = 0;
                    for (size_t i = 0; i < lattice.volume(); ++i) {
                        l = lattice.global_coordinates(i);
                        ln = (*(lattice.nn_map[inntopo]))(l, d);
                        dif1 = abs(f1l[i] - ln);
                        if (dif1.squarenorm() != 0) {
                            hila::out0 << " gen std up-gather test error! Node " << lattice.node_rank(l)
                                    << " direction " << (unsigned)d << " (to node "
                                    << lattice.node_rank(ln) << ")" << " dif1=(" << dif1 << ") l=("
                                    << l << ") ln=(" << ln << ") f1l=(" << f1l[i] << ")" << '\n';
                            terminate = true;
                        }
                    }
                }
            }
            {
                Field<CoordinateVector> f2;
                onsites(ALL) {
                    f2[X] = f[X - d];
                }
                auto f2l = f2.get_slice({-1, -1, -1, -1});
                if (hila::myrank() == 0) {
                    CoordinateVector l, ln;
                    CoordinateVector dif2 = 0;
                    for (size_t i = 0; i < lattice.volume(); ++i) {
                        l = lattice.global_coordinates(i);
                        ln = (*(lattice.nn_map[inntopo]))(l, -d);
                        dif2 = abs(f2l[i] - ln);
                        if (dif2.squarenorm() != 0) {
                            hila::out0 << " gen std down-gather test error! Node "
                                    << lattice.node_rank(l) << " direction " << (unsigned)d
                                    << " (to node " << lattice.node_rank(ln) << ")" << " dif2=("
                                    << dif2 << ") l=(" << l << ") ln=(" << ln << ") f2l=(" << f2l[i]
                                    << ")" << '\n';
                            terminate = true;
                        }
                    }
                }
            }
        }
        hila::broadcast(terminate);
        if (terminate) {
            hila::terminate(1);
        }
        //hila::out << "rank " << hila::myrank() << " : testing of nn_map_" << inntopo << " completed\n";
        hila::out0 << "testing of nn_map_" << inntopo << " completed\n";
    }

    if(lattice.nn_map.size() > 1) {
        hila::out0 << "testing field referencing with different nn-topology\n";
        // the following creates a new Field instance "rf" whose data references the field data of
        // Field "f". "rf" can then have independent nn-topology settings
        Field<CoordinateVector> rf(f, 0);
        bool terminate = false;
        foralldir(d) {
            {
                Field<CoordinateVector> f1;
                onsites(ALL) {
                    f1[X] = rf[X + d];
                }
                // Since (*lattice.nn_map)(l, d) can't be called inside site-loop, we transfer the
                // field to node 0 (should probably be turned off for large lattices)
                auto f1l = f1.get_slice({-1, -1, -1, -1});
                if (hila::myrank() == 0) {
                    CoordinateVector l, ln;
                    CoordinateVector dif1 = 0;
                    for (size_t i = 0; i < lattice.volume(); ++i) {
                        l = lattice.global_coordinates(i);
                        ln = (*(lattice.nn_map[0]))(l, d);
                        dif1 = abs(f1l[i] - ln);
                        if (dif1.squarenorm() != 0) {
                            hila::out0 << " gen std up-gather test error! Node "
                                       << lattice.node_rank(l) << " direction " << (unsigned)d
                                       << " (to node " << lattice.node_rank(ln) << ")" << " dif1=("
                                       << dif1 << ") l=(" << l << ") ln=(" << ln << ") f1l=("
                                       << f1l[i] << ")" << '\n';
                            terminate = true;
                        }
                    }
                }
            }
            {
                Field<CoordinateVector> f2;
                onsites(ALL) {
                    f2[X] = rf[X - d];
                }
                auto f2l = f2.get_slice({-1, -1, -1, -1});
                if (hila::myrank() == 0) {
                    CoordinateVector l, ln;
                    CoordinateVector dif2 = 0;
                    for (size_t i = 0; i < lattice.volume(); ++i) {
                        l = lattice.global_coordinates(i);
                        ln = (*(lattice.nn_map[0]))(l, -d);
                        dif2 = abs(f2l[i] - ln);
                        if (dif2.squarenorm() != 0) {
                            hila::out0 << " gen std down-gather test error! Node "
                                       << lattice.node_rank(l) << " direction " << (unsigned)d
                                       << " (to node " << lattice.node_rank(ln) << ")" << " dif2=("
                                       << dif2 << ") l=(" << l << ") ln=(" << ln << ") f2l=("
                                       << f2l[i] << ")" << '\n';
                            terminate = true;
                        }
                    }
                }
            }
        }
        hila::out0 << "testing of field referencing with different nn-topology completed\n";
    }
#endif
}

void test_std_gathers() {
    // gather_test<int>();
    auto t0 = hila::gettime();

#ifdef MPI_BENCHMARK_TEST
    hila::out0 << "MPI_BENCHMARK_TEST defined, not doing communication tests!\n";
    return;
#endif

    gather_test();

#if defined(CUDA) || defined(HIP)
    gpuMemPoolPurge();
#endif

    hila::out0 << "Communication tests done - time " << hila::gettime() - t0 << "s\n";

    hila::print_dashed_line();

    if (hila::myrank() == 0)
        hila::out.flush();
}
