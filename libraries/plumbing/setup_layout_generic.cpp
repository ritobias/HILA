/// Setup layout does the node division.  This version
/// first tries an even distribution, with equally sized
/// nodes, and if that fails allows slightly different
/// node sizes.

#include "plumbing/defs.h"
#include "plumbing/lattice.h"

/***************************************************************/

/* number of primes to be used in factorization */
#define NPRIMES 12
const static int prime[NPRIMES] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

/* Set up now squaresize and nsquares - arrays
 * Print info to outf as we proceed
 */

size_t lattice_struct::block_boundary_size(const CoordinateVector &blsiz) {
    size_t tsa = 0;
    size_t tvol = 1;
    foralldir(d) tvol *= blsiz[d];
    assert(tvol > 0 && "tvol=0 in lattice_struct::block_boundary_size()");
    foralldir(d) tsa += tvol / blsiz[d];
    return 2 * tsa;
}

void lattice_struct::setup_layout() {
    int nfactors[NPRIMES];
    CoordinateVector nodesiz;

    hila::print_dashed_line();
    hila::out0 << "LAYOUT: lattice size  ";
    foralldir(d) {
        if (d != 0)
            hila::out0 << " x ";
        hila::out0 << size(d);
    }
    hila::out0 << "  =  " << l_volume << " sites\n";
    hila::out0 << "Dividing to " << hila::number_of_nodes() << " nodes\n";

#ifndef GENGATHER
    foralldir(d) if (size(d) % 2 != 0) {
        hila::out0 << "Lattice must be even to all directions (odd size:TODO)\n";
        hila::finishrun();
    }
#endif
    // Factorize the node number in primes
    // These factors must be used in slicing the lattice!
    int nn = hila::number_of_nodes();

    int i = nn;
    for (int n = 0; n < NPRIMES; n++) {
        nfactors[n] = 0;
        while (i % prime[n] == 0) {
            i /= prime[n];
            nfactors[n]++;
        }
    }
    if (i != 1) {
        hila::out0 << "Cannot factorize " << nn << " nodes with primes up to " << prime[NPRIMES - 1]
                   << '\n';
        hila::finishrun();
    }

    int64_t remainder = l_volume % nn; // remainder = 0 even division



    if(1) {

        CoordinateVector nsize;
        int64_t ghosts[NDIM]; // int is too small
        int64_t maxnghostl = 1;
        foralldir(d) {
            int64_t cosize = l_volume / size(d);
            int64_t n = size(d);
            while ((n * cosize) % nn != 0)
                n++; // virtual size can be odd
            // now nsize is the new would-be size
            int64_t tnghostl = n - size(d);
            ghosts[d] = tnghostl * cosize;
            nsize[d] = n;
            if(tnghostl > maxnghostl) {
                maxnghostl = tnghostl;
            }
        }

        int mdir = 0;

        foralldir(j) if (ghosts[j] < ghosts[mdir]) mdir = j;
        // mdir is the direction where we do uneven division (if done)
        // hila::out0 << "MDIR " << mdir << " ghosts mdir " << ghosts[mdir] << " nsize " <<
        // nsize[mdir] << '\n';

        int64_t g_nvol = (l_volume + ghosts[mdir]) / nn;  // node volume

        // find node shape that minimizes boundary:
        std::vector<size_t> fx(NDIM);
        CoordinateVector tNx, tndiv;
        std::vector<std::vector<int>> nlxp(NDIM); //list of possible side lengths
        std::vector<std::vector<int>> ndivlxp(NDIM); // corresponding list of divisions
        foralldir(d) {
            size_t maxsiz;
            for (int td = 0; td <= maxnghostl; ++td) {
                maxsiz = size(d) + td;
                //if(d == mdir) {
                //    maxsiz = nsize[d];
                //}
                for (size_t i = 1; i <= maxsiz / 2; ++i) {
                    if(maxsiz % i == 0 && nn % i == 0 && size(d) / i >= 2) {
                        nlxp[d].push_back(maxsiz / i);
                        ndivlxp[d].push_back(i);
                    }
                }
            }
            fx[d] = 0;
            tNx[d] = maxsiz; // initial size is full (ghosted) lattice
            tndiv[d] = 1;
        }

        nodesiz = tNx;
        nodes.n_divisions = tndiv;
        size_t minbndry = block_boundary_size(tNx) + 1; // initial boundary size
        size_t minV = 1;
        foralldir(d) minV *= tNx[d]; // initial volume
        minV /= nn;
        size_t ttV, tbndry;
        bool succ = false;
        // now run through all shapes that can be formed with the side lengths listed in nlxp:
        while (true) {
            ttV = 1;
            foralldir(d) {
                tNx[d] = nlxp[d][fx[d]];
                tndiv[d] = ndivlxp[d][fx[d]];
                ttV *= tNx[d];
            }
            if (ttV >= g_nvol) {
                // shape fits the node volume
                tbndry = block_boundary_size(tNx);
                if (ttV < minV || (ttV == minV && tbndry < minbndry)) {
                    // boundary of current shape is smaller than minbndry
                    //  -> update minbndry and nodesiz to current shape
                    minV = ttV;
                    minbndry = tbndry;
                    nodesiz = tNx;
                    nodes.n_divisions = tndiv;
                    succ = true;
                }
            }

            // determine next shape in nlxp:
            ++fx[0];
            for (int d = 0; d < NDIM - 1; ++d) {
                if(fx[d] >= nlxp[d].size()) {
                    ++fx[d + 1];
                    fx[d] = 0;
                }
            }
            if(fx[NDIM - 1] >= nlxp[NDIM - 1].size()) {
                // no more shapes left
                break;
            }
        }

        if (!succ) {
            hila::out0 << "Could not successfully lay out the lattice with "
                       << hila::number_of_nodes() << " nodes\n";
            hila::finishrun();
        }

        // set up struct nodes variables
        nodes.number = hila::number_of_nodes();
        foralldir(dir) {
            nodes.divisors[dir].resize(nodes.n_divisions[dir] + 1);
            // Node divisors: note, this MUST BE compatible with
            // node_rank in lattice.cpp
            // to be sure, we use naively the same method than in node_rank
            // last element will be size(dir), for convenience
            int n = -1;
            for (int i = 0; i <= size(dir); i++)
                if ((i * nodes.n_divisions[dir]) / size(dir) != n) {
                    ++n;
                    nodes.divisors[dir][n] = i;
                }
            // hila::out0 << "Divisors ";
            // for (int i=0;i<nodes.n_divisions[dir]; i++) hila::out0 << nodes.divisors[dir][i]
            // << " "; hila::out0 << '\n';
        }
        CoordinateVector ghost_slices;
        foralldir(d) {
            ghost_slices[d] = nodes.n_divisions[d] * nodesiz[d] - size(d);
            if (ghost_slices[d] > 0) {
                hila::out0 << "\nUsing uneven node division to direction " << d << ":\n";
                hila::out0 << "Divisions: ";
                for (int i = 0; i < nodes.n_divisions[d]; i++) {
                    if (i > 0)
                        hila::out0 << " - ";
                    hila::out0 << nodes.divisors[d][i + 1] - nodes.divisors[d][i];
                }
                hila::out0 << "\nFilling efficiency: " << (100.0 * size(d)) / (size(d) + ghost_slices[d]) << "%\n";

                if (ghost_slices[d] > nodes.n_divisions[d] / 2)
                    hila::out0 << "NOTE: number of smaller nodes > large nodes \n";
            }
        }

        // this was hila::number_of_nodes() > 1
        if (1) {
            hila::out0 << "\nSites on node: ";
            foralldir(d) {
                if (d > 0)
                    hila::out0 << " x ";
                if (ghost_slices[d] > 0)
                    hila::out0 << '(' << size(d) / nodes.n_divisions[d] << '-' << nodesiz[d] << ')';
                else
                    hila::out0 << nodesiz[d];
            }
            int ns = 1;
            foralldir(d) ns *= nodesiz[d];
            int ns2 = 1;
            foralldir(d) ns2 *= size(d) / nodes.n_divisions[d];
            if(ns2 != ns) {
                hila::out0 << "  =  " << ns2 << " - " << ns << '\n';
            } else {
                hila::out0 << "  =  " << ns << '\n';
            }

            hila::out0 << "Processor layout: ";
            foralldir(d) {
                if (d > 0)
                    hila::out0 << " x ";
                hila::out0 << nodes.n_divisions[d];
            }
            hila::out0 << "  =  " << hila::number_of_nodes() << " nodes\n";
        }
    } else {
        // strategy: try to increase the box size to one of the directions until rem = 0
        // find the optimal direction to do it
        // Use simple heuristic: take the dim with the least amount of added "ghost sites"

        CoordinateVector nsize;
        int64_t ghosts[NDIM]; // int is too small

        foralldir(d) {
            int64_t cosize = l_volume / size(d);
            int64_t n = size(d);
            while ((n * cosize) % nn != 0)
                n++; // virtual size can be odd
            // now nsize is the new would-be size
            ghosts[d] = (n - size(d)) * cosize;
            nsize[d] = n;
        }

        int mdir = 0;

        foralldir(j) if (ghosts[mdir] > ghosts[j]) mdir = j;
        // mdir is the direction where we do uneven division (if done)
        // hila::out0 << "MDIR " << mdir << " ghosts mdir " << ghosts[mdir] << " nsize " <<
        // nsize[mdir] << '\n';

        bool secondtime = false;
        do {
            // try the division a couple of times, if the 1st fails

            foralldir(i) {
                nodesiz[i] = (i == mdir) ? nsize[i] : size(i); // start with ghosted lattice size
                nodes.n_divisions[i] = 1;
            }

            for (int n = NPRIMES - 1; n >= 0; n--) {
                for (int i = 0; i < nfactors[n]; i++) {
                    // figure out which direction to divide -- start from the largest prime,
                    // because we don't want this to be last divisor! (would probably wind up
                    // with size 1)

                    // find largest divisible dimension of h-cubes - start from last, because
                    int msize = 1, dir;
                    for (dir = 0; dir < NDIM; dir++)
                        if (nodesiz[dir] > msize && nodesiz[dir] % prime[n] == 0)
                            msize = nodesiz[dir];

                    // if one direction with largest dimension has already been
                    // divided, divide it again.  Otherwise divide first direction
                    // with largest dimension.

                    // Switch here to first divide along t-direction, in
                    // order to
                    // a) minimize spatial blocks
                    // b) In sf t-division is cheaper (1 non-communicating slice)

                    for (dir = NDIM - 1; dir >= 0; dir--)
                        if (nodesiz[dir] == msize && nodes.n_divisions[dir] > 1 &&
                            nodesiz[dir] % prime[n] == 0)
                            break;

                    // If not previously sliced, take one direction to slice
                    if (dir < 0)
                        for (dir = NDIM - 1; dir >= 0; dir--)
                            if (nodesiz[dir] == msize && nodesiz[dir] % prime[n] == 0)
                                break;

                    if (dir < 0) {
                        // This cannot happen
                        hila::out0 << "CANNOT HAPPEN! in setup_layout_generic.c\n";
                        hila::finishrun();
                    }

                    // Now slice it
                    nodesiz[dir] /= prime[n];
                    nodes.n_divisions[dir] *= prime[n];
                }
            }

            // now check that the div makes sens
            bool fail = false;
            foralldir(dir) if (nodesiz[dir] < 2) fail = true; // don't allow nodes of size 1
            if (fail && !secondtime) {
                secondtime = true;
                ghosts[mdir] =
                    (1ULL << 62); // this short-circuits direction mdir, some other taken next
            } else if (fail) {
                hila::out0 << "Could not successfully lay out the lattice with "
                        << hila::number_of_nodes() << " nodes\n";
                hila::finishrun();
            }

        } while (secondtime);

        // set up struct nodes variables
        nodes.number = hila::number_of_nodes();
        foralldir(dir) {
            nodes.divisors[dir].resize(nodes.n_divisions[dir] + 1);
            // Node divisors: note, this MUST BE compatible with
            // node_rank in lattice.cpp
            // to be sure, we use naively the same method than in node_rank
            // last element will be size(dir), for convenience
            int n = -1;
            for (int i = 0; i <= size(dir); i++)
                if ((i * nodes.n_divisions[dir]) / size(dir) != n) {
                    ++n;
                    nodes.divisors[dir][n] = i;
                }
            // hila::out0 << "Divisors ";
            // for (int i=0;i<nodes.n_divisions[dir]; i++) hila::out0 << nodes.divisors[dir][i]
            // << " "; hila::out0 << '\n';
        }

        // Now division done - check how good it is
        int ghost_slices = nsize[mdir] - size(mdir);
        if (ghost_slices > 0) {
            hila::out0 << "\nUsing uneven node division to direction " << mdir << ":\n";
            hila::out0 << "Lengths: " << nodes.n_divisions[mdir] - ghost_slices << " * ("
                       << nodesiz[mdir] << " sites) + " << ghost_slices << " * ("
                       << nodesiz[mdir] - 1 << " sites)\n";
            hila::out0 << "Divisions: ";
            for (int i = 0; i < nodes.n_divisions[mdir]; i++) {
                if (i > 0)
                    hila::out0 << " - ";
                hila::out0 << nodes.divisors[mdir][i + 1] - nodes.divisors[mdir][i];
            }
            hila::out0 << "\nFilling efficiency: " << (100.0 * size(mdir)) / nsize[mdir] << "%\n";

            if (ghost_slices > nodes.n_divisions[mdir] / 2)
                hila::out0 << "NOTE: number of smaller nodes > large nodes \n";
        }

        // this was hila::number_of_nodes() > 1
        if (1) {
            hila::out0 << "\nSites on node: ";
            foralldir(dir) {
                if (dir > 0)
                    hila::out0 << " x ";
                if (dir == mdir && ghost_slices > 0)
                    hila::out0 << '(' << nodesiz[dir] - 1 << '-' << nodesiz[dir] << ')';
                else
                    hila::out0 << nodesiz[dir];
            }
            int ns = 1;
            foralldir(dir) ns *= nodesiz[dir];
            if (ghost_slices > 0) {
                int ns2 = ns * (nodesiz[mdir] - 1) / nodesiz[mdir];
                hila::out0 << "  =  " << ns2 << " - " << ns << '\n';
            } else {
                hila::out0 << "  =  " << ns << '\n';
            }

            hila::out0 << "Processor layout: ";
            foralldir(dir) {
                if (dir > 0)
                    hila::out0 << " x ";
                hila::out0 << nodes.n_divisions[dir];
            }
            hila::out0 << "  =  " << hila::number_of_nodes() << " nodes\n";
        }
    }





    // For MPI, remap the nodes for periodic torus
    // in the desired manner
    // we have at least 2 options:
    // map_node_layout_trivial.c
    // map_node_layout_block2.c - for 2^n n.n. blocks

    nodes.create_remap();


    hila::print_dashed_line();
}
