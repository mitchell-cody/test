#include <serial.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>

#include <json.hpp>

#include <common.hpp>

//
// Benchmarking program
//

namespace serial {
//
// Serial implementation of simulation
//
    void serial(nlohmann::json &cli_parameters) {
        int navg;
        int nabsavg = 0;
        double davg;
        double dmin;
        double absmin = 1.0;
        double absavg = 0.0;

        // Check number of particles for simulation
        int n = cli_parameters["number_particles"];

        // Check if output files should be saved.
        std::string save_filename = cli_parameters["output_filename"];
        std::string summary_filename = cli_parameters["summary_filename"];

        // Open files (if specified)
        std::ofstream fsave;
        if (save_filename.empty()) {
            fsave.open(nullptr);
        } else {
            fsave.open(save_filename);
        }

        std::ofstream fsum;
        if (summary_filename.empty()) {
            fsum.open(nullptr);
        } else {
            fsum.open(summary_filename, std::ios::app);
        }

        // Initialize particles for simulation
        std::vector<particle_t> particles(static_cast<unsigned long>(n));
        set_size(n);
        init_particles(n, particles);
        const double density = 0.0005;
        const double cutoff = 0.01;
        double size = sqrt(density * n);

        /*
         * Set size of grid
         * turn grid into 1 d array
         * determine what bin particles are in
         */
        double rad_int = (2 * cutoff);
        double grid_len = ceil(size/ rad_int);
        std::multimap<int,int> grid_b;
        std::multimap<int,int>::iterator it;
        std::map<int,int> grid_p;
        std::pair <std::multimap<int,int>::iterator, std::multimap<int,int>::iterator> ret;

        for (int i = 0; i < n; i++)
        {
            int px = particles[i].x / rad_int;
            int py = particles[i].y / rad_int;
            int grid_bin = px*grid_len + py;
            grid_b.insert( std::pair<int,int>(grid_bin,i) );
            grid_p[i] = grid_bin;
        }

        // Disable checks and output
        bool disable_checks = cli_parameters["disable_checks"];

        //
        // Simulate a number of time steps
        //

        auto timer = read_timer();

        for (int step = 0; step < NSTEPS; step++) {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            //
            // Compute forces
            //
            /*
             * Loop over bins instead of all particles
             * loop over x and y dimensions of the grid
             * loop over all particles in the chosen grid
             *
             */
            // loop particles
            for (int i = 0; i < n; i++) {
                particles[i].ax = 0.0;
                particles[i].ay = 0.0;
                int parti_bin = grid_p[i];
                // find the adjacent blocks
                for (int adj_i = -1; adj_i < 2; adj_i++) {
                    for (int adj_k = -1; adj_k < 2; adj_k++) {
                        // look at the particles in adjacent bins
                        int bin = parti_bin + (adj_i * grid_len + adj_k);
                        ret = grid_b.equal_range(bin);
                        for (std::multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it){
                            apply_force(particles[i], particles[(*it).second], dmin, davg, navg);
                                }
                            }
                        }
                    }


            //need to account for particles already used

            //
            // Move particles
            //
            for (int i = 0; i < n; i++) {
                move(particles[i]);
            }

            //near to clear and rebind
            grid_b.clear();
            grid_p.clear();
            for (int i = 0; i < n; i++)
            {
                int px = particles[i].x / rad_int;
                int py = particles[i].y / rad_int;
                int grid_bin = px*grid_len + py;
                grid_b.insert( std::pair<int,int>(grid_bin,i) );
                grid_p[i] = grid_bin;
            }

            if (!disable_checks) {
                //
                // Computing statistical data
                //
                if (navg) {
                    absavg += davg / navg;
                    nabsavg++;
                }
                if (dmin < absmin) {
                    absmin = dmin;
                }

                //
                // Save if necessary
                //
                if (fsave && (step % SAVEFREQ) == 0) {
                    save(fsave, n, particles);
                }
            }
        }

        auto simulation_time =
                std::chrono::duration<double, std::micro>(read_timer() - timer).count()
                / 1e6;

        std::cout << "n = " << n << ", simulation time = " << simulation_time
                  << " seconds";

        if (!disable_checks) {
            if (nabsavg) {
                absavg /= nabsavg;
            }
            // - the minimum distance absmin between 2 particles during the run of
            //   the simulation
            // - A Correct simulation will have particles stay at greater than 0.4
            //   (of cutoff) with typical values between .7-.8
            // - A simulation were particles don't interact correctly will be less
            //   than 0.4 (of cutoff) with typical values between .01-.05
            // - The average distance absavg is ~.95 when most particles are
            //   interacting correctly and ~.66 when no particles are interacting
            std::cout << ", absmin = " << absmin << ", absavg = " << absavg;
            if (absmin < 0.4) {
                std::cout << "\nThe minimum distance is below 0.4 meaning that some "
                        "particle is not interacting";
            }
            if (absavg < 0.8) {
                std::cout << "\nThe average distance is below 0.8 meaning that most "
                        "particles are not interacting";
            }
        }
        std::cout << "\n";

        //
        // Printing summary data
        //
        if (fsum) {
            fsum << n << " " << simulation_time << "\n";
        }
    }
}

