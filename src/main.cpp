//#include "fmt/core.h"
#include "simulator.hpp"
#include <iostream>
#include <optional>
#include <stdlib.h>
#include "omp.h"
//#include "acc.h"
//#include "MPIHandler.hpp"

int main(int argc, char** argv) {
    //MPIHandler::getInstance()->setArgs(argc, argv);
    // Size from command line if specified
    Simulator::SizeType grid = 256;
    if (argc > 1) {
        auto tmp = atoi(argv[1]);
        if (tmp <= 0) {
            std::cerr << "Grid should be larger than 0\n";
            return EXIT_FAILURE;
        }
        grid = static_cast<Simulator::SizeType>(tmp);
    }
    std::optional<unsigned> maxSteps;
    if (argc > 2) {
        auto tmp = atoi(argv[2]);
        if (tmp <= 0) {
            std::cerr << "MaxSteps should be larger than 0\n";
            return EXIT_FAILURE;
        }
        maxSteps = static_cast<decltype(maxSteps)::value_type>(tmp);
    }
    Simulator s{ grid };
    //printf("The size of the grid is {}\n", grid);
    Simulator::setPrinting(true);
    //printf("Number of used OpenMP threads is %d\n", omp_get_max_threads());
    if (maxSteps) {
        printf("The maximum number of steps is %d\n", maxSteps.value());
        s.run(4.5, 100.0, maxSteps.value());
    } else {
        printf("There is no maximum steps, it will run until it goes into a steady state");
        s.run(4.5, 100.0);
    }
}
