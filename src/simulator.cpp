#include "simulator.hpp"
//#include "fmt/core.h"
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
//#include <mpi.h>
//#include "MPIHandler.hpp"
//#include "MPILaplace.hpp"

void Simulator::setPrinting(bool toPrint) { printing = toPrint; }

void Simulator::initU() {
    for (SizeType i = 0; i <= (grid - 1); i++) {
        u[(i) * (grid + 1) + grid] = 1.0;
        u[(i) * (grid + 1) + grid - 1] = 1.0;
        for (SizeType j = 0; j < (grid - 1); j++) {
            u[(i) * (grid + 1) + j] = 0.0;
        }
    }
}

void Simulator::initV() {
    for (SizeType i = 0; i <= (grid); i++) {
        for (SizeType j = 0; j <= (grid); j++) {
            v[(i)*(grid + 1) + j] = 0.0;
        }
    }
}

void Simulator::initP() {
    for (SizeType i = 0; i <= (grid); i++) {
        for (SizeType j = 0; j <= (grid); j++) {
            p[(i) * (grid + 1) + j] = 1.0;
        }
    }
}

void Simulator::solveUMomentum(const FloatType Re) {
	#pragma acc parallel loop independent collapse(2) present( u, un, v, vn, p, pn, m)
    for (SizeType i = 1; i <= (grid - 2); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            un[(i) * (grid + 1) + j] = u[(i) * (grid + 1) + j]
                - dt
                    * ((u[(i + 1) * (grid + 1) + j] * u[(i + 1) * (grid + 1) + j] - u[(i - 1) * (grid + 1) + j] * u[(i - 1) * (grid + 1) + j]) / 2.0 / dx
                    + 0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*(grid + 1) + j] + v[(i + 1) * (grid + 1) + j])
                            - (u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) * (v[(i + 1) * (grid + 1) + j - 1] + v[(i)*(grid + 1) + j - 1])) / dy)
                    - dt / dx * (p[(i + 1) * (grid + 1) + j] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                    * ((u[(i + 1) * (grid + 1) + j] - 2.0 * u[(i) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j]) / dx / dx
                     + (u[(i) * (grid + 1) + j + 1] - 2.0 * u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j - 1]) / dy / dy);
        }
    }
}

void Simulator::applyBoundaryU() {
	#pragma acc parallel loop independent collapse(1) present( u, un, v, vn, p, pn, m)
    for (SizeType j = 1; j <= (grid - 1); j++) {
        un[(0) * (grid + 1) + j] = 0.0;
        un[(grid - 1) * (grid + 1) + j] = 0.0;
    }

    for (SizeType i = 0; i <= (grid - 1); i++) {
        un[(i) * (grid + 1) + 0] = -un[(i) * (grid + 1) + 1];
        un[(i) * (grid + 1) + grid] = 2 - un[(i) * (grid + 1) + grid - 1];
    }
}

void Simulator::solveVMomentum(const FloatType Re) {

    #pragma acc parallel loop independent collapse(2) present( u, un, v, vn, p, pn, m)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 2); j++) {
            vn[(i)*(grid + 1) + j] = v[(i)*(grid + 1) + j]
                - dt * (0.25 * ((u[(i) * (grid + 1) + j] + u[(i) * (grid + 1) + j + 1]) * (v[(i)*(grid + 1) + j] + v[(i + 1) * (grid + 1) + j])
                              - (u[(i - 1) * (grid + 1) + j] + u[(i - 1) * (grid + 1) + j + 1]) * (v[(i)*(grid + 1) + j] + v[(i - 1) * (grid + 1) + j])) / dx
                              + (v[(i)*(grid + 1) + j + 1] * v[(i)*(grid + 1) + j + 1] - v[(i)*(grid + 1) + j - 1] * v[(i)*(grid + 1) + j - 1]) / 2.0 / dy)
                              - dt / dy * (p[(i) * (grid + 1) + j + 1] - p[(i) * (grid + 1) + j]) + dt * 1.0 / Re
                              * ((v[(i + 1) * (grid + 1) + j] - 2.0 * v[(i)*(grid + 1) + j] + v[(i - 1) * (grid + 1) + j]) / dx / dx
                              + (v[(i)*(grid + 1) + j + 1] - 2.0 * v[(i)*(grid + 1) + j] + v[(i)*(grid + 1) + j - 1]) / dy / dy);
        }
    }
}

void Simulator::applyBoundaryV() {
	#pragma acc parallel loop independent collapse(1) present( vn)
    for (SizeType j = 1; j <= (grid - 2); j++) {
        vn[(0) * (grid + 1) + j] = -vn[(1) * (grid + 1) + j];
        vn[(grid)*(grid + 1) + j] = -vn[(grid - 1) * (grid + 1) + j];
    }
	#pragma acc parallel loop independent collapse(1) present( vn)
    for (SizeType i = 0; i <= (grid); i++) {
        vn[(i)*(grid + 1) + 0] = 0.0;
        vn[(i)*(grid + 1) + grid - 1] = 0.0;
    }
}

void Simulator::solveContinuityEquationP(const FloatType delta) {
	#pragma acc parallel loop independent collapse(2) present(  un,  vn, p, pn)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            pn[(i) * (grid + 1) + j] = p[(i) * (grid + 1) + j]
                - dt * delta * ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*(grid + 1) + j] - vn[(i)*(grid + 1) + j - 1]) / dy);
        }
    }
}

void Simulator::applyBoundaryP() {
	#pragma acc parallel loop independent collapse(1) present( u, un, v, vn, p, pn, m)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        pn[(i) * (grid + 1) + 0] = pn[(i) * (grid + 1) + 1];
        pn[(i) * (grid + 1) + grid] = pn[(i) * (grid + 1) + grid - 1];
    }
	#pragma acc parallel loop independent collapse(1) present( u, un, v, vn, p, pn, m)
    for (SizeType j = 0; j <= (grid); j++) {
        pn[(0) * (grid + 1) + j] = pn[(1) * (grid + 1) + j];
        pn[(grid) * (grid + 1) + j] = pn[(grid - 1) * (grid + 1) + j];
    }
}

Simulator::FloatType Simulator::calculateError() {
    FloatType error = 0.0;
	#pragma acc parallel loop collapse(2) reduction(+:error) present(un,vn, m)
    for (SizeType i = 1; i <= (grid - 1); i++) {
        for (SizeType j = 1; j <= (grid - 1); j++) {
            m[(i) * (grid + 1) + j] =
                ((un[(i) * (grid + 1) + j] - un[(i - 1) * (grid + 1) + j]) / dx + (vn[(i)*(grid + 1) + j] - vn[(i)*(grid + 1) + j - 1]) / dy);
            error += fabs(m[(i) * (grid + 1) + j]);
        }
    }

    return error;
}

void Simulator::iterateU() {
    //std::swap(u, un);
	#pragma acc parallel loop independent collapse(2) present( u, un, v, vn, p, pn, m)
     for (SizeType i = 0; i <= (grid - 1); i++) {
         for (SizeType j = 0; j <= (grid); j++) {
             u[(i) * (grid + 1) + j] = un[(i) * (grid + 1) + j];
         }
     }
}

void Simulator::iterateV() {
    //std::swap(v, vn);
	#pragma acc parallel loop independent collapse(2) present( u, un, v, vn, p, pn, m)
	 for (SizeType i = 0; i <= (grid); i++) {
	     for (SizeType j = 0; j <= (grid - 1); j++) {
	         v[(i)*(grid + 1) + j] = vn[(i)*(grid + 1) + j];
	     }
	 }
}

void Simulator::iterateP() {
    //std::swap(p, pn);
	#pragma acc parallel loop independent collapse(2) present( u, un, v, vn, p, pn, m)
     for (SizeType i = 0; i <= (grid); i++) {
         for (SizeType j = 0; j <= (grid); j++) {
             p[(i) * (grid + 1) + j] = pn[(i) * (grid + 1) + j];
         }
     }
}

void Simulator::deallocate() {

	delete u;
	delete un;
	delete v;
	delete vn;
	delete p;
	delete pn;
	delete m;

    // it doesn't do anything until we use vectors
    // because that deallocates automatically
    // but if we have to use a more raw data structure later it is needed
    // and when the the Tests overwrites some member those might won't deallocate
}

Simulator::Simulator(SizeType gridP)
    : grid([](auto g) {
          if (g <= 1) {
              throw std::runtime_error("Grid is smaller or equal to 1.0, give larger number");
          }
          return g;
      }(gridP)),
      dx(1.0 / static_cast<FloatType>(grid - 1)),
      dy(1.0 / static_cast<FloatType>(grid - 1)),
      dt(0.001 / std::pow(grid / 128.0 * 2.0, 2.0)) {

    //MPIHandler::getInstance()->handleMPIResource();
    //MPISetup(&grid, &grid);

    u=	new double[(grid + 1) * (grid + 1)];
    un=	new double[(grid + 1) * (grid + 1)];
    v=	new double[(grid + 1) * (grid + 1)];
    vn=	new double[(grid + 1) * (grid + 1)];
    p=	new double[(grid + 1) * (grid + 1)];
    pn=	new double[(grid + 1) * (grid + 1)];
    m=	new double[(grid + 1) * (grid + 1)];

    initU();
    initV();
    initP();
}

void Simulator::run(const FloatType delta, const FloatType Re, unsigned maxSteps) {
    if (printing) {
        printf("Running simulation with delta: %f, Re: %f\n", delta, Re);
    }
    auto error = std::numeric_limits<FloatType>::max();
    unsigned step = 1;
	unsigned int size_a=(grid + 1) * (grid + 1);
	#pragma acc data copy( u[size_a], un[size_a], v[size_a], vn[size_a], p[size_a], pn[size_a], m[size_a])
	{
    while (error > 0.00000001 && step <= maxSteps) {
        solveUMomentum(Re);
        applyBoundaryU();

        solveVMomentum(Re);
        applyBoundaryV();

        solveContinuityEquationP(delta);
        applyBoundaryP();

        error = calculateError();

        if (printing && (step % 1000 == 1)) {
            printf("Error is %f for the step %d\n", error, step);
        }

        iterateU();
        iterateV();
        iterateP();
        ++step;
    }
	}
}

Simulator::~Simulator() { deallocate(); }
