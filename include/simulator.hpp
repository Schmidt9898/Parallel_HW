#pragma once
#include <experimental/filesystem>
#include <vector>
//#include <mpi.h>
//#include "MPILaplace.hpp"

class Simulator {
    static inline bool printing = false;
public:
    static void setPrinting(bool toPrint);

    typedef unsigned SizeType;
    typedef double FloatType;

    // these should be a private section (until constructor), but for testing and benchmarking we keep it public
    SizeType grid;
    
    const FloatType dx, dy, dt;
    double * __restrict__ u;
    double * __restrict__ un;
    double * __restrict__ v;
    double * __restrict__ vn;
    double * __restrict__ p;
    double * __restrict__ pn;
    double * __restrict__ m;

    // helper functions for constructor
    void initU();
    void initV();
    void initP();

    // helper functions for run
    void solveUMomentum(const FloatType Re);
    void applyBoundaryU();

    void solveVMomentum(const FloatType Re);
    void applyBoundaryV();

    void solveContinuityEquationP(const FloatType delta);
    void applyBoundaryP();

    FloatType calculateError();

    void iterateU();
    void iterateV();
    void iterateP();

    void deallocate();

public:
    Simulator(SizeType gridP);
    void run(const FloatType delta, const FloatType Re, unsigned maxSteps = std::numeric_limits<unsigned>::max());
    ~Simulator();
};
