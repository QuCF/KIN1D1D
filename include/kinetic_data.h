#pragma once
#include "mix.h"


// ---------------------------------------------------------
// --- Data describing a stationary kinetic system ---
// ---------------------------------------------------------
struct KDATA
{
    uint32_t nx; //log2(Nx)
    uint32_t Nx; 

    uint32_t nv;
    uint32_t Nv;

    uint32_t Nv_h; // half of Nv

    uint32_t Nvars;

    uint32_t N_discr; // a constant defining the discretization;

    double w;  // normalized frequency;
    double h;  // normalized spatial step;
    double dv; // normalized velocity step;

    double xmax; // maximum x-coordinate normalized to the Debye length;
    double vmax; // maximum x-velocity normalized to the thermal velocity;

    SpMatrixC A; // Matrix describing the kinetic problem (Ax = b); [on device].
    cuDoubleComplex* b; // Right-hand-side vector; on device]/
    cuDoubleComplex* psi; // Solution of the system A*psi=b; [on device].
    double* FB; // background distribution function [x,v];
    double* Y;  // combined background profiles [x,v];

    void set_to_zero()
    {
        nx = 0;
        Nx = 0;

        nv = 0; 
        Nv = 0;

        Nv_h = 0; 
    }
};