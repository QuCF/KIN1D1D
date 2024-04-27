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
    uint32_t Ndv_bo; // for the second derivative in velocity at the velocity boundary;
    uint32_t Ndv; // for the second derivative in velocity in bulk velocity points;

    qreal w;  // normalized frequency;
    qreal h;  // normalized spatial step;
    qreal dv; // normalized velocity step;

    qreal diff; // diffusivity in the velocity space;

    qreal xmax; // maximum x-coordinate normalized to the Debye length;
    qreal vmax; // maximum x-velocity normalized to the thermal velocity;

    SpMatrixC A; // Matrix describing the kinetic problem (Ax = b); [on device].
    ycuComplex* b; // Right-hand-side vector; on device]/
    ycuComplex* psi; // Solution of the system A*psi=b; [on device].
    qreal* FB; // background distribution function [x,v];
    qreal* Y;  // combined background profiles [x,v];

    qreal x0; // source center;
    qreal ds; // source width;

    void set_to_zero()
    {
        nx = 0;
        Nx = 0;

        nv = 0; 
        Nv = 0;

        Nv_h = 0; 
    }
};