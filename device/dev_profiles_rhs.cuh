#pragma once

#include "dev_common.cuh"


__global__ void init_background_distribution_form1(qreal* T, qreal* n)
{
    uint32_t iv = threadIdx.x;  // velocity id;
    uint32_t ix = blockIdx.x;   // space id;

    // row-id of the resulting (Nx*Nv)\times(Nx*Nv) matrix;
    uint32_t ir = ix * dev_dd_.Nv + iv; 

    qreal T1 = T[ix];
    qreal n1 = n[ix];

    qreal v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv);
    qreal v2 = v1*v1;

    qreal coef = n1 / sqrt(2*T1*M_PI);
    coef = coef * dev_dd_.dv / T1;
 
    dev_dd_.FB[ir] = coef * exp(-v2/(2*T1)); // dv * Maxwellian / T
}


/**
 * Initialize the right-hand-side vector: Guassian-shaped source for E at x0; 
 * case WITHOUT copies of E;
 * here, iv = 0 always;
*/
__global__ void init_rhs_gauss()
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x; // space id;

    // there is a shift because the source is set for the electric field;
    uint32_t sh_E = dev_dd_.Nx * dev_dd_.Nv;

    // row-id of the resulting 2*(Nx*Nv) vector;
    uint32_t ir = ix * dev_dd_.Nv + sh_E; 

    qreal x1 = get_x1(dev_dd_.h, ix);
    qreal dr = x1 - dev_dd_.x0;
    qreal dr2 = dr * dr;
    qreal ds2 = 2. * dev_dd_.ds * dev_dd_.ds;

    qreal coef_ch = dev_dd_.w; // because the source is derived from the charge source;

    // qreal coef_norm = 1./ (ds*sqrt(2.*M_PI));
    qreal coef_norm = 1.0;


    dev_dd_.b[ir].x = 0.0;

    // in imag because the source is derived from the charge source;
    dev_dd_.b[ir].y = coef_ch * coef_norm * exp(-dr2/ds2); 
}