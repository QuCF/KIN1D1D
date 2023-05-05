#pragma once

#include "dev_common.cuh"


__global__ void init_background_distribution_form1(double* T, double* n)
{
    uint32_t iv = threadIdx.x;  // velocity id;
    uint32_t ix = blockIdx.x;   // space id;

    // row-id of the resulting (Nx*Nv)\times(Nx*Nv) matrix;
    uint32_t ir = ix * dev_dd_.Nv + iv; 

    double T1 = T[ix];
    double n1 = n[ix];

    double v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv);
    double v2 = v1*v1;

    double coef = n1 / sqrt(2*T1*M_PI);
    coef = coef * dev_dd_.dv / T1;
 
    dev_dd_.FB[ir] = coef * exp(-v2/(2*T1)); // dv * Maxwellian / T
}


/**
 * Initialize the right-hand-side vector: Guassian-shaped source for E at x0; 
 * case WITHOUT copies of E;
 * here, iv = 0 always;
*/
__global__ void init_rhs_gauss(double x0, double ds, double w)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x; // space id;

    // there is a shift because the source is set for the electric field;
    uint32_t sh_E = dev_dd_.Nx * dev_dd_.Nv;

    // row-id of the resulting 2*(Nx*Nv) vector;
    uint32_t ir = ix * dev_dd_.Nv + sh_E; 

    double x1 = get_x1(dev_dd_.h, ix);
    double dr = x1 - x0;
    double dr2 = dr * dr;
    double ds2 = 2.*ds*ds;

    double coef_ch = w; // because the source is derived from the charge source;

    // double coef_norm = 1./ (ds*sqrt(2.*M_PI));
    double coef_norm = 1.0;


    dev_dd_.b[ir].x = 0.0;

    // in imag because the source is derived from the charge source;
    dev_dd_.b[ir].y = coef_ch * coef_norm * exp(-dr2/ds2); 
}


/**
 * Initialize the right-hand-side vector: up-down asymmetric source; 
 * case WITHOUT copies of E;
 * here, iv = 0 always;
*/
__global__ void init_rhs_woc_uda(double a_step, double r0, double ds)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x; // space id;

    // there is a shift because the source is set for the electric field;
    uint32_t sh_E = dev_dd_.Nx * dev_dd_.Nv;

    // row-id of the resulting 2*(Nx*Nv) vector;
    uint32_t ir = ix * dev_dd_.Nv + sh_E; 

    double r1 = get_x1(dev_dd_.h, ix) / dev_dd_.xmax;
    double dr = r1-r0;
    double f_step = 2. / (1. + exp(- a_step * dr)) - 1.;

    // double r2 = dr*dr;
    // double wg2 = 2*ds*ds;
    // dev_dd_.b[ir].x = f_step * exp(-r2/wg2);
    dev_dd_.b[ir].x = f_step;

    dev_dd_.b[ir].y = 0.0;
}


/**
 * Initialize the right-hand-side vector: up-down asymmetric source (Erf); 
 * case WITHOUT copies of E;
 * here, iv = 0 always;
*/
__global__ void init_rhs_woc_erf(double r0)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x; // space id;

    // there is a shift because the source is set for the electric field;
    uint32_t sh_E = dev_dd_.Nx * dev_dd_.Nv;

    // row-id of the resulting 2*(Nx*Nv) vector;
    uint32_t ir = ix * dev_dd_.Nv + sh_E; 

    // double r1 = get_x1(dev_dd_.h, ix) / dev_dd_.xmax;
    // double dr = r1-r0;
    // double ds = 0.02;
    // double xx = dr / (sqrt(2.)*ds);


    double r1 = get_x1(dev_dd_.h, ix);
    double dr = r1 - r0*dev_dd_.xmax;
    double ds = 1.0;
    double xx = dr / (sqrt(2.)*ds);

    dev_dd_.b[ir].x = 0.0;
    dev_dd_.b[ir].y = ds*erf(xx);
}