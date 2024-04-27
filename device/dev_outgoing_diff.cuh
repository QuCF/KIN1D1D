#pragma once

#include "dev_common.cuh"


// __constant__ qreal coef_plot = 10000000;
__constant__ qreal coef_plot = 0;

/**
 * threads: velocity indices;
 * blocks: spatial indices;
 * each thread works with a single matrix row;
*/
__global__ void Diff_set_sparse_matrix_rows()
{
    int*& rows = dev_dd_.A.rows;

    uint32_t iv = threadIdx.x;
    uint32_t ix = blockIdx.x;
    uint32_t ir = ix*dev_dd_.Nv + iv; // row id;
    if(ix == dev_dd_.Nx)
    {
        rows[dev_dd_.A.N] = dev_dd_.A.Nnz;
        // printf("rows[%d] = %d\n", dev_dd_.A.N, rows[dev_dd_.A.N]); 
        return;
    }

    uint32_t Nd = dev_dd_.N_discr;
    uint32_t Ndv_bo = dev_dd_.Ndv_bo;
    uint32_t Ndv = dev_dd_.Ndv;
    uint32_t Nx = dev_dd_.Nx;
    uint32_t Nv = dev_dd_.Nv;
    uint32_t Nvh = dev_dd_.Nv_h;
    uint32_t id_r1;

    // --- Upper half matrix ---
    uint32_t sh_diff = Ndv*(Nv-2) + 2*Ndv_bo;   // additional number of elements due d_v^2 g;
    uint32_t sh_ix0  = Nd * Nv + sh_diff;         // N of nonzero elements at ix = 0;
    uint32_t sh_bulk = ((Nd+1) * Nv + sh_diff) * (Nx-2); // N of nonzero elements at ix = 1,2,...Nx-2;

    uint32_t n_out = Nd + 1 + Ndv; // at ix = 0, at a single iv: 0 < iv < Nvh;
    uint32_t n_in = 2 + Ndv; // at ix = Nx-1, at a single iv: 0 < iv < Nvh;

    uint32_t sh_iv0_ix0 = Nd + 1 + Ndv_bo; // number of nonzero elements at ix = 0, iv = 0;
    uint32_t sh_vh_ix0 = sh_iv0_ix0 + n_out * (Nvh-1); // at ix = 0, at all iv = [0, Nvh);

    uint32_t sh_iv0_ixr = 2 + Ndv_bo; // number of nonzero elements at ix = Nx-1, iv = 0;
    uint32_t sh_vh_ixr = sh_iv0_ixr + n_in * (Nvh-1); // at ix = Nx-1, at all iv = [0, Nvh);
    if(ix == 0) // left spatial edge;
    {
        if(iv < Nvh)
            id_r1 = (iv == 0)? 0: sh_iv0_ix0 + n_out * (iv-1);
        else
            id_r1 = sh_vh_ix0 + n_in * (iv - Nvh);
    }
    else if(ix == Nx-1) // right spatial edge;
    {
        id_r1 = sh_ix0 + sh_bulk;
        if(iv < Nvh)
            id_r1 += (iv == 0)? 0: sh_iv0_ixr +  n_in*(iv-1);
        else
            id_r1 += sh_vh_ixr + n_out * (iv - Nvh);
    }
    else // bulk spatial points;
    {
        id_r1 = sh_ix0 + sh_diff * (ix-1) + (Nd+1) * (ir - Nv);
        id_r1 += (iv == 0)? 0: Ndv_bo + Ndv * (iv-1);
    }
    rows[ir] = id_r1;  

    // printf("rows[%d] = %d\n", ir, rows[ir]); 

    // --- Lower half matrix ---
    uint32_t sh_up_half = 2 * sh_ix0 + sh_bulk; // number of nonzero elements in the upper half matrix;

    id_r1 = sh_up_half + 2 * Nv * ix + iv;

    rows[Nv * Nx + ir] = (iv == 0) ? id_r1: id_r1 + Nv;

    // printf("rows[%d] = %d\n", Nv * Nx + ir, rows[Nv * Nx + ir]);
}


__device__ void left_vel_boundary(
    YCU ir, int*& columns, ycuComplex*& values, int32_t& kk, uint32_t& sh,
    YCU sh_change, YCD coef
){
    qreal idv = dev_dd_.diff/(dev_dd_.dv*dev_dd_.dv);
    qreal idv2 = 2.*idv;
    qreal idv5 = 5.*idv;
    qreal idv4 = 4.*idv;
    qreal w = dev_dd_.w;

    sh += sh_change;

    ++kk;
    columns[sh+kk]  = ir;
    values[sh+kk].x = (coef - idv2 + coef_plot)/(coef_plot+1);
    values[sh+kk].y = (w + coef_plot)/(coef_plot+1); // !!!

    ++kk;
    columns[sh+kk]  = ir + 1;
    values[sh+kk].x = (idv5 - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;

    ++kk;
    columns[sh+kk]  = ir + 2;
    values[sh+kk].x = (- idv4 - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;

    ++kk;
    columns[sh+kk]  = ir + 3;
    values[sh+kk].x = (idv - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;
}


__device__ void bulk_vel(
    YCU ir, int*& columns, ycuComplex*& values, int32_t& kk, uint32_t& sh,
    YCU sh_change, YCD coef
){
    qreal idv  = dev_dd_.diff/(dev_dd_.dv*dev_dd_.dv);
    qreal idv2 = 2.*idv;
    qreal w    = dev_dd_.w;

    sh += sh_change;

    ++kk;
    columns[sh+kk]  = ir - 1;
    values[sh+kk].x = (-idv - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;

    ++kk;
    columns[sh+kk]  = ir;
    values[sh+kk].x = (coef + idv2 + coef_plot)/(coef_plot+1);
    values[sh+kk].y = (w + coef_plot)/(coef_plot+1);; // !!!

    ++kk;
    columns[sh+kk]  = ir + 1;
    values[sh+kk].x = (-idv - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;
}


__device__ void right_vel_boundary(
    YCU ir, int*& columns, ycuComplex*& values, int32_t& kk, uint32_t& sh,
    YCU sh_change, YCD coef
){
    qreal idv = dev_dd_.diff/(dev_dd_.dv*dev_dd_.dv);
    qreal idv2 = 2.*idv;
    qreal idv5 = 5.*idv;
    qreal idv4 = 4.*idv;
    qreal w = dev_dd_.w;

    sh += sh_change;

    ++kk;
    columns[sh+kk]  = ir - 3;
    values[sh+kk].x = (idv - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;

    ++kk;
    columns[sh+kk]  = ir - 2;
    values[sh+kk].x = (- idv4 - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;

    ++kk;
    columns[sh+kk]  = ir - 1;
    values[sh+kk].x = (idv5 - coef_plot)/(coef_plot+1);
    values[sh+kk].y = 0;

    ++kk;
    columns[sh+kk]  = ir;
    values[sh+kk].x = (coef - idv2 + coef_plot)/(coef_plot+1);
    values[sh+kk].y = (w + coef_plot)/(coef_plot+1);; // !!!
}



/**
 * id threads: velocity indices;
 * id blocks: spacial indices;
*/
__global__ void Diff_set_matrix_values()
{
    ycuComplex*& values = dev_dd_.A.values;
    int*& columns = dev_dd_.A.columns;
    qreal*& FB = dev_dd_.FB;

    uint32_t iv = threadIdx.x;
    uint32_t ix = blockIdx.x;
    uint32_t ir = ix*dev_dd_.Nv + iv; // row id;

    uint32_t Nd = dev_dd_.N_discr;
    uint32_t Ndv_bo = dev_dd_.Ndv_bo;
    uint32_t Ndv = dev_dd_.Ndv;

    uint32_t Nx = dev_dd_.Nx;
    uint32_t Nv = dev_dd_.Nv;
    uint32_t Nvh = dev_dd_.Nv_h;
    qreal w = dev_dd_.w;

    uint32_t sh, sh_next;

    double v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv);

    double ih = v1/(2.*dev_dd_.h);
    double ih3 = 3.*ih;
    double ih4 = 4.*ih;
    
    uint32_t sh_var  = Nv * Nx;
    uint32_t sh_diff = Ndv*(Nv-2) + 2*Ndv_bo;
    uint32_t sh_ix0  = (Nd+3) * Nvh + sh_diff;           // N of nonzero elements at ix = 0;
    uint32_t sh_bulk = ((Nd+1) * Nv + sh_diff) * (Nx-2); // N of nonzero elements at ix = 1,2,...Nx-2;
    uint32_t sh_up_half = 2 * sh_ix0 + sh_bulk;

    uint32_t n_out = Nd + 1 + Ndv; // at ix = 0, at a single iv: 0 < iv < Nvh;
    uint32_t n_in = 2 + Ndv; // at ix = Nx-1, at a single iv: 0 < iv < Nvh;

    uint32_t sh_iv0_ix0 = Nd + 1 + Ndv_bo; // number of nonzero elements at ix = 0, iv = 0;
    uint32_t sh_vh_ix0 = sh_iv0_ix0 + n_out * (Nvh-1); // at ix = 0, at all iv = [0, Nvh);

    uint32_t sh_iv0_ixr = 2 + Ndv_bo; // number of nonzero elements at ix = Nx-1, iv = 0;
    uint32_t sh_vh_ixr = sh_iv0_ixr + n_in * (Nvh-1); // at ix = Nx-1, at all iv = [0, Nvh);

    

    // ******************************************************
    // *** matrix F ***
    // --- F: left spatial boundary ---
    int32_t kk = -1;
    if(ix == 0) 
    {
        if(iv < Nvh) // outgoing waves
        {
            sh = 0;
            if(iv == 0) // left velocity boundary:
                left_vel_boundary(
                    ir, columns, values, kk, sh, 
                    0, 
                    ih3
                );
            else // bulk velocity points:
                bulk_vel(
                    ir, columns, values, kk, sh, 
                    sh_iv0_ix0 + n_out * (iv-1), 
                    ih3
                );
            
            ++kk;
            columns[sh+kk]  = ir + Nv;
            values[sh+kk].x = (-ih4 + coef_plot)/(coef_plot+1);
            values[sh+kk].y =  0;
            
            ++kk;
            columns[sh+kk]  = ir + 2*Nv;
            values[sh+kk].x = (ih + coef_plot)/(coef_plot+1);
            values[sh+kk].y = 0;
        }
        else // incoming waves
        {
            sh = sh_vh_ix0 + n_in * (iv - Nvh);
            if(iv < Nv-1) // bulk velocity points:
                bulk_vel(
                    ir, columns, values, kk, sh, 
                    0, 
                    0.0
                );
            else // right velocity boundary:
                right_vel_boundary(
                    ir, columns, values, kk, sh, 
                    0, 
                    0.0
                );
        }
    }
    // --- F: right spatial boundary ---
    else if(ix == Nx-1) 
    {
        sh = sh_ix0 + sh_bulk; 
        if(iv < Nvh) // incoming waves
        {
            if(iv == 0) // left velocity boundary:
                left_vel_boundary(
                    ir, columns, values, kk, sh, 
                    0, 
                    0.0
                );
            else // bulk velocity points:
                bulk_vel(
                    ir, columns, values, kk, sh, 
                    sh_iv0_ixr +  n_in*(iv-1), 
                    0.0
                );
        }
        else // outgoing waves
        {
            sh += sh_vh_ixr + n_out * (iv - Nvh);

            ++kk;
            columns[sh+kk]  = ir - 2*Nv;
            values[sh+kk].x = (-ih + coef_plot)/(coef_plot+1);
            values[sh+kk].y = 0;

            ++kk;
            columns[sh+kk]  = ir - Nv;
            values[sh+kk].x = (ih4 + coef_plot)/(coef_plot+1);
            values[sh+kk].y =  0;

            if(iv < Nv-1) // bulk velocity points:
                bulk_vel(
                    ir, columns, values, kk, sh, 
                    0, 
                    - ih3
                );
            else // right velocity boundary:
                right_vel_boundary(
                    ir, columns, values, kk, sh, 
                    0, 
                    - ih3
                );
        }
    }
    // --- F: bulk spatial points ---
    else
    {
        sh = sh_ix0 + sh_diff * (ix-1) + (Nd+1) * (ir - Nv);
        sh += (iv == 0)? 0: Ndv_bo + Ndv * (iv-1);

        ++kk;
        columns[sh+kk]  = ir - Nv;
        values[sh+kk].x = (ih + coef_plot)/(coef_plot+1);
        values[sh+kk].y = 0.0;

        if(iv == 0) // left velocity boundary
            left_vel_boundary(
                ir, columns, values, kk, sh, 
                0, 
                0.0
            );
        if(iv == (Nv-1)) // right velocity boundary
            right_vel_boundary(
                ir, columns, values, kk, sh, 
                0, 
                0.0
            );
        if(iv > 0 && iv < Nv-1) // bulk velocity points:
            bulk_vel(
                ir, columns, values, kk, sh, 
                0, 
                0.0
            );
        
        ++kk;
        columns[sh+kk]  = ir + Nv;
        values[sh+kk].x = (- ih + coef_plot)/(coef_plot+1);
        values[sh+kk].y = 0.0;
    }

    sh_next = sh + kk + 1;

    // ******************************************************
    // *** matrix CE ***
    columns[sh_next]  = ix*Nv + sh_var; // column at iv = 0
    values[sh_next].x = (-v1*FB[ir] + coef_plot)/(coef_plot+1);
    values[sh_next].y = 0.0;

    // ******************************************************
    // *** matrix Cf ***
    sh = sh_up_half + 2 * Nv * ix + iv;

    columns[sh]  = ir;
    values[sh].x = (v1 + coef_plot)/(coef_plot+1);
    values[sh].y = 0.0;

    // ******************************************************
    // *** matrix S ***
    sh += Nv;

    columns[sh]  = sh_var + ir; 
    values[sh].x = 0.0;
    values[sh].y = (w + coef_plot)/(coef_plot+1);
}