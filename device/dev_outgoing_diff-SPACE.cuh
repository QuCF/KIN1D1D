#pragma once

#include "dev_common.cuh"

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
        return;
    }

    uint32_t Nd = dev_dd_.N_discr;
    uint32_t Nx = dev_dd_.Nx;
    uint32_t Nv = dev_dd_.Nv;
    uint32_t Nvh = dev_dd_.Nv_h;
    uint32_t id_r1;

    // --- Upper half matrix ---
    uint32_t sh_ix0  = (Nd+1+3) * Nvh;         // N of nonzero elements at ix = 0;
    uint32_t sh_bulk = (Nd+1) * Nv * (Nx-2); // N of nonzero elements at ix = 1,2,...Nx-2;
    if(ix == 0) 
    {
        if(iv < Nvh)
            id_r1 = (Nd+1+1) * iv;
        else
            id_r1 = (Nd+1+1) * Nvh + 2*(iv - Nvh);
    }
    else if(ix == Nx-1) 
    {
        if(iv < Nvh)
            id_r1 = sh_ix0 + sh_bulk + 2*iv; 
        else
            id_r1 = sh_ix0 + sh_bulk + dev_dd_.Nv + (Nd+1+1)*(iv - Nvh);
    }
    else
    {
        id_r1 = sh_ix0 + (Nd+1) * (ir - Nv);
    }
    rows[ir] = id_r1;   

    // --- Lower half matrix ---
    uint32_t sh_up_half = 2 * sh_ix0 + sh_bulk; // number of nonzero elements in the upper half matrix;

    id_r1 = sh_up_half + 2 * Nv * ix + iv;

    rows[Nv * Nx + ir] = (iv == 0) ? id_r1: id_r1 + Nv;
}


/**
 * id threads: velocity indices;
 * id blocks: spacial indices;
*/
__global__ void Diff_set_matrix_values()
{
    cuqrealComplex*& values = dev_dd_.A.values;
    int*& columns = dev_dd_.A.columns;
    qreal*& FB = dev_dd_.FB;

    uint32_t iv = threadIdx.x;
    uint32_t ix = blockIdx.x;
    uint32_t ir = ix*dev_dd_.Nv + iv; // row id;
    uint32_t Nd = dev_dd_.N_discr;
    uint32_t Nx = dev_dd_.Nx;
    uint32_t Nv = dev_dd_.Nv;
    uint32_t Nvh = dev_dd_.Nv_h;
    qreal w = dev_dd_.w;

    uint32_t sh, sh_next;

    qreal ih = 1./(2.*dev_dd_.h);
    qreal ih3 = 3.*ih;
    qreal ih4 = 4.*ih;
    qreal v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv);

    qreal ihh = dev_dd_.diff/(dev_dd_.dv*dev_dd_.dv);
    qreal ihh2 = 2.*ihh;
    qreal ihh5 = 5.*ihh;
    qreal ihh4 = 4.*ihh;

    uint32_t sh_var  = Nv * Nx;

    uint32_t sh_ix0  = (Nd+1+3) * Nvh;         // N of nonzero elements at ix = 0;
    uint32_t sh_bulk = (Nd+1) * Nv * (Nx-2); // N of nonzero elements at ix = 1,2,...Nx-2;
    uint32_t sh_up_half = 2 * sh_ix0 + sh_bulk;

    // ******************************************************
    // *** matrix F ***
    // --- F: left boundary ---
    if(ix == 0) 
    {
        if(iv < Nvh)
        {
            sh = (Nd+1+1)*iv;

            columns[sh]  = ir;
            values[sh].x = ih3 * v1 - ihh2;
            values[sh].y = w;
            
            columns[sh+1]  = ir + Nv;
            values[sh+1].x = -ih4 * v1 + ihh5;
            values[sh+1].y =  0;
            
            columns[sh+2]  = ir + 2*Nv;
            values[sh+2].x = ih * v1 - ihh4;
            values[sh+2].y = 0;

            columns[sh+3]  = ir + 3*Nv;
            values[sh+3].x = ihh;
            values[sh+3].y = 0;

            sh_next = sh+4;
        }
        else
        {
            sh = (Nd+1+1) * Nvh + 2*(iv - Nvh);

            columns[sh]  = ir;
            values[sh].x = 0;
            values[sh].y = w;

            sh_next = sh+1;
        }
    }
    // --- F: right boundary ---
    else if(ix == Nx-1) 
    {
        if(iv < Nvh)
        {
            sh = sh_ix0 + sh_bulk + 2*iv; 
            
            columns[sh]  = ir;
            values[sh].x = 0;
            values[sh].y = w;

            sh_next = sh+1;
        }
        else
        {
            sh = sh_ix0 + sh_bulk + Nv + (Nd+1+1)*(iv - Nvh);

            columns[sh]  = ir - 3*Nv;
            values[sh].x = ihh;
            values[sh].y = 0;

            columns[sh+1]  = ir - 2*Nv;
            values[sh+1].x = -ih * v1 - ihh4;
            values[sh+1].y = 0;

            columns[sh+2]  = ir - Nv;
            values[sh+2].x = ih4 * v1 + ihh5;
            values[sh+2].y =  0;

            columns[sh+3]  = ir;
            values[sh+3].x = - ih3 * v1 - ihh2;
            values[sh+3].y = w;

            sh_next = sh+4;
        }
    }
    // --- F: bulk points ---
    else
    {
        sh = sh_ix0 + (Nd+1) * (ir - Nv); 

        columns[sh]  = ir - Nv;
        values[sh].x = ih * v1 - ihh;
        values[sh].y = 0.0;

        columns[sh+1]  = ir;
        values[sh+1].x = ihh2;
        values[sh+1].y = w;

        columns[sh+2]  = ir + Nv;
        values[sh+2].x = - ih * v1 - ihh;
        values[sh+2].y = 0.0;

        sh_next = sh+3;
    }


    // ******************************************************
    // *** matrix CE ***
    columns[sh_next]  = ix*Nv + sh_var; // column at iv = 0
    values[sh_next].x = -v1*FB[ir];
    values[sh_next].y = 0.0;


    // ******************************************************
    // *** matrix Cf ***
    sh = sh_up_half + 2 * Nv * ix + iv;

    columns[sh]  = ir;
    values[sh].x = v1;
    values[sh].y = 0.0;
    

    // ******************************************************
    // *** matrix S ***
    sh += Nv;

    columns[sh]  = sh_var + ir; 
    values[sh].x = 0.0;
    values[sh].y = w;
}