#pragma once

#include "kin1d.cuh"
#include "../device/dev_outgoing.cuh"


class KWout_form1 : public KW
{
public:
    KWout_form1(
        YCU nx, YCU nv, 
        YCU Lx, YCU Lv, 
        YCD Tref, YCD den_ref, 
        YCD wa, 
        YCS id_profile
    ) : KW(nx, nv, Lx, Lv, Tref, den_ref, wa, id_profile)
    {
        compute_Nnz();
        print_init_parameters();
    }

    ~KWout_form1(){};

    void form_sparse_matrix()
    {
        // using namespace std::string_literals;
        YTimer timer;

        printf("--- Creating matrices... ---\n");
        timer.Start();

        set_sparse_matrix_rows<<<dd_.Nx+1, dd_.Nv>>>();
        set_matrix_values<<<dd_.Nx, dd_.Nv>>>();
       
        cudaDeviceSynchronize();    
        timer.Stop();
        printf("--- Done: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }

protected:
    void compute_Nnz()
    {
        dd_.A.Nnz = 3 * dd_.Nv * (dd_.Nx - 2) + 4 * dd_.Nv + 3 * dd_.Nv * dd_.Nx;
    }

    void form_submatrix_F(YMatrix<ycomplex> &A, const double* Y)
    {
        using namespace std::complex_literals;

        uint32_t idx;
        uint64_t sh_r;
        ycomplex wi = 1i*dd_.w;
        double ih = 1./(2.*dd_.h);
        double ih4 = 4.*ih;
        double ih3 = 3.*ih;

        // left edge:
        idx  = 0;
        sh_r = idx * dd_.Nv;
        for(uint32_t iv = 0; iv < dd_.Nv_h; iv++)
        {
            A(iv, iv)         = wi + ih3 * v_[iv];
            A(iv, dd_.Nv + iv)   = -ih4 * v_[iv];
            A(iv, 2*dd_.Nv + iv) =   ih * v_[iv];
        }
        for(uint32_t iv = dd_.Nv_h; iv < dd_.Nv; iv++)
            A(iv, iv) = wi;

        // right edge:
        idx  = dd_.Nx - 1;
        sh_r = idx * dd_.Nv;
        for(uint32_t iv = 0; iv < dd_.Nv_h; iv++)
            A(sh_r + iv, sh_r + iv) = wi;
        for(uint32_t iv = dd_.Nv_h; iv < dd_.Nv; iv++)
        {
            A(sh_r + iv, sh_r - 2*dd_.Nv + iv) = -ih * v_[iv];
            A(sh_r + iv, sh_r - dd_.Nv + iv)   = ih4 * v_[iv];
            A(sh_r + iv, sh_r + iv)       =  wi - ih3 * v_[iv];
        }

        // bulk:
        for(uint ix = 1; ix < (dd_.Nx-1); ix++)
        {
            sh_r = ix * dd_.Nv;
            for(uint32_t iv = 0; iv < dd_.Nv; iv++)
            {
                A(sh_r + iv, sh_r - dd_.Nv + iv) =  ih * v_[iv];
                A(sh_r + iv, sh_r + iv)           =  wi;
                A(sh_r + iv, sh_r + dd_.Nv + iv) = -ih * v_[iv];
            }
        }
    }


    void form_submatrix_CE(YMatrix<ycomplex> &A, const double* FB)
    {   
        using namespace std::complex_literals;
        uint64_t sh_r;
        uint64_t sh_var = dd_.Nx*dd_.Nv;

        for(uint ix = 0; ix < dd_.Nx; ix++)
        {
            sh_r = ix * dd_.Nv;
            for(uint32_t iv = 0; iv < dd_.Nv; iv++)
                A(sh_r + iv, sh_var + sh_r) = - v_[iv] * FB[sh_r + iv];
        }
    }


    void form_submatrix_CF(YMatrix<ycomplex> &A, const double* FB)
    {
        using namespace std::complex_literals;
        uint64_t sh_r;
        uint64_t sh_var = dd_.Nx*dd_.Nv;

        for(uint ix = 0; ix < dd_.Nx; ix++)
        {
            sh_r = ix * dd_.Nv;
            for(uint32_t ivc = 0; ivc < dd_.Nv; ivc++)
                A(sh_var + sh_r, sh_r + ivc) =  v_[ivc];
        }
    }


    void form_submatrix_S(YMatrix<ycomplex> &A)
    {
        using namespace std::complex_literals;
        ycomplex wi = 1i*dd_.w;
        uint64_t sh_r;
        uint64_t sh_var = dd_.Nx * dd_.Nv;
        for(uint ix = 0; ix < dd_.Nx; ix++)
        {
            // diagonal elements added to iv != 0 to guarantee the inversibility of the matrix:
            sh_r = ix * dd_.Nv;
            for(uint32_t iv = 0; iv < dd_.Nv; iv++)
                A(sh_var + sh_r + iv, sh_var + sh_r + iv) = wi;
        }
    }
};