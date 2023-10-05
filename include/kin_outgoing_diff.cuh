#pragma once

#include "kin1d.cuh"
#include "../device/dev_outgoing_diff.cuh"

class KWout_diff : public KW
{
public:
    KWout_diff(
        YCU nx, YCU nv, 
        YCU Lx, YCU Lv, 
        YCD Tref, YCD den_ref, 
        YCD wa,
        YCD x0, YCD ds, 
        YCS id_profile,
        YCD diff
    ) : KW(nx, nv, Lx, Lv, Tref, den_ref, wa, x0, ds, id_profile, diff)
    {
        dd_.Ndv_bo = 3;
        dd_.Ndv = 2;

        compute_Nnz();
        print_init_parameters();
    }

    ~KWout_diff(){};

    void form_sparse_matrix()
    {
        // using namespace std::string_literals;
        YTimer timer;

        printf("--- Creating matrices... ---\n");
        timer.Start();

        Diff_set_sparse_matrix_rows<<<dd_.Nx+1, dd_.Nv>>>();
        Diff_set_matrix_values<<<dd_.Nx, dd_.Nv>>>();
       
        cudaDeviceSynchronize();    
        timer.Stop();
        printf("--- Done: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }

protected:
    void compute_Nnz()
    {
        uint32_t Ndv_bo = dd_.Ndv_bo;
        uint32_t Ndv = dd_.Ndv;

        dd_.A.Nnz = 3 * dd_.Nv * (dd_.Nx - 2) + 4 * dd_.Nv + (Ndv*(dd_.Nv-2) + 2*Ndv_bo) * dd_.Nx; 

        // matrices CE, CF, S:
        dd_.A.Nnz += 3 * dd_.Nv * dd_.Nx;
    }

    void form_submatrix_F(YMatrix<ycomplex> &A, const double* Y)
    {
    }


    void form_submatrix_CE(YMatrix<ycomplex> &A, const double* FB)
    {   
    }


    void form_submatrix_CF(YMatrix<ycomplex> &A, const double* FB)
    {
    }


    void form_submatrix_S(YMatrix<ycomplex> &A)
    {
    }
};