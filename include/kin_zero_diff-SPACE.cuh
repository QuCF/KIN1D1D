#pragma once

#include "kin1d.cuh"
#include "../device/dev_zero_diff-SPACE.cuh"

class KWzero_diff : public KW
{
public:
    KWzero_diff(
        YCU nx, YCU nv, 
        YCU Lx, YCU Lv, 
        YCD Tref, YCD den_ref, 
        YCD wa, 
        YCD x0, YCD ds,
        YCS id_profile,
        YCD diff
    ) : KW(nx, nv, Lx, Lv, Tref, den_ref, wa, x0, ds, id_profile, diff)
    {
        compute_Nnz();
        print_init_parameters();
    }

    ~KWzero_diff(){};

    void form_sparse_matrix()
    {
        YTimer timer;

        printf("--- Creating matrices... ---\n");
        timer.Start();

        Diff_ZERO_set_sparse_matrix_rows<<<dd_.Nx+1, dd_.Nv>>>();
        Diff_ZERO_set_matrix_values<<<dd_.Nx, dd_.Nv>>>();
       
        cudaDeviceSynchronize();    
        timer.Stop();
        printf("--- Done: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }

protected:
    void compute_Nnz()
    {
        dd_.A.Nnz = 3 * dd_.Nv * (dd_.Nx - 2) + 5 * dd_.Nv; 

        // matrices CF, S:
        dd_.A.Nnz += 2 * dd_.Nv * dd_.Nx;

        // matrix CE:
        dd_.A.Nnz += dd_.Nv * (dd_.Nx-2) + dd_.Nv;
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