#pragma once

#include "kin1d.cuh"
#include "../device/dev_outgoing_diff-SPACE.cuh"

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
        dd_.A.Nnz = 3 * dd_.Nv * (dd_.Nx - 2) + 5 * dd_.Nv; 

        // matrices CE, CF, S:
        dd_.A.Nnz += 3 * dd_.Nv * dd_.Nx;
    }

    void form_submatrix_F(YMatrix<ycomplex> &A, const qreal* Y)
    {
    }


    void form_submatrix_CE(YMatrix<ycomplex> &A, const qreal* FB)
    {   
    }


    void form_submatrix_CF(YMatrix<ycomplex> &A, const qreal* FB)
    {
    }


    void form_submatrix_S(YMatrix<ycomplex> &A)
    {
    }
};