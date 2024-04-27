#pragma once

#include "LA.cuh"
#include "../device/dev_profiles_rhs.cuh"

// -------------------------------------------------------------------
// --- Linear kinetic waves in a 1-D electron plasma (skin effect) ---
// Standard formulation: formulation 2;
// -------------------------------------------------------------------
class KW
{
protected:
    YMatrix<ycomplex> L_; // Preconditioner 
    KDATA dd_; // plasma parameters
    
    YMatrix<ycomplex> A_dense_; // dense version of the system matrix;

    std::string id_profile_; // background profiles;

    qreal coef_superposition_;

    std::shared_ptr<qreal[]> x_; // normalized to kx spatial grid;
    std::shared_ptr<qreal[]> v_; // normalized velocity grid;

    std::shared_ptr<qreal[]> rx_; // normalized to 1 spatial grid;
    std::shared_ptr<qreal[]> rv_; // normalized to 1 velocity grid;

    std::shared_ptr<qreal[]> T_;   // normalized temperature profile;
    std::shared_ptr<qreal[]> den_; // normalized density profile;

    qreal Tref_;    // reference electron temperature (erg);
    qreal den_ref_; // reference density profile (cm-3);
    qreal wp_;  // reference plasma frequency (1/s);
    qreal ld_;  // reference Debye length (cm);
    qreal vth_; // reference thermal speed (cm/s);

    std::string gl_path_out_;   // path to the output files;
    std::string hdf5_name_out_; // name of the output .hdf5 file;

    YHDF5 f_;

public:

    /**
     * nx = log2 of spatial grid size.
     * nv = log2 of velocity grid size.
     * Lx - number of Debye lengths in the spatial grid (defines the length of the grid).
     * Lv - number of thermal velocities in the velocity grid.
     * Tref - reference electron temperature (erg).
     * den_ref - reference density profile (cm-3).
     * wa - normalized (to wp_) antenna frequency.
     * id_profile: "flat", "linear", "gauss", "exp": type of the background profiles; 
    */
    KW(
        YCU nx, YCU nv, 
        YCU Lx, YCU Lv, 
        YCD Tref, YCD den_ref, 
        YCD wa, 
        YCD x0, YCD ds,
        YCS id_profile,
        YCD diff = 0.0,
        YCS folder_to_save="../../results/KIN1D1D-results/"
    ) : Tref_(Tref), 
        den_ref_(den_ref),
        gl_path_out_(folder_to_save + "/")
    {
        using namespace std;
        Constants cc;
        
        dd_.set_to_zero();
        dd_.w = wa;
        dd_.diff = diff;

        dd_.x0 = x0;
        dd_.ds = ds;

        dd_.xmax = Lx;  // for the normalized to ld_  x-grid;
        dd_.vmax = Lv; // for the normalized to vth_ v-grid;

        dd_.nx = nx; dd_.Nx = 1 << dd_.nx;
        dd_.nv = nv; dd_.Nv = 1 << dd_.nv;
        
        dd_.Nv_h = 1 << (dd_.nv - 1);

        dd_.Nvars = 2;
        dd_.A.N = dd_.Nx * dd_.Nv * dd_.Nvars;

        dd_.N_discr = 3;
        dd_.Ndv_bo = 0;
        dd_.Ndv = 0;

        id_profile_ = id_profile;
        std::transform(
            id_profile_.begin(), id_profile_.end(), 
            id_profile_.begin(), 
            ::tolower
        );

        // set the output-file name:
        set_output_name();

        // plasma parameters:
        qreal temp = 4*cc.pi_*pow(cc.e_,2)*den_ref_;
        wp_  = sqrt(temp/cc.me_);
        ld_  = sqrt(Tref_/temp);
        vth_ = ld_ * wp_;

        // --- spatial and velocity grids ---
        qreal h1, dv1; // steps for the normalized to 1 grids;

        dd_.h  = dd_.xmax / (dd_.Nx - 1);
        dd_.dv = 2.*dd_.vmax / (dd_.Nv - 1); // negative and positive velocities;

        h1  = dd_.h / dd_.xmax;
        dv1 = dd_.dv / dd_.vmax;

        x_  = shared_ptr<qreal[]>(new qreal[dd_.Nx]);
        rx_ = shared_ptr<qreal[]>(new qreal[dd_.Nx]);
        for(uint32_t ii = 0; ii < dd_.Nx; ii++){
            x_[ii]  = get_x1(dd_.h, ii);
            rx_[ii] = get_x1(   h1, ii);
        }

        v_  = shared_ptr<qreal[]>(new qreal[dd_.Nv]);
        rv_ = shared_ptr<qreal[]>(new qreal[dd_.Nv]);
        for(uint32_t ii = 0; ii < dd_.Nv; ii++){
            v_[ii]  = get_v1(dd_.vmax, dd_.dv, ii);
            rv_[ii] = get_v1(       1,    dv1, ii);
        }

        // --- Create the HDF5 file ---
        f_.create(gl_path_out_ + hdf5_name_out_);
        f_.add_group("basic");
        f_.add_group("parameters");
        f_.add_group("grids");
        f_.add_group("profiles");
        f_.add_group("matrices");
        f_.add_group("result");

        // date of a simulation:
        string str_date_time;
        YMIX::get_current_date_time(str_date_time);
        f_.add_scalar(str_date_time, "date-of-simulation", "basic");
        f_.add_scalar(filesystem::current_path(), "launch-path", "basic");

        // parameters:
        f_.add_scalar(dd_.w, "w", "parameters");
        f_.add_scalar(dd_.diff, "diff", "parameters");
        f_.add_scalar(dd_.x0, "source_x0", "parameters");
        f_.add_scalar(dd_.ds, "source_dx", "parameters");

        // save the grids:
        f_.add_array(x_.get(), dd_.Nx, std::string("x"), "grids");
        f_.add_array(v_.get(), dd_.Nv, std::string("v"), "grids");
        f_.add_array(rx_.get(), dd_.Nx, std::string("rx"), "grids");
        f_.add_array(rv_.get(), dd_.Nv, std::string("rv"), "grids");

        // close the file:
        f_.close();
    }


    ~KW()
    {
        clean_device();
    }


    void clean_device()
    {
        dd_.A.clean();
        CUDA_CHECK(cudaFree(dd_.b));
        CUDA_CHECK(cudaFree(dd_.psi));
        CUDA_CHECK(cudaFree(dd_.FB));
        CUDA_CHECK(cudaFree(dd_.Y));
    }


    void init_device()
    {
        printf("--- GPU initialization... ---\n");
        cudaSetDevice(0);

        printf("-> allocating matrices...\n");

        dd_.A.allocate();

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.b),  
            sizeof(ycuComplex) * dd_.A.N
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.psi),  
            sizeof(ycuComplex) * dd_.A.N
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.FB),  
            sizeof(qreal) * dd_.Nx * dd_.Nv
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.Y),  
            sizeof(qreal) * dd_.Nx * dd_.Nv
        ));

        printf("-> initializing vectors...\n");
        CUDA_CHECK(cudaMemset(dd_.b, 0, sizeof(ycuComplex) * dd_.A.N));
        CUDA_CHECK(cudaMemset(dd_.psi, 0, sizeof(ycuComplex) * dd_.A.N));

        printf("-> copying constant parameters...\n");
        CUDA_CHECK(cudaMemcpyToSymbol(dev_dd_, &dd_, sizeof(KDATA)));
    }


    void set_background_profiles()
    {
        bool flag_found = false;
        printf("--- Setting background profiles... ---\n");
        YTimer timer;
        timer.Start();

        T_  = std::shared_ptr<qreal[]>(new qreal[dd_.Nx]);
        den_ = std::shared_ptr<qreal[]>(new qreal[dd_.Nx]);
        if(id_profile_.compare("flat") == 0)
        {
            printf("-> Forming flat profiles...\n");
            for(uint32_t ii = 0; ii < dd_.Nx; ii++){
                T_[ii] = 1.0;
                den_[ii] = 1.0;
            }
            flag_found = true;
        }
        if(id_profile_.compare("tanh2") == 0)
        {
            // qreal a = 40;
            // qreal x0_left = 0.05;
            // qreal x0_right = 0.95;

            qreal a = 80;
            qreal x0_left = 0.05;
            qreal x0_right = 0.95;

            qreal temp;
            printf("-> Forming tanh-tanh profiles...\n");
            for(uint32_t ii = 0; ii < dd_.Nx; ii++)
            {
                temp = 0.5 * ( tanh(a*(rx_[ii]-x0_left)) - tanh(a*(rx_[ii]-x0_right)) );
                T_[ii]   = temp;
                den_[ii] = temp;
            }
            flag_found = true;
        }
        if(id_profile_.compare("exp") == 0)
        {
            qreal coef = 0.005;
            printf("-> Forming exp profiles...\n");
            for(uint32_t ii = 0; ii < dd_.Nx; ii++){
                T_[ii]   = exp(coef * (x_[ii]-dd_.x0));
                den_[ii] = exp(coef * (x_[ii]-dd_.x0));
            }
            flag_found = true;
        }
        // if(id_profile_.compare("gauss") == 0)
        // {
        //     qreal sigma_T = 0.2;
        //     qreal sigma_n = 0.1;
        //     qreal sigma_T2 = 2.*sigma_T*sigma_T;
        //     qreal sigma_n2 = 2.*sigma_n*sigma_n;
        //     printf("-> Forming Gaussian profiles...\n");
        //     for(uint32_t ii = 0; ii < dd_.Nx; ii++){
        //         T_[ii]   = exp(-pow(rx_[ii] - 0.50,2)/sigma_T2);
        //         den_[ii] = exp(-pow(rx_[ii] - 0.50,2)/sigma_n2);
        //     }
        //     flag_found = true;
        // }
        if(flag_found)
            build_background_distribution();
        else
            std::cerr << "\t>>> Error: No background profiles found." << std::endl;  
        timer.Stop();
        printf("\tDone: total elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void form_rhs()
    {
        printf("--- Setting the right-hand-side vector... ---\n");
        YTimer timer;
        timer.Start();

        uint32_t N_threads, N_blocks;
        if(nq_THREADS < dd_.nx)
        {
            N_threads = 1 << nq_THREADS;
            N_blocks = 1 << (dd_.nx - nq_THREADS);
        }
        else
        {
            N_threads = 1 << dd_.nx;
            N_blocks = 1;
        }

        init_rhs_gauss<<<N_blocks, N_threads>>>();
       
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void form_dense_matrix()
    {
        YTimer timer;
        uint32_t NxNv = dd_.Nx*dd_.Nv;
        auto size_v = sizeof(qreal) * NxNv;
        qreal* F;
        qreal* Y;
        
        printf("--- Creating a dense matrix... ---\n");
        timer.Start();

        // --- Get background profiles ---
        F = new qreal[NxNv];
        Y = new qreal[NxNv];
        CUDA_CHECK(cudaMemcpy(F, dd_.FB, size_v, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Y, dd_.Y, size_v, cudaMemcpyDeviceToHost));

        // --- Form a dense matrix on the host ---
        A_dense_.zeros(dd_.A.N, dd_.A.N);

        // form_submatrix_F(A_dense_, Y);
        // form_submatrix_S(A_dense_);
        // form_submatrix_CE(A_dense_, F);
        // form_submatrix_CF(A_dense_, F);

        timer.Stop();
        printf("   Done: elapsed time [s]: %0.3e\n", timer.get_dur_s());

        // --- Count non-zero elements ---
        printf("Counting nonzero elements...\n");
        timer.Start();

        A_dense_.count_nz();

        printf("analytically computed Nnz: %d\n", dd_.A.Nnz);
        printf("numerically computed Nnz: %lu\n", A_dense_.get_Nnz());
        if(A_dense_.get_Nnz() > dd_.A.Nnz)
            std::cerr << "Error: Numerical Nnz > analytical Nnz." << std::endl;

        timer.Stop();
        printf("   Done: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());

        // --- Create a non-sparse matrix (still on the host) ---
        printf("Forming non-sparse matrix...\n");
        timer.Start();

        A_dense_.form_sparse_format();

        timer.Stop();
        printf("   Done: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());

        // --- Copy the host-based sparse matrix to the device-based sparse matrix ---
        CUDA_CHECK(cudaMemcpy(
            dd_.A.values,  A_dense_.get_nz_values(),  
            A_dense_.get_size_nz_values(),  cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            dd_.A.columns, A_dense_.get_nz_columns(), 
            A_dense_.get_size_nz_columns(), cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            dd_.A.rows,    A_dense_.get_nz_rows(),    
            A_dense_.get_size_nz_rows(),    cudaMemcpyHostToDevice
        ));

        // clear memory:
        delete [] F;
        delete [] Y;
    }


    void solve_system()
    {
        solve_sparse_system();
    }


    void save_xv_background_profs_to_hdf5()
    {
        printf("--- Saving the (x,v) background profiles to %s... ---", hdf5_name_out_.c_str()); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        uint32_t NxNv = dd_.Nx*dd_.Nv;
        auto size_v = sizeof(qreal) * NxNv;

        qreal* F = new qreal[NxNv];
        qreal* Y = new qreal[NxNv];
        
        // transfer data from GPU to host:
        CUDA_CHECK(cudaMemcpy(F, dd_.FB, size_v, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Y, dd_.Y, size_v, cudaMemcpyDeviceToHost));

        // save data in the .hdf5 file:
        f_.open_w();
        f_.add_array(F, NxNv, std::string("F"), "profiles");
        f_.add_array(Y, NxNv, std::string("Y"), "profiles");
        f_.close();

        // remove temporary arrays;
        delete [] F;
        delete [] Y;
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void save_rhs_to_hdf5()
    {
        printf("--- Saving the right-hand-side vector to %s... ---", hdf5_name_out_.c_str()); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        auto size_v = sizeof(ycomplex) * dd_.A.N;
        ycomplex* b = new ycomplex[dd_.A.N];

        CUDA_CHECK(cudaMemcpy(b, dd_.b, size_v, cudaMemcpyDeviceToHost));

        f_.open_w();
        f_.add_array(b, dd_.A.N, std::string("b"), "profiles");
        f_.close();

        delete [] b;
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void save_matrices()
    {
        printf("--- Saving the matrices...---"); std::cout << std::endl;
        YTimer timer;
        timer.Start();

        save_one_matrix(dd_.A,   "A");

        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void save_result()
    {
        printf("--- Saving the result to %s... ---", hdf5_name_out_.c_str()); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        auto size_v = sizeof(ycomplex) * dd_.A.N;
        ycomplex* psi_host = new ycomplex[dd_.A.N];

        CUDA_CHECK(cudaMemcpy(psi_host, dd_.psi, size_v, cudaMemcpyDeviceToHost));

        f_.open_w();
        f_.add_array(psi_host, dd_.A.N, std::string("psi"), "result");
        f_.close();

        delete [] psi_host;
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void print_grids()
    {
        printf("\n\n1/(2h) = %0.3e\n", 1./(2.*dd_.h));
        printf("--- x-grid ---\n");
        for(uint32_t ix = 0; ix < dd_.Nx; ix++)
        {
            printf("%0.3e  ", x_[ix]);
            if(ix != 0 && ix%18 == 0) printf("\n");
        }

        printf("\n--- v-grid ---\n");
        for(uint32_t iv = 0; iv < dd_.Nv; iv++)
        {
            printf("%0.3e  ", v_[iv]);
            if(iv != 0 && iv%18 == 0) printf("\n");
        }
        std::cout << "\n" << std::endl;
    }


    void print_sparse_matrix()
    {
        // YMatrix<ycomplex> A;
        // dd_.A.form_dense_matrix(A);

        // // qreal ih = 1./(2.*dd_.h);
        // // qreal ih3 = 3.*ih;
        // // qreal ih4 = 4.*ih;
        // // printf("sigma*vmax:  %0.3e\n",   ih*dd_.vmax);
        // // printf("4sigma*vmax: %0.3e\n", ih4*dd_.vmax);
        // // printf("\n");

        // uint32_t idx;
        // uint64_t sh_r;
        // uint64_t sh_var = dd_.Nx*dd_.Nv;
        // printf("\n");

        // // printf("\n --- matrix F_EL ---\n");
        // // A.print(0, dd_.Nv, 0, dd_.Nv);

        // // printf("\n --- matrix F_EL_0 ---\n");
        // // A.print(0, dd_.Nv, dd_.Nv, 2*dd_.Nv);

        // // printf("\n --- matrix F_EL_1 ---\n");
        // // A.print(0, dd_.Nv, 2*dd_.Nv, 3*dd_.Nv);



        // // sh_r = (dd_.Nx-1)*dd_.Nv;
        // // printf("\n --- matrix F_ER_1 ---\n");
        // // A.print(sh_r, sh_r + dd_.Nv, sh_r - 2*dd_.Nv, sh_r - dd_.Nv);

        // // sh_r = (dd_.Nx-1)*dd_.Nv;
        // // printf("\n --- matrix F_ER_0 ---\n");
        // // A.print(sh_r, sh_r + dd_.Nv, sh_r - dd_.Nv, sh_r);

        // // sh_r = (dd_.Nx-1)*dd_.Nv;
        // // printf("\n --- matrix F_ER ---\n");
        // // A.print(sh_r, sh_r + dd_.Nv, sh_r, sh_r + dd_.Nv);



        // idx = 1;
        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix left F_BD ---\n");
        // A.print(sh_r, sh_r + dd_.Nv, sh_r - dd_.Nv, sh_r);

        // idx = 1;
        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix F_B ---\n");
        // A.print(sh_r, sh_r + dd_.Nv, sh_r, sh_r + dd_.Nv);

        // idx = 1;
        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix right F_BD ---\n");
        // A.print(sh_r, sh_r + dd_.Nv, sh_r + dd_.Nv, sh_r + 2*dd_.Nv);




        // // idx = 7;
        // // sh_r = idx * dd_.Nv;
        // // printf("\n --- matrix CE ---\n");
        // // A.print(sh_r, sh_r + dd_.Nv, sh_var + sh_r, sh_var + sh_r + dd_.Nv);



        // // idx = 0;
        // // sh_r = idx * dd_.Nv;
        // // printf("\n --- matrix CF ---\n");
        // // A.print(
        // //     sh_var + sh_r, sh_var + sh_r + dd_.Nv, 
        // //     sh_r, sh_r + dd_.Nv
        // // );
    


        // // idx = 7;
        // // sh_r = idx * dd_.Nv;
        // // printf("\n --- matrix S ---\n");
        // // A.print(
        // //     sh_var + sh_r, sh_var + sh_r + dd_.Nv, 
        // //     sh_var + sh_r, sh_var + sh_r + dd_.Nv
        // // );
    }


    void print_dense_matrix()
    {
        uint32_t idx;
        uint64_t sh_r;
        uint64_t sh_var = dd_.Nx*dd_.Nv;

        idx = 1;
        // idx = dd_.Nx - 1;

        printf("\n");

        // printf("\n --- matrix F_EL ---\n");
        // A_dense_.print(0, dd_.Nv, 0, dd_.Nv);

        // printf("\n --- matrix F_EL_0 ---\n");
        // A_dense_.print(0, dd_.Nv, dd_.Nv, 2*dd_.Nv);

        // printf("\n --- matrix F_EL_1 ---\n");
        // A_dense_.print(0, dd_.Nv, 2*dd_.Nv, 3*dd_.Nv);



        // sh_r = (dd_.Nx-1)*dd_.Nv;
        // printf("\n --- matrix F_ER_1 ---\n");
        // A_dense_.print(sh_r, sh_r + dd_.Nv, sh_r - 2*dd_.Nv, sh_r - dd_.Nv);

        // sh_r = (dd_.Nx-1)*dd_.Nv;
        // printf("\n --- matrix F_ER_0 ---\n");
        // A_dense_.print(sh_r, sh_r + dd_.Nv, sh_r - dd_.Nv, sh_r);

        // sh_r = (dd_.Nx-1)*dd_.Nv;
        // printf("\n --- matrix F_ER ---\n");
        // A_dense_.print(sh_r, sh_r + dd_.Nv, sh_r, sh_r + dd_.Nv);



        
        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix left F_BD ---\n");
        // A_dense_.print(sh_r, sh_r + dd_.Nv, sh_r - dd_.Nv, sh_r);

        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix F_B ---\n");
        // A_dense_.print(sh_r, sh_r + dd_.Nv, sh_r, sh_r + dd_.Nv);

        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix right F_BD ---\n");
        // A_dense_.print(sh_r, sh_r + dd_.Nv, sh_r + dd_.Nv, sh_r + 2*dd_.Nv);




        sh_r = idx * dd_.Nv;
        printf("\n --- matrix CE ---\n");
        A_dense_.print(sh_r, sh_r + dd_.Nv, sh_var + sh_r, sh_var + sh_r + dd_.Nv);


        sh_r = idx * dd_.Nv;
        printf("\n --- matrix CF ---\n");
        A_dense_.print(
            sh_var + sh_r, sh_var + sh_r + dd_.Nv, 
            sh_r, sh_r + dd_.Nv
        );


        // sh_r = idx * dd_.Nv;
        // printf("\n --- matrix S ---\n");
        // A_dense_.print(
        //     sh_var + sh_r, sh_var + sh_r + dd_.Nv, 
        //     sh_var + sh_r, sh_var + sh_r + dd_.Nv
        // );
    }


    void recheck_result()
    {
        qreal max_abs_error = 0.0;
        qreal max_rel_error = 0.0;

        printf("--- Check the solver... ---"); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        // maximum absolute error:
        ycuComplex* dev_out;
        LA::recheck_linear_solver(dd_.A, dd_.psi, dd_.b, dev_out);
        ycomplex* out = new ycomplex[dd_.A.N];
        CUDA_CHECK(cudaMemcpy(out, dev_out, sizeof(ycomplex) * dd_.A.N, cudaMemcpyDeviceToHost));

        // RHS vector:
        ycomplex* b = new ycomplex[dd_.A.N];
        CUDA_CHECK(cudaMemcpy(b, dd_.b, sizeof(ycomplex) * dd_.A.N, cudaMemcpyDeviceToHost));

        // relative errors:
        qreal* rel_errors = new qreal[dd_.A.N];
        for(uint32_t ii = 0; ii < dd_.A.N; ii++)
        {
            auto err = std::abs(out[ii]);
            auto v1 = std::abs(b[ii]);
            if(err < ZERO_ERROR) 
                rel_errors[ii] = 0.0;
            else 
                rel_errors[ii] = err / v1;
        }
            
        
        for(uint32_t ii = 0; ii < dd_.A.N; ii++)
        {
            auto out_v1 = std::abs(out[ii]);
            if(max_abs_error < out_v1) max_abs_error = out_v1;

            auto rel_v1 = rel_errors[ii];
            if(max_rel_error < rel_v1) max_rel_error = rel_v1;
        }
        printf("\tMax. abs. error: %0.3e\n", max_abs_error);
        printf("\tMax. rel. error: %0.3e\n", max_rel_error);

        // // print the errors:
        // printf("*** abs errors ***");
        // for(uint32_t ii = 0; ii < dd_.A.N; ii++)
        // {
        //     if(ii%10 == 0) printf("\n");
        //     printf("%0.3e ", std::abs(out[ii]));
        // }
        // printf("\n***\n");

        // max_v = 0.0;
        // for(uint32_t ii = 0; ii < dd_.A.N; ii++)
        // {
        //     auto v1 = std::abs(b[ii]);
        //     if(max_v < v1) max_v = v1;
        // }
        // printf("\tMax. abs. value in the RHS vector: %0.3e\n", max_v);

        // // print the RHS vector:
        // printf("\n*** RHS ***");
        // for(uint32_t ii = 0; ii < dd_.A.N; ii++)
        // {
        //     if(ii%10 == 0) printf("\n");
        //     printf("%0.3e ", std::abs(b[ii]));
        // }
        // printf("\n***\n");

        delete [] out;
        delete [] b;
        delete [] rel_errors;
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void print_rows()
    {
        uint32_t ir;
        uint32_t Nr = dd_.A.N+1;
        auto size_rows = sizeof(int) * Nr;
        int* rows_host = new int[Nr];

        CUDA_CHECK(cudaMemcpy(rows_host, dd_.A.rows, size_rows, cudaMemcpyDeviceToHost));

        printf("******************************************\n");
        printf("*** upper half matrix ***\n");
        for(uint32_t ix = 0; ix < dd_.Nx; ix++)
        {
            printf("--- ix = %d ---\n", ix);
            for(uint32_t iv = 0; iv < dd_.Nv; iv++)
            {
                ir = ix * dd_.Nv + iv;
                printf("  iv = %d, ir = %d: row id = %d\n", iv, ir, rows_host[ir]);
            }
        }

        printf("\n******************************************\n");
        printf("*** lower half matrix ***\n");
        for(uint32_t ix = 0; ix < dd_.Nx; ix++)
        {
            printf("--- ix = %d ---\n", ix);
            for(uint32_t iv = 0; iv < dd_.Nv; iv++)
            {
                ir = dd_.Nv*dd_.Nx + ix * dd_.Nv + iv;
                printf("  iv = %d, ir = %d: row id = %d\n", iv, ir, rows_host[ir]);
            }
        }
        delete [] rows_host;
    }


protected:

    void solve_sparse_system()
    {
        printf("--- Solving the system using the sparse solver... ---\n");
        YTimer timer;
        timer.Start();

        LA::solve_sparse_system(dd_.A, dd_.b, dd_.psi);

        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: total elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void build_background_distribution()
    {
        printf("-> Building (x,v) background distributions...\n");

        qreal *dT = new qreal[dd_.Nx]; 
        qreal *dn = new qreal[dd_.Nx]; 
        
        // find derivatives:
        YMATH::find_der(T_.get(),   dd_.h, dd_.Nx, dT);
        YMATH::find_der(den_.get(), dd_.h, dd_.Nx, dn);
        
        // save the x-profiles:
        f_.open_w();
        f_.add_array(T_.get(),   dd_.Nx, "T", "profiles");
        f_.add_array(den_.get(), dd_.Nx, "n", "profiles");
        f_.add_array(dT, dd_.Nx, "der-T", "profiles");
        f_.add_array(dn, dd_.Nx, "der-n", "profiles");
        f_.close();

        // save the x-background profiles on the GPU:
        qreal* dev_T;
        qreal* dev_n;
        qreal* dev_dT;
        qreal* dev_dn;
        auto size_v = sizeof(qreal) * dd_.Nx;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_T), size_v));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_n), size_v));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_dT), size_v));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_dn), size_v));

        CUDA_CHECK(cudaMemcpy(dev_T,   T_.get(), size_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_n, den_.get(), size_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_dT,      dT, size_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_dn,      dn, size_v, cudaMemcpyHostToDevice));

        // initialize the (x,v)-profiles on the GPU:
        init_background_distribution(dev_T, dev_n, dev_dT, dev_dn);

        // remove the x-background profiles from the GPU:
        CUDA_CHECK(cudaFree(dev_T));
        CUDA_CHECK(cudaFree(dev_n));
        CUDA_CHECK(cudaFree(dev_dT)); 
        CUDA_CHECK(cudaFree(dev_dn)); 
        delete [] dT;
        delete [] dn;

        cudaDeviceSynchronize();
    }


    void save_one_matrix(SpMatrixC& A, YCS name)
    {
        using namespace std;

        uint32_t Nr = A.N+1;

        ycomplex* values_host = new ycomplex[A.Nnz];
        int* columns_host     = new int[A.Nnz];
        int* rows_host        = new int[Nr];

        auto size_complex = sizeof(ycomplex) * A.Nnz;
        auto size_columns = sizeof(int) * A.Nnz;
        auto size_rows    = sizeof(int) * Nr;

        CUDA_CHECK(cudaMemcpy(values_host,  A.values,  size_complex, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(columns_host, A.columns, size_columns, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rows_host,    A.rows,    size_rows,    cudaMemcpyDeviceToHost));

        f_.open_w();
        f_.add_scalar(A.N,   name + "-N"s,   "matrices"s);
        f_.add_scalar(A.Nnz, name + "-Nnz"s, "matrices"s);
        f_.add_array(values_host,  A.Nnz, name + "-values"s,  "matrices"s);
        f_.add_array(columns_host, A.Nnz, name + "-columns"s, "matrices"s);
        f_.add_array(rows_host,       Nr, name + "-rows"s,    "matrices"s);
        f_.close();

        delete [] values_host;
        delete [] columns_host;
        delete [] rows_host;
    }


    void print_init_parameters()
    {
        Constants cc;

        printf("--------------------------------------------------\n");
        printf("------------------ Matrix parameters -------------\n");
        printf("--------------------------------------------------\n");
        printf("\tNx = %d\n", dd_.Nx);
        printf("\tNv = %d\n", dd_.Nv);
        printf("Number of rows in the matrix: %d\n", dd_.A.N);
        printf("Number of nonzero elements: %d\n", dd_.A.Nnz);
        printf("Background profiles: %s\n", id_profile_.c_str());
        printf("output .hdf5 file: %s\n", hdf5_name_out_.c_str());
        printf("------------------ Plasma parameters -------------\n");
        printf("\tTref[erg] = %0.3e,   Tref[K] = %0.3e\n", Tref_, Tref_/cc.kB_);
        printf("\tden-ref[cm-3] = %0.3e\n", den_ref_);
        printf("\twp[s-1] = %0.3e\n", wp_);
        printf("\tld[cm] = %0.3e\n", ld_);
        printf("\tvth[cm/s] = %0.3e,   vth/c = %0.3e\n", vth_, vth_/cc.c_light_);
        printf("------------------ Normalized parameters -------------\n");
        printf("antenna frequency (norm. to wp): \t%0.3f\n", dd_.w);
        printf("diff. in the velocity space: \t%0.3f\n", dd_.diff);
        printf("spatial step (norm. to ld): \t%0.3e\n", dd_.h);
        printf("velocity step (norm. to vth): \t%0.3e\n", dd_.dv);
        printf("\txmax/ld = %0.1f,  \tfull size [cm] = %0.3e\n",  x_[dd_.Nx-1], x_[dd_.Nx-1] * ld_);
        printf("\tvmax/vth = %0.1f, \tfull size [cm/s] = %0.3e\n", v_[dd_.Nv-1], v_[dd_.Nv-1] * vth_);
        printf("--------------------------------------------------\n\n");
    }

    // form the name of the output file:
    void set_output_name()
    {
        std::stringstream sstr;
        std::string temp_string;
        sstr << "out_" << dd_.nx << "_" << dd_.nv << "_w" << dd_.w;
        sstr << "_Lx" << dd_.xmax << "_Lv" << dd_.vmax;
        sstr << "_" << id_profile_ << ".hdf5";
        hdf5_name_out_ = sstr.str();
    }

    // set analytically the number of nonzero elements in the matrix.
    virtual void compute_Nnz() = 0;

    void init_background_distribution(
        qreal* dev_T, qreal* dev_n, qreal* dev_dT, qreal* dev_dn) 
    {
        init_background_distribution_form1<<<dd_.Nx, dd_.Nv>>>(dev_T, dev_n);
    } 
};