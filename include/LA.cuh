#pragma once

#include "mix.h"


/**
 * ----------------------------------------------------------------------
 * --- Linear algebra ---
 * ----------------------------------------------------------------------
*/
class LA{
public:


    /**
     * Find SVD of the matrix @param A. 
     * The method does not work for large matrices.
    */
    static void cond_number_cusolver(YMatrix<ycomplex> &A)
    {
        float time;
        cudaEvent_t start, stop;
        int info_gpu = 0; /* host copy of error info */
        int *devInfo = nullptr;

        uint32_t N = A.get_nr();

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cusolverDnHandle_t cusolverH = NULL;
        cublasHandle_t cublasH = NULL;
        int32_t lda = N; 
        uint32_t N2 = N * N;

        std::shared_ptr<ycomplex[]> U  = std::shared_ptr<ycomplex[]>(new ycomplex[N2]); // left singular vectors;
        std::shared_ptr<ycomplex[]> VT = std::shared_ptr<ycomplex[]>(new ycomplex[N2]); // complex-conjugated right singular vectors;
        std::shared_ptr<qreal[]> S  = std::shared_ptr<qreal[]>(new qreal[N]); // numerical singular values ordered in 
                                                                                // the descending order;
        ycuComplex *d_A  = nullptr;
        ycuComplex *d_U  = nullptr;  
        ycuComplex *d_VT = nullptr; 
        qreal *d_S = nullptr;  

        int lwork = 0; /* size of workspace */
        ycuComplex *d_work = nullptr;
        qreal *d_rwork = nullptr;

        /* create cusolver handle, bind a stream */
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUBLAS_CHECK(cublasCreate(&cublasH));

        /* allocate matrices on the device */
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),  sizeof(ycuComplex) * N2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U),  sizeof(ycuComplex) * N2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(ycuComplex) * N2));
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W),  sizeof(ycuComplex) * N2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S),  sizeof(qreal) * N));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

        /* copy the matrix to the device */
        CUDA_CHECK(cudaMemcpy(d_A, A.get_1d_column_major(), sizeof(ycomplex) * N2, cudaMemcpyHostToDevice));

        /* query working space of SVD */
        CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(cusolverH, N, N, &lwork));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(ycuComplex) * lwork));

        /* compute SVD*/
        signed char jobu  = 'N';  // all m columns of U
        signed char jobvt = 'N'; // all n columns of VT
        CUSOLVER_CHECK(ycuSVD(
            cusolverH, jobu, jobvt, 
            N, N, d_A, lda, 
            d_S, 
            d_U,  lda, 
            d_VT, lda, 
            d_work, lwork, d_rwork, devInfo
        ));

        qreal s_min, s_max;
        CUDA_CHECK(cudaMemcpy(&s_max, &(d_S[0]),   sizeof(qreal), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&s_min, &(d_S[N-1]), sizeof(qreal), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&info_gpu, devInfo,  sizeof(int), cudaMemcpyDeviceToHost));

        // std::cout << "s-min = " << s_min << std::endl;
        // std::cout << "s-max = " << s_max << std::endl;

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Done. Elapsed time:  %0.3e s \n", time/1e3);
        
        printf("after gesvd: info_gpu = %d\n", info_gpu);
        if (0 == info_gpu) {
            std::printf("gesvd converges \n");
        } else if (0 > info_gpu) {
            std::printf("--- WARNING: %d-th parameter is wrong --- \n", -info_gpu);
            exit(1);
        } else {
            std::printf("--- WARNING: info = %d : gesvd does NOT CONVERGE --- \n", info_gpu);
        }

        qreal cn = s_max / s_min;

        printf("\n\ts-max:\t\t\t %0.3e\n", s_max);
        printf("\ts-min:\t\t\t %0.3e\n", s_min);
        printf("\tcondition number:\t %0.3e\n", cn);

        /* free resources */
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_U));
        CUDA_CHECK(cudaFree(d_VT));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(devInfo));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_rwork));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
        CUBLAS_CHECK(cublasDestroy(cublasH));
        // CUDA_CHECK(cudaDeviceReset());

        std::cout << std::endl;
    }


    /**
     * Solve Ax = b for a sparse A.
     * Input and output variables have to be in the device memory.
    */
   static int solve_sparse_system(
        const SpMatrixC& A, 
        ycuComplex* b, 
        ycuComplex*& x, 
        const qreal& tol = 1e-12
    ){
        cusolverSpHandle_t cusolverH = NULL;
        cusparseMatDescr_t descrA = NULL;
        int singularity = 0;
        int result = 0;

        CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

        /* Matrix description */
        CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
        CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 

        /* Solver */
        ycuInverse(
                cusolverH,
                A.N,
                A.Nnz,
                descrA,
                A.values,
                A.rows,
                A.columns,
                b,
                tol,
                0,
                x,
                &singularity
        );

        if(singularity >= 0)
        {
            printf(
                "--> ERROR in LA::solve_sparse_system: singularity in R(%d,%d)\n", 
                singularity, 
                singularity
            );
            result = -1;
        }

        /* free resources */
        CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));
        // CUDA_CHECK(cudaDeviceReset());
        return result;
   }

    /**
     * Compute dev_out = A\dot x - b.
    */
    static int recheck_linear_solver(
        const SpMatrixC& A,
        const ycuComplex* x, 
        const ycuComplex* b, 
        ycuComplex*& dev_out,
        const qreal& tol = 1e-12
    ){
        float alpha =  1.0f;
        float beta  = -1.0f;
        cusparseHandle_t handle = NULL;
        void* dBuffer    = NULL;
        size_t bufferSize = 0;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;

        CUSPARSE_CHECK(cusparseCreate(&handle));

        // create supplemental vector:
        CUDA_CHECK(
            cudaMalloc((void**) &dev_out, A.N * sizeof(ycuComplex))
        );
        CUDA_CHECK( 
            cudaMemcpy(dev_out, b, A.N * sizeof(ycuComplex), cudaMemcpyDeviceToDevice) 
        );

        // Create sparse matrix A in CSR format
        CUSPARSE_CHECK(cusparseCreateCsr(
            &matA, A.N, A.N, A.Nnz,
            A.rows, A.columns, A.values,
            CUSPARSE_INDEX_32I, 
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, 
            CUDA_C_64F
        ));

        // Create dense vectors:
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, A.N, (void *) x, CUDA_C_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, A.N, (void *) dev_out, CUDA_C_64F));

        // allocate a buffer for the matrix-vector multiplication:
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, 
            &beta, vecY, 
            CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            &bufferSize
        ));
        CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

        // Multiplication of a sparse matrix by a dense vector:
        CUSPARSE_CHECK(cusparseSpMV(
            handle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, 
            &beta, vecY, 
            CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, 
            dBuffer
        ));

        // device memory deallocation
        CUSPARSE_CHECK(cusparseDestroySpMat(matA));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
        CUSPARSE_CHECK(cusparseDestroy(handle));
        CUDA_CHECK(cudaFree(dBuffer));

        return 0;
    }

};





