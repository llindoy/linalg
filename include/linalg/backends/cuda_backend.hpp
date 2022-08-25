#ifndef LINALG_CUDA_BACKEND_HPP
#define LINALG_CUDA_BACKEND_HPP

#include "cuda_utils.hpp"
#include "blas_backend.hpp"
#include "cuda_environment.hpp"
#include "cuda_kernels.hpp"
#include "cublas_wrapper.hpp"

#include <iostream>
#include <vector>
#include <tuple>
#include <utility>

#ifdef __NVCC__
#include <thrust/equal.h>

namespace linalg
{

namespace internals
{
template <typename T>struct ones_indexer;
template <>struct ones_indexer<float>{static constexpr size_t index(){return 0;}};
template <>struct ones_indexer<double>{static constexpr size_t index(){return 1;}};
template <>struct ones_indexer<complex<float>>{static constexpr size_t index(){return 2;}};
template <>struct ones_indexer<complex<double>>{static constexpr size_t index(){return 3;}};
}

class cuda_backend : public backend_base
{
public:
    using size_type = typename cuda_environment::size_type;
    using index_type = typename cuda_environment::index_type;
    using int_type = index_type;
    static cuda_environment _environment;
    static inline cuda_environment& environment(){return _environment;}

protected:
    static constexpr size_type default_nstreams = 1;
    static constexpr size_type default_device_id = 0;

//    static cuda_environment _environment;
    //static cuda_kernel_opt<T> m_kernel_launcher;
    
    template <typename T>
    using ones_type = std::pair<T*, size_type>;
    using ones_tuple = std::tuple<ones_type<float>, ones_type<double>, ones_type<complex<float>> , ones_type<complex<double>> >;

    static ones_tuple& ones()
    {
        static ones_tuple m_ones;
        return m_ones;
    }

public:
    using transform_type = cublasOperation_t;
    static constexpr transform_type op_n = CUBLAS_OP_N;
    static constexpr transform_type op_t = CUBLAS_OP_T;
    static constexpr transform_type op_h = CUBLAS_OP_C;
    static constexpr transform_type op_c = static_cast<transform_type>('I');

    static bool is_initialised(){return _environment.is_initialised();}


    //the initialisation routines for the cuda_backend are not thread safe.
    static void initialise(cuda_environment&& env)
    {
        initialise_empty_ones_buffers();
        _environment = std::move(env);
    }
    static void initialise(size_type device_id = default_device_id, size_type nstreams = default_nstreams)
    {
        initialise_empty_ones_buffers();
        CALL_AND_HANDLE(_environment.init(device_id, nstreams), "Failed to initialise cuda_backend.  Error when initialising the cuda environment object.");
    }

    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type allocate_ones(size_type n)
    {
        try
        {
            ASSERT(_environment.is_initialised(), "his must be allocated following the instantiation of the cuda environment.");
            ones_type<T>& ones = std::get<internals::ones_indexer<T>::index()>(ones());
            if(std::get<1>(ones) > n){return;}
            else
            {
                //if the ones array has previously been allocated but its size is too small we need to deallocate it
                if(std::get<0>(ones) != nullptr)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaFree(std::get<0>(ones))), "Unable to free previously allocated ones array.");
                }
                std::get<1>(ones) = n;
            }
            //now we need to allocate the ones buffers.
            CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&std::get<0>(ones), std::get<1>(ones)*sizeof(T))), "Error when calling cudaMalloc.");


            //now we fill the ones array with ones
            size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
            dim3 dg((std::get<1>(ones)+nthreads-1)/nthreads);
            dim3 db(nthreads);

            cuda_kernels::fill_n<<<dg, db, 0, _environment.current_stream()>>>(std::get<0>(ones), std::get<1>(ones), T(1.0));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to allocate ones array.");
        }
    }

    static void destroy()
    {
        CALL_AND_HANDLE(clean_up_ones(), "Failed to destroy cuda_backend.  Error when clearing the allocated ones vectors.");
        CALL_AND_HANDLE(_environment.destroy(), "Failed to destroy cuda_backend.  Error when destroying the cuda environment object.");
    }

    static std::ostream& device_properties(std::ostream& out)
    {
        try
        {
            int nDevices;
            cudaGetDeviceCount(&nDevices);
            for (int i = 0; i < nDevices; i++) 
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                out << "Device Number: " << i << std::endl;
                out << prop;
            }
            return out;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to print cuda_backend device properties.");
        }
    }

    template <typename F, typename ... Args>
    static inline void async_for(size_type start, size_type end, F&& f, Args&& ... args)
    {
        try
        {
            ASSERT(_environment.is_initialised(), "Failed to perform async_for calculation on cuda_backend.  The backend has not been initialised.");
            ASSERT(start <= end, "Unable to perform async_for.  The final index must be less than the starting index");
            for(size_type i=start; i < end; ++i)
            {
                f(i, std::forward<Args>(args)...);
                _environment.increment_stream_id();
            }
            _environment.reset_stream_id();
            //now we sync all of the streams
            cudaDeviceSynchronize();
        }
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating for loop using asynchronous execution.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate for loop using asynchronous execution.");
        }
    }


    static inline cudaStream_t current_stream(){return _environment.current_stream();}
protected:
    static inline void set_cublas_stream()
    {
        CALL_AND_HANDLE(cublas_safe_call(cublasSetStream(_environment.cublas_handle(), _environment.current_stream())), "Failed to set the current value of the cublas stream.");
    }

public:
    template <typename T> static inline bool is_equal(const T* a, const T* b, size_type n)
    {
        ASSERT(_environment.is_initialised(), "cuda backend is_equal call failed.  The cuda environment has not yet been initialised.");
        return thrust::equal(a, a+n, b, []__device__(T a, T b){return a == b;} );
    }

    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type vector_scalar_product(int N, T A, const T* X, int INCX, T* Y, int INCY)
    {
        ASSERT(_environment.is_initialised(), "cuda backend vector_scalar_product call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((N+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::vector_scalar_product<<<dg, db, 0, _environment.current_stream()>>>(N, A, X, INCX, Y, INCY);
    }

    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, T>::type trace(int N, const T* X, int INCX)
    {
        ASSERT(_environment.is_initialised(), "cuda backend vector_scalar_product call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((N+nthreads-1)/nthreads);
        dim3 db(nthreads);
        RAISE_EXCEPTION("Trace has not yet been implemented for the cuda backend.");
    }

    //valid blas routines
    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type axpy(int N, T A, const T* X, int INCX, T* Y, int INCY)
    {
        ASSERT(_environment.is_initialised(), "cuda backend axpy call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "axpy call Failed.");
        CALL_AND_HANDLE(cublas::axpy(_environment.cublas_handle(), N, &A, X, INCX, Y, INCY), "axpy call failed using cuda_backend.");
    }

    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type scal(int N, T A, T* X, int INCX)
    {
        ASSERT(_environment.is_initialised(), "cuda backend Scal call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "scal call Failed.");
        CALL_AND_HANDLE(cublas::scal(_environment.cublas_handle(), N, &A, X, INCX), "Scal call failed using cuda_backend.");
    }

    template <typename T>
    static inline typename std::enable_if<is_valid_value_type<T>::value, T>::type dot(bool conj, int N, const T* X, int INCX, const T* Y, int INCY)
    {
        ASSERT(_environment.is_initialised(), "cuda backend dot call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "dot call Failed.");
        CALL_AND_HANDLE(return cublas::dot(_environment.cublas_handle(), conj, N, X, INCX, Y, INCY), "Failed to compute dot product using cuda backend.");
    }

    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    gemm(transform_type TRANSA, transform_type TRANSB, int M, int N, int K, T ALPHA, const T* A, int LDA, const T* B, int LDB, T BETA, T* C, int LDC)
    {
        ASSERT(_environment.is_initialised(), "cuda backend gemm call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "gemm call Failed.");
        CALL_AND_HANDLE(cublas::gemm(_environment.cublas_handle(), TRANSA, TRANSB, M, N, K, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC), "Failed to compute general matrix matrix product using cuda backend.");
    }


    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    gemv(transform_type trans, int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy)
    {
        ASSERT(_environment.is_initialised(), "cuda backend gemv call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "gemv call Failed.");
        CALL_AND_HANDLE(cublas::gemv(_environment.cublas_handle(), trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy), "Failed to compute general matrix vector product using cuda backend.");
    }


    //batched_gemm
    template <typename T> 
    static inline typename std::enable_if<is_valid_value_type<T>::value, void>::type
    batched_gemm(transform_type opA, transform_type opB, int m, int n, int k, T alpha, const T* A, int lda, long long int strideA, const T* B, int ldb, long long int strideB, T beta, T* C, int ldc, long long int strideC, int batchCount)
    {   
        ASSERT(_environment.is_initialised(), "cuda backend batched_gemm call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "batched_gemm call Failed.");
        CALL_AND_HANDLE(cublas::batched_gemm(_environment.cublas_handle(), opA, opB, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount), "Failed to cublas strided batched gemm");
    }

    //outer contraction of rank 3 tensors
    template <typename T> 
    static inline void outer_contract(transform_type opA, transform_type opB, int m, int n, int k, T alpha, const T* A, 
                                int lda, long long int strideA, const T* B, int ldb, long long int strideB, T beta, T* C, int ldc, 
                                long long int strideC, int batchCount, T* res)
    {   
        ASSERT(_environment.is_initialised(), "cuda backend outer_contract call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "outer_contract call Failed.");
        T b(0.0);
        //first we go ahead and perform the many parallel small matrix products required
        CALL_AND_HANDLE(batched_gemm(opA, opB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, b, C, ldc, strideC, batchCount), "Failed to compute contraction of outer indices of two rank three tensors.  Failed to compute the contraction over the last index.");

        //first we check that the backend has a large enough vector of ones to perform the final contraction using gemv.  If it isn't then we will throw an exception - as we don't want these routines to break thread safety.
        CALL_AND_HANDLE(allocate_ones<T>(batchCount), "Failed to allocate ones array.");
        auto& ones = std::get<internals::ones_indexer<T>::index()>(ones());

        //now we set up the gemv call to contract over k.  To do this we do C_{ij} = \sum_k A_{kij} v_k where v_k = 1.  This can be 
        //performed using a gemv call.  
        T a(1.0);   transform_type op = op_n;   int incx = 1;   int incy = 1;   int mv = m*n;   int nv = batchCount;    int ldav = mv;
        CALL_AND_HANDLE(gemv(op, mv, nv, a, C, ldav, std::get<0>(ones), incx, beta, res, incy), "Failed to compute contraction of outer indices of two rank three tensor.  Failed to compute the contraction over the first index.");
    }


public:
    //sparse matrix vector operations
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmv(bool conjA, bool conjB, int m, int n, T1 alpha, const T2* A, int inca, const T3* X, int incx, T4 beta, T5* Y, int incy)
    {   
        ASSERT(_environment.is_initialised(), "cuda backend dgmv call failed.  The cuda environment has not yet been initialised.");
        dgmv_kernel_selector(conjA, conjB, m, n, alpha, A, inca, X, incx, beta, Y, incy);
    }

    //sparse matrix matrix operations
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2* A, size_type inca, const T3* B, size_type ldb, T4 beta, T5* C, size_type ldc)
    {   
        ASSERT(_environment.is_initialised(), "cuda backend dgmm call failed.  The cuda environment has not yet been initialised.");
        dgmm_kernel_parameter_selector(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
    }

public:
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmv_kernel_selector(bool conjA, bool conjB, size_type m, size_type n, T1 alpha, const T2* A, int inca, const T3* X, int incx, T4 beta, T5* Y, int incy)
    {
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((m+nthreads-1)/nthreads);  dim3 db(nthreads);

        auto ecopt2 = [] __device__ (const T2& a){return conj(a);};
        auto ecopt3 = [] __device__ (const T3& a){return conj(a);};
        auto enopt2 = [] __device__ (const T2& a){return a;};
        auto enopt3 = [] __device__ (const T3& a){return a;};
        if(inca == 1 && incx == 1 && incy == 1)
        {
            if(conjA)
            {
                if(conjB){  cuda_kernels::elementwise_multiplication<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, X, beta, Y, ecopt2, ecopt3);}
                else{       cuda_kernels::elementwise_multiplication<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, X, beta, Y, ecopt2, enopt3);}
            }
            else
            {
                if(conjB){  cuda_kernels::elementwise_multiplication<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, X, beta, Y, enopt2, ecopt3);}
                else{       cuda_kernels::elementwise_multiplication<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, X, beta, Y, enopt2, enopt3);}
            }
            
        }
        else
        {
            if(conjA)
            {
                if(conjB){  cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, ecopt2, ecopt3);}
                else{       cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, ecopt2, enopt3);}
            }
            else
            {
                if(conjB){  cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, enopt2, ecopt3);}
                else{       cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, _environment.current_stream()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, enopt2, enopt3);}
            }

        }
    }


private:
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm_kernel_parameter_selector(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2* A, size_type inca, const T3* B, size_type ldb, T4 beta, T5* C, size_type ldc)
    {
        //determine the TILE_DIM and BLOCK_ROWS parameters used for the dgmm operation.  This also checks whether this will result in to much shared memory and if it will then it 
        //uses a kernel with a smaller TILE_DIM.  We will want to optimise this in the future.
        if(sizeof(T3) <= sizeof(float))
        {
            dgmm_kernel_launcher<32, 16>(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
        else if(sizeof(T3) <= sizeof(double))
        {
            dgmm_kernel_launcher<16, 8>(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
        else
        {
            dgmm_kernel_launcher<16, 8>(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
    }

public:
    template <size_t TILE_DIM, size_t BLOCK_ROWS, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm_kernel_launcher(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2* A, size_type inca, const T3* B, size_type ldb, T4 beta, T5* C, size_type ldc)
    {   
        bool conjA = false;
        bool conjB = false;
        bool transDense = false;

        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
        
        auto ecopt2 = [] __device__ (const T2& a){return conj(a);};
        auto ecopt3 = [] __device__ (const T3& a){return conj(a);};
        auto enopt2 = [] __device__ (const T2& a){return a;};
        auto enopt3 = [] __device__ (const T3& a){return a;};
        auto stream = _environment.current_stream();
        //we will be calling the dgm_dm_m? kernels so we should determine what parameters we need
        if(sparse_left)
        {
            if(opA == op_c || opA == op_h){conjA = true;}
            if(opB == op_c || opB == op_h){conjB = true;}
            if(opB == op_t || opB == op_h){transDense = true;}
            size_type max_km = m > k ? m : k;

            if(!transDense)
            {   
                constexpr size_t smem = sizeof(T5)*TILE_DIM*(TILE_DIM+1);
                dim3 dimGrid((n+TILE_DIM-1)/TILE_DIM, (max_km+TILE_DIM-1)/TILE_DIM, 1);
                if(conjA && conjB){         cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, ecopt3);}
                else if(conjA && !conjB){   cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, enopt3);}
                else if(!conjA && conjB){   cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, ecopt3);}
                else{                       cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, enopt3);}
            }
            else
            {
                constexpr size_t smem = sizeof(T5)*TILE_DIM*(TILE_DIM+2);
                dim3 dimGrid((max_km+TILE_DIM-1)/TILE_DIM, (n+TILE_DIM-1)/TILE_DIM, 1);
                if(conjA && conjB){         cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, ecopt3);}
                else if(conjA && !conjB){   cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, enopt3);}
                else if(!conjA && conjB){   cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, ecopt3);}
                else{                       cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, enopt3);}
            }
        }
        //if the sparse matrix is on the right then we need to call the dm_dgm_m? kernels
        else
        {
            if(opB == op_c || opB == op_h){conjA = true;}
            if(opA == op_c || opA == op_h){conjB = true;}
            if(opA == op_t || opA == op_h){transDense = true;}
            size_type max_kn = n > k ? n : k;

            if(!transDense)
            {
                constexpr size_t smem = sizeof(T5)*TILE_DIM*(TILE_DIM+1);
                dim3 dimGrid((max_kn+TILE_DIM-1)/TILE_DIM, (m+TILE_DIM-1)/TILE_DIM, 1);

                if(conjA && conjB){         cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, ecopt2);}
                else if(conjA && !conjB){   cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, enopt2);}
                else if(!conjA && conjB){   cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, ecopt2);}
                else{                       cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, enopt2);}
            }
            else
            {
                constexpr size_t smem = sizeof(T5)*TILE_DIM*(TILE_DIM+2);
                dim3 dimGrid((m+TILE_DIM-1)/TILE_DIM, (max_kn+TILE_DIM-1)/TILE_DIM, 1);

                if(conjA && conjB){         cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, ecopt2);}
                else if(conjA && !conjB){   cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, enopt2);}
                else if(!conjA && conjB){   cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, ecopt2);}
                else{                       cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, enopt2);}
            }
        }
    }

public:
    template <typename T>
    static inline void complex_conjugate(size_type /*size*/, const T* const /*X*/, T* const /*Y*/){}

    template <typename T>
    static inline void complex_conjugate(size_type size, const complex<T>* const X, complex<T>* const Y)
    {   
        ASSERT(_environment.is_initialised(), "cuda backend complex_conjugate call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((size+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::complex_conjugate<<<dg, db, 0, _environment.current_stream()>>>(X, size, Y);
    }
    
    template <typename T, typename expr> 
    static inline void evaluate_expression_tree(T* res, size_type n, const expr& e)
    {
        ASSERT(_environment.is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::eval_expression_kernel<<<dg, db, 0, _environment.current_stream()>>>(res, n, e);
    }

    template <typename T, typename expr> 
    static inline void evaluate_add_expression_tree(T* res, size_type n, const expr& e)
    {
        ASSERT(_environment.is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::eval_add_expression_kernel<<<dg, db, 0, _environment.current_stream()>>>(res, n, e);
    }

    template <typename T, typename expr> 
    static inline void evaluate_sub_expression_tree(T* res, size_type n, const expr& e)
    {
        ASSERT(_environment.is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::eval_sub_expression_kernel<<<dg, db, 0, _environment.current_stream()>>>(res, n, e);
    }

    template <typename T, typename expr> 
    static inline void evaluate_expression_tree_strided(T* res, size_type n, size_type stride, const expr& e)
    {
        ASSERT(_environment.is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::eval_expression_strided_kernel<<<dg, db, 0, _environment.current_stream()>>>(res, n, stride, e);
    }

    template <typename T, typename expr> 
    static inline void evaluate_add_expression_tree_strided(T* res, size_type n, size_type stride, const expr& e)
    {
        ASSERT(_environment.is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::eval_add_expression_strided_kernel<<<dg, db, 0, _environment.current_stream()>>>(res, n, stride, e);
    }

    template <typename T, typename expr> 
    static inline void evaluate_sub_expression_tree_strided(T* res, size_type n, size_type stride, const expr& e)
    {
        ASSERT(_environment.is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::eval_sub_expression_strided_kernel<<<dg, db, 0, _environment.current_stream()>>>(res, n, stride, e);
    }
public:
    template <typename T>
    static inline void transpose(bool conj, int m, int n, const T& alpha, const T* in, const T& beta, T* out)
    {
        ASSERT(_environment.is_initialised(), "cuda backend transpose call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(set_cublas_stream(), "transpose call Failed.");
        ASSERT(in != out, "Failed to evaluate cuda_backend::transpose.  The input and output buffers must not be the same.");
        cublasOperation_t op = conj ? op_h : op_t;
        CALL_AND_HANDLE(cublas::geam(_environment.cublas_handle(), op, op_n, m, n, &alpha, in, n, &beta, in, m, out, m), "Failed to evaluate cuda_backend::transpose.  Error when calling geam.");
    }

    //we might want to modify this in the future so that it uses a specialised kernel for performing the batched transpose operation.  This is likely to lead to improved performance
    //for problems in which the overhead of the kernel launches dominates the time to run the operation.
    template <typename T>
    static inline void batched_transpose(bool conj, size_type m, size_type n, const T& alpha, const T* in, const T& beta, T* out, size_type batchCount)
    {
        ASSERT(_environment.is_initialised(), "cuda backend batched_transpose call failed.  The cuda environment has not yet been initialised.");
        ASSERT(in != out, "Failed to evaluate cuda_backend::batched_transpose.  The input and output buffers must not be the same.");
        cublasOperation_t op = conj ? op_h : op_t;

        size_type skip = m*n;
        for(size_type i=0; i<batchCount; ++i)
        {
            size_type bskip = skip*i;
            CALL_AND_HANDLE(set_cublas_stream(), "batched transpose call Failed.  Failed to set cublas stream to parallelise transpose calls.");
            CALL_AND_HANDLE(cublas::geam(_environment.cublas_handle(), op, op_n, m, n, &alpha, in, n, &beta, in+bskip, m, out+bskip, m), "Failed to evaluate cuda_backend::transpose.  Error when calling geam.");
            _environment.increment_stream_id();
        }
        _environment.reset_stream_id();
        cudaDeviceSynchronize();
    }

    //function for copying between two buffers
    template <typename T> 
    static inline void copy(const T* src, size_type n, T* dest)
    {
        ASSERT(_environment.is_initialised(), "cuda backend copy operation failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(dest, src, n*sizeof(T), cudaMemcpyDeviceToDevice)), "Failed to copy memory buffer from one buffer to another.  cudaMemcpy call failed.");
    }


    template <typename T> 
    static inline void assign(const T* src, size_type n, T* dest, T beta = T(0))
    {
        ASSERT(_environment.is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::addition_assign_array<<<dg, db, 0, _environment.current_stream()>>>(src, n, dest);
    }

    template <typename T> 
    static inline void addition_assign(const T* src, size_type n, T* dest)
    {
        ASSERT(_environment.is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::addition_assign_array<<<dg, db, 0, _environment.current_stream()>>>(src, n, dest);
    }

    template <typename T> 
    static inline void subtraction_assign(const T* src, size_type n, T* dest)
    {
        ASSERT(_environment.is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::subtraction_assign_array<<<dg, db, 0, _environment.current_stream()>>>(src, n, dest);
    }

    template <typename T> 
    static inline void copy_real_to_complex(const T* src, size_type n, complex<T>* dest)
    {
        ASSERT(_environment.is_initialised(), "cuda backend copy_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::copy_real_to_complex_array<<<dg, db, 0, _environment.current_stream()>>>(src, n, dest);
    }

    template <typename T> 
    static inline void addition_assign_real_to_complex(const T* src, size_type n, complex<T>* dest)
    {
        ASSERT(_environment.is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::addition_assign_real_to_complex_array<<<dg, db, 0, _environment.current_stream()>>>(src, n, dest);
    }

    template <typename T> 
    static inline void subtraction_assign_real_to_complex(const T* src, size_type n, complex<T>* dest)
    {
        ASSERT(_environment.is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::addition_assign_real_to_complex_array<<<dg, db, 0, _environment.current_stream()>>>(src, n, dest);
    }
    
    //function for filling a buffer with a value
    template <typename T> 
    static inline void fill_n(T* dest, size_type n, const T& val)
    {
        ASSERT(_environment.is_initialised(), "cuda backend fill_n operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((n+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::fill_n<<<dg, db, 0, _environment.current_stream()>>>(dest, n, val);
    }

    template<typename T, typename Func, typename ... Args>
    static inline void func_fill_1(T* res, size_type m, Func&& f, Args&&... args)
    {
        ASSERT(_environment.is_initialised(), "cuda backend func_fill_1 operation failed.  The cuda environment has not yet been initialised.");
        size_type nthreads = _environment.maximum_dimensions_threads_per_block()[0];
        dim3 dg((m+nthreads-1)/nthreads);
        dim3 db(nthreads);
        cuda_kernels::fill_func_1<<<dg, db, 0, _environment.current_stream()>>>(res, m, std::forward<Func>(f), std::forward<Args>(args)...);
    }

    template<typename T, typename Func, typename ... Args>
    static inline void func_fill_2(T* res, size_type m, size_type n, Func&& f, Args&&... args)
    {
        ASSERT(_environment.is_initialised(), "cuda backend func_fill_2 operation failed.  The cuda environment has not yet been initialised.");
        dim3 dg, db;
        size_type mnt = _environment.maximum_threads_per_block();
        size_type mnx = _environment.maximum_dimensions_threads_per_block()[0];
        size_type mny = _environment.maximum_dimensions_threads_per_block()[1];
        if(n >= mnt)
        {
            db.y = mnt < mny ? mnt : mny;
            db.x = mnt/db.y;
        }
        else
        {
            if(n >= 16 && (n & (n - 1)) == 0){ db.y = n < mny ? n : mny; db.x = mnt/db.y; } //if m >= 16 and m is a power of 2 then we use m as the value for db.x
            else{db.y = 16 < mny ? 16 : mny; db.x = mnt / db.y;}    //otherwise we will just use 16 for db.x
        }
        db.x = db.x < mnx ? db.x : mnx;
        dg.x = (m+db.x-1)/db.x;
        dg.y = (n+db.y-1)/db.y;

        cuda_kernels::fill_func_2<<<dg, db, 0, _environment.current_stream()>>>(res, m, n, std::forward<Func>(f), std::forward<Args>(args)...);
    }

    template<typename T, typename Func, typename ... Args>
    static inline void func_fill_3(T* res, size_type m, size_type n, size_type o, Func&& f, Args&&... args)
    {
        ASSERT(_environment.is_initialised(), "cuda backend func_fill_3 operation failed.  The cuda environment has not yet been initialised.");
        dim3 dg, db;
        size_type mnt = _environment.maximum_threads_per_block();
        size_type mnx = _environment.maximum_dimensions_threads_per_block()[0];
        size_type mny = _environment.maximum_dimensions_threads_per_block()[1];
        size_type mnz = _environment.maximum_dimensions_threads_per_block()[2];

        if(o >= mnt)
        {
            db.z = mnt < mnz ? mnt : mnz; 

            size_type ncomp = mnt / db.z;
            if(n >= ncomp){db.y = ncomp < mny ? ncomp : mny; db.x = ncomp / db.y;}
            else
            {
                size_type maxy = mny < ncomp ? mny : ncomp;
                if(n >= 8 && (n & (n - 1)) == 0){ db.y = n < maxy ? n : maxy; db.x = ncomp/db.y; } //if m >= 16 and m is a power of 2 then we use m as the value for db.x
                else{db.y = 8 < maxy ? 8 : maxy; db.x = ncomp / db.y;}    //otherwise we will just use 16 for db.x
            }
        }
        else
        {
            if(o >= 16 && (o & (o - 1)) == 0){db.z = o;}
            else{db.z = 16;}
            db.z = db.z < mnz ? db.z : mnz;

            size_type ncomp = mnt/db.z;
            db.y = ncomp < mny ? ncomp : mny;   db.x = ncomp/db.y;
            while(db.y > n){db.y = db.y >> 1;  db.x = db.x << 1;}
        }
        db.x = db.x < mnx ? db.x : mnx;

        dg.x = (m+db.x-1)/db.x;
        dg.y = (n+db.y-1)/db.y;
        dg.z = (o+db.z-1)/db.z;
        cuda_kernels::fill_func_3<<<dg, db, 0, _environment.current_stream()>>>(res, m, n, o, std::forward<Func>(f), std::forward<Args>(args)...);
    }

    template <typename T> 
    static inline void fill_matrix_block(const T* src, size_type m, size_type n, T* dest, size_type m2, size_type n2)
    {   
        ASSERT(n <= n2 && m <= m2, "fill_block call failed.  The subblock is larger than the full matrix.");
        ASSERT(_environment.is_initialised(), "cuda backend func_fill_2 operation failed.  The cuda environment has not yet been initialised.");
        dim3 dg, db;
        size_type mnt = _environment.maximum_threads_per_block();
        size_type mnx = _environment.maximum_dimensions_threads_per_block()[0];
        size_type mny = _environment.maximum_dimensions_threads_per_block()[1];
        if(n >= mnt)
        {
            db.y = mnt < mny ? mnt : mny;
            db.x = mnt/db.y;
        }
        else
        {
            if(n >= 16 && (n & (n - 1)) == 0){ db.y = n < mny ? n : mny; db.x = mnt/db.y; } //if m >= 16 and m is a power of 2 then we use m as the value for db.x
            else{db.y = 16 < mny ? 16 : mny; db.x = mnt / db.y;}    //otherwise we will just use 16 for db.x
        }
        db.x = db.x < mnx ? db.x : mnx;
        dg.x = (m+db.x-1)/db.x;
        dg.y = (n+db.y-1)/db.y;

        cuda_kernels::fill_matrix_block<<<dg, db, 0, _environment.current_stream()>>>(src, m, n, dest, m2, n2);
    }


    template <typename T>
    static inline void transfer_coo_tuple_to_csr(const std::vector<std::tuple<index_type, index_type, T> >& coo, T* vals, index_type* colinds)
    {
        ASSERT(_environment.is_initialised(), "cuda backend transfer_coo_tuple_to_csr operation failed.  The cuda environment has not yet been initialised.");
        T* d_vals;  index_type* d_colinds;  size_type n = coo.size();
        CALL_AND_HANDLE(d_vals = new T[n], "Failed to transfer coo tuple to csr object.  Error when allocating device vector for storing values.");
        CALL_AND_HANDLE(d_colinds = new index_type[n], "Failed to transfer coo tuple to csr object.  Error when allocating device vector for storing colinds.")

        //now we copy the vector buffer to set colinds
        for(size_type i=0; i<n; ++i)
        {
            d_colinds[i] = std::get<1>(coo[i]);
            d_vals[i] = std::get<2>(coo[i]);
        }

        //and now we copy the results
        CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(vals, d_vals, n*sizeof(T), cudaMemcpyHostToDevice)), "Failed to transfer coo tuple to csr object.  cudaMemcpy call failed.");
        CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(colinds, d_colinds, n*sizeof(index_type), cudaMemcpyHostToDevice)), "Failed to transfer coo tuple to csr object.  cudaMemcpy call failed.");

        CALL_AND_HANDLE(delete[] d_vals, "Failed to transfer coo tuple to csr object.  Error when deallocating device vector for storing values.");
        CALL_AND_HANDLE(delete[] d_colinds, "Failed to transfer coo tuple to csr object.  Error when deallocating device vector for storing colinds.")
    }

public:
    template <typename T> 
    static void heev(cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, T* A, int lda, typename get_real_type<T>::type* W, T* work, int lwork, int* devinfo)
    {
        ASSERT(_environment.is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::heev(_environment.cusolver_dn_handle(), jobz, uplo, n, A, lda, W, work, lwork, devinfo), "cuda backend heev call failed.  Error when calling heev.");
    }

    template <typename T> 
    static void heev_buffersize(cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, T* A, int lda, typename get_real_type<T>::type* W, int* lwork)
    {
        ASSERT(_environment.is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::heev_buffersize(_environment.cusolver_dn_handle(), jobz, uplo, n, A, lda, W, lwork), "cuda backend heev_buffersize call failed.  Error when calling determining the workspace buffer size.");
    }

public:
    template <typename T>
    static void getrf(int m, int n, T* A, int lda, T* work, int* ipiv, int* devinfo)
    {
        ASSERT(_environment.is_initialised(), "cuda backend getrf call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::getrf(_environment.cusolver_dn_handle(), m, n, A, lda, work, ipiv, devinfo), "cuda backend getrf call failed.  Error when calling getrf.");
    }

    template <typename T>
    static void getrf_buffersize(int m, int n, T* A, int lda, int* lwork)
    {
        ASSERT(_environment.is_initialised(), "cuda backend getrf call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::getrf_buffersize(_environment.cusolver_dn_handle(), m, n, A, lda, lwork), "cuda backend getrf_buffersize call failed.  Error when determining the workspace buffer size.");
    }

public:
    template <typename T> 
    static void gesvd(const char jobu, const char jobv, const int m, const int n, T* A, const int lda, typename get_real_type<T>::type* S, T* U, const int ldu, T* VT, const int ldvt, T* work, const int lwork, typename get_real_type<T>::type* rwork, int* devinfo)
    {
        ASSERT(_environment.is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::gesvd(_environment.cusolver_dn_handle(), jobu, jobv, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devinfo), "cuda backend gesvd call failed.  Error when calling gesvd.");
    }

    template <typename T> 
    static void gesvd_buffersize(int m, int n, int& lwork)
    {
        ASSERT(_environment.is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::gesvd_params<T>::buffersize(_environment.cusolver_dn_handle(), m, n, lwork), "cuda backend gesvd_buffersize call failed.  Error when calling determining the workspace buffer size.");
    }

    template <typename T> 
    static void gesvdj_buffersize(cusolverEigMode_t jobz, const int econ, const int m, const int n, T* A, const int lda, typename get_real_type<T>::type* S, T* U, const int ldu, T* VT, const int ldvt, int& lwork, gesvdjInfo_t params)
    {
        ASSERT(_environment.is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::gesvdj_buffersize(_environment.cusolver_dn_handle(), jobz, econ, m, n, A, lda, S, U, ldu, VT, ldvt, lwork, params), "cuda backend gesvd_buffersize call failed.  Error when calling determining the workspace buffer size.");
    }

    template <typename T> 
    static void gesvdj(cusolverEigMode_t jobz, const int econ, const int m, const int n, T* A, const int lda, typename get_real_type<T>::type* S, T* U, const int ldu, T* VT, const int ldvt, T* work, const int lwork, int* devinfo, gesvdjInfo_t params)
    {
        ASSERT(_environment.is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
        CALL_AND_HANDLE(cusolver::gesvdj(_environment.cusolver_dn_handle(), jobz, econ, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, devinfo, params), "cuda backend gesvd call failed.  Error when calling gesvd.");
    }

private:
    static inline void initialise_empty_ones_buffers()
    {
        ones_type<float>& fones = std::get<0>(ones());  std::get<0>(fones) = nullptr;   std::get<1>(fones) = 0;
        ones_type<double>& dones = std::get<1>(ones());  std::get<0>(dones) = nullptr;   std::get<1>(dones) = 0;
        ones_type<complex<float>>& cones = std::get<2>(ones());  std::get<0>(cones) = nullptr;   std::get<1>(cones) = 0;
        ones_type<complex<double>>& zones = std::get<3>(ones());  std::get<0>(zones) = nullptr;   std::get<1>(zones) = 0;
    }

    static inline void clean_up_ones()
    {
        {
            ones_type<float>& ones = std::get<0>(ones());  
            if(std::get<0>(ones) != nullptr){CALL_AND_HANDLE(cuda_safe_call(cudaFree(std::get<0>(ones))), "Failed to clean up ones buffer.  Error when calling cudaFree.");}
            std::get<0>(ones) = nullptr;
            std::get<1>(ones) = 0;
        }
        {
            ones_type<double>& ones = std::get<1>(ones());  
            if(std::get<0>(ones) != nullptr){CALL_AND_HANDLE(cuda_safe_call(cudaFree(std::get<0>(ones))), "Failed to clean up ones buffer.  Error when calling cudaFree.");}
            std::get<0>(ones) = nullptr;
            std::get<1>(ones) = 0;
        }
        {
            ones_type<complex<float>>& ones = std::get<2>(ones());  
            if(std::get<0>(ones) != nullptr){CALL_AND_HANDLE(cuda_safe_call(cudaFree(std::get<0>(ones))), "Failed to clean up ones buffer.  Error when calling cudaFree.");}
            std::get<0>(ones) = nullptr;
            std::get<1>(ones) = 0;
        }
        {
            ones_type<complex<double>>& ones = std::get<3>(ones());  
            if(std::get<0>(ones) != nullptr){CALL_AND_HANDLE(cuda_safe_call(cudaFree(std::get<0>(ones))), "Failed to clean up ones buffer.  Error when calling cudaFree.");}
            std::get<0>(ones) = nullptr;
            std::get<1>(ones) = 0;
        }
    }
};  //cuda_backend

cuda_environment cuda_backend::_environment;
cuda_backend::ones_tuple cuda_backend::ones();

}   //namespace linalg

#endif


#endif  //LINALG_CUDA_BACKEND_HPP//

