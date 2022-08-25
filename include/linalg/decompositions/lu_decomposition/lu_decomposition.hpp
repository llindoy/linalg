#ifndef LINALG_DECOMPOSITIONS_LU_DECOMPOSITION_HPP
#define LINALG_DECOMPOSITIONS_LU_DECOMPOSITION_HPP

#include "../decompositions_common.hpp"

namespace linalg
{

namespace internal
{
struct lu_result_validation
{
    template <typename matrix, typename ipiv_type>
    static void validate_ipiv(const matrix& m, ipiv_type& ipiv)
    {
        size_t minmn = m.shape(0) > m.shape(1) ? m.shape(1) : m.shape(0);
        if(ipiv.size() != minmn){CALL_AND_HANDLE(ipiv.resize(minmn), "Failed to reshape the pivot array.");}
    }
};
}

template <typename matrix_type> 
class lu_decomposition<matrix_type, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, blas_backend>::value, void>::type > 
{
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;

public:
    lu_decomposition(){}

    template <typename mat_type, typename mat_typeb>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value && internal::valid_decomposition_matrix<mat_typeb, value_type, backend_type>::value, void>::type 
    operator()(const mat_type& A, mat_typeb& LU, vector<int, backend_type>& ipiv)
    {
        try
        {
            CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");
            CALL_AND_HANDLE(LU = A, "Failed to copy matrix.");
            CALL_AND_HANDLE(blas_backend::getrf(LU.size(1), LU.size(0), LU.buffer(), LU.size(1), ipiv.buffer()), "Lapack call failed.");
        }        
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating LU decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
        }
    }
    
    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type 
    operator()(mat_type& A, vector<int, backend_type>& ipiv)
    {
        try
        {
            CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");
            CALL_AND_HANDLE(blas_backend::getrf(A.size(1), A.size(0), A.buffer(), A.size(1), ipiv.buffer()), "Lapack call failed.");
        }        
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating LU decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
        }
    }
};


#ifdef __NVCC__
template <typename matrix_type> 
class lu_decomposition<matrix_type, typename std::enable_if<is_dense_matrix<matrix_type>::value && std::is_same<typename traits<matrix_type>::backend_type, cuda_backend>::value, void>::type > 
{
public:
    using value_type = typename std::remove_cv<typename traits<matrix_type>::value_type>::type;
    using backend_type = typename traits<matrix_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using mem_trans = memory::transfer<backend_type, backend_type>;
    using int_type = typename backend_type::int_type;

protected:
    vector<value_type, backend_type> d_work;
    tensor<int, 1, cuda_backend> m_gpu_info;
    tensor<int, 1> m_cpu_info;

protected:
    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, int>::type query_worksize(mat_type& A)
    {
        int_type lwork;
        CALL_AND_HANDLE(cuda_backend::getrf_buffersize(A.size(1), A.size(0), A.buffer(), A.size(1), &lwork), "Failed to query worksize for LU decomposition.");
        return lwork;
    }

public:
    lu_decomposition() : m_gpu_info(1), m_cpu_info(1) {}


    template <typename mat_type, typename mat_typeb>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value && internal::valid_decomposition_matrix<mat_typeb, value_type, backend_type>::value, void>::type 
    operator()(const mat_type& A, mat_typeb& LU, vector<int, backend_type>& ipiv)
    {
        try
        {
	        m_gpu_info.resize(1);
            CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");
            CALL_AND_HANDLE(LU = A, "Failed to copy matrix.");
            size_type lwork;
            CALL_AND_HANDLE(lwork = query_worksize(LU), "Failed to query worksize");
            CALL_AND_HANDLE(d_work.resize(lwork), "Failed to resize workspace array.");
            CALL_AND_HANDLE(cuda_backend::getrf(LU.size(1), LU.size(0), LU.buffer(), LU.size(1), d_work.buffer(), ipiv.buffer(), m_gpu_info.buffer()), "Lapack call failed.");
            m_cpu_info = m_gpu_info;
            CALL_AND_RETHROW(cusolver::getrf_error_handling(m_cpu_info(0), 'a'));
        }        
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating LU decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
        }
    }
    
    template <typename mat_type>
    typename std::enable_if<internal::valid_decomposition_matrix<mat_type, value_type, backend_type>::value, void>::type 
    operator()(mat_type& A, vector<int, backend_type>& ipiv)
    {
        try
        {
	        m_gpu_info.resize(1);
            CALL_AND_HANDLE(internal::lu_result_validation::validate_ipiv(A, ipiv), "Failed to validate pivot array.");
            size_type lwork;
            CALL_AND_HANDLE(lwork = query_worksize(A), "Failed to query worksize");
            CALL_AND_HANDLE(d_work.resize(lwork), "Failed to resize workspace array.");
            CALL_AND_HANDLE(cuda_backend::getrf(A.size(1), A.size(0), A.buffer(), A.size(1), d_work.buffer(), ipiv.buffer(), m_gpu_info.buffer()), "Lapack call failed.");
            m_cpu_info = m_gpu_info;
            CALL_AND_RETHROW(cusolver::getrf_error_handling(m_cpu_info(0), 'a'));
        }        
        catch(const invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating LU decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate LU decomposition.");
        }
    }
};
#endif

}   //namespace linalg

#endif //LINALG_DECOMPOSITIONS_LU_DECOMPOSITION_HPP//

