#ifndef LINALG_ALGEBRA_TRANSPOSE_EXPRESSION_HPP
#define LINALG_ALGEBRA_TRANSPOSE_EXPRESSION_HPP

#include "../../../linalg_forward_decl.hpp"
#include <array>

namespace linalg
{

namespace expression_templates
{

template <typename _mat_type, bool conjugate>
class transpose_expression<_mat_type, conjugate, 
                            typename std::enable_if<is_dense_matrix<_mat_type>::value, void>::type> 
    : public expression_base<transpose_expression<_mat_type, conjugate, void>, false>, public dense_type
{
public: 
    using base_type = expression_base<transpose_expression<_mat_type, conjugate, void>, false>;
    using value_type = typename std::remove_cv<typename traits<_mat_type>::value_type>::type;
    using backend_type = typename traits<_mat_type>::backend_type;
    using matrix_type = const _mat_type&;
    using size_type = typename backend_type::size_type;
    using shape_type = std::array<size_type, 2>;

protected:
    matrix_type m_arr;
    value_type m_alpha;

public:
    transpose_expression() = delete;
    transpose_expression(matrix_type arr, value_type alpha = value_type(1.0)) 
    try : base_type(shape_type{{arr.shape(1), arr.shape(0)}}), m_arr(arr), m_alpha(alpha){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct transpose expression object.");
    }

    matrix_type matrix() const{return m_arr;}
    value_type coeff() const{return m_alpha;}

    //function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
    template <typename array_type>
    typename std::enable_if<is_dense_matrix<array_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type applicative(array_type& res, value_type beta = value_type(0.0)) const
    {
        try
        {
            if(res.buffer() == m_arr.buffer()){ASSERT(m_arr.shape(0) == m_arr.shape(1), "Inplace matrix transpose is only supported for square matrices when using the blas backend.");}
            CALL_AND_HANDLE(backend_type::transpose(conjugate, m_arr.shape(0), m_arr.shape(1), m_alpha, m_arr.buffer(), beta, res.buffer()), "Call to transpose failed.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate matrix transpose expression.");
        }
    }

    template <typename array_type>
    typename std::enable_if<is_dense_matrix<array_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type addition_applicative(array_type& res) const
    {
        CALL_AND_RETHROW(applicative(res, 1.0));
    }

    template <typename array_type>
    typename std::enable_if<is_dense_matrix<array_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type subtraction_applicative(array_type& res) const
    {
        CALL_AND_RETHROW(applicative(res, -1.0));
    }
};



template <typename _mat_type, bool conjugate>
class transpose_expression<_mat_type, conjugate, 
                            typename std::enable_if<is_diagonal_matrix_type<_mat_type>::value, void>::type> 
    : public expression_base<transpose_expression<_mat_type, conjugate, void>, false>, public diagonal_matrix_type
{
public: 
    using base_type = expression_base<transpose_expression<_mat_type, conjugate, void>, false>;
    using value_type = typename traits<_mat_type>::value_type;
    using backend_type = typename traits<_mat_type>::backend_type;
    using matrix_type = const _mat_type&;
    using size_type = typename backend_type::size_type;
    using shape_type = std::array<size_type, 2>;

protected:
    matrix_type m_arr;
    value_type m_alpha;

public:
    transpose_expression() = delete;
    transpose_expression(matrix_type arr, value_type alpha = value_type(1.0)) 
    try : base_type(shape_type{{arr.shape(1), arr.shape(0)}}), m_arr(arr), m_alpha(alpha){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct transpose expression object.");
    }

    matrix_type matrix() const{return m_arr;}
    value_type coeff() const{return m_alpha;}

    //function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
    template <typename array_type>
    typename std::enable_if<is_diagonal_matrix_type<array_type>::value, void>::type applicative(array_type& res) const
    {
        try
        {
            size_type nnz = m_arr.nnz();
            if(res.buffer() != m_arr.buffer())
            {
                CALL_AND_HANDLE(backend_type::vector_scalar_product(nnz, m_alpha, m_arr.buffer(), m_arr.incx(), res.buffer(), res.incx()), "Error when calling vector scalar product.");
            }
            else
            {
                if(m_alpha != value_type(1.0)){CALL_AND_HANDLE(backend_type::scal(nnz, m_alpha, res.buffer(),res.incx()), "Error when calling vector scalar product.");}
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate diagonal matrix transpose expression.");
        }
    }
};


template <typename _mat_type, bool conjugate>
class transpose_expression<_mat_type, conjugate, typename std::enable_if<is_symtridiag_matrix_type<_mat_type>::value, void>::type> : public expression_base<transpose_expression<_mat_type, conjugate, void>, false>, public symmetric_tridiagonal_matrix_type
{
public: 
    using base_type = expression_base<transpose_expression<_mat_type, conjugate, void>, false>;
    using value_type = typename traits<_mat_type>::value_type;
    using backend_type = typename traits<_mat_type>::backend_type;
    using matrix_type = const _mat_type&;
    using size_type = typename backend_type::size_type;
    using shape_type = std::array<size_type, 2>;

protected:
    matrix_type m_arr;
    value_type m_alpha;

public:
    transpose_expression() = delete;
    transpose_expression(matrix_type arr, value_type alpha = value_type(1.0)) 
    try : base_type(shape_type{{arr.shape(1), arr.shape(0)}}), m_arr(arr), m_alpha(alpha){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct transpose expression object.");
    }

    matrix_type matrix() const{return m_arr;}
    value_type coeff() const{return m_alpha;}

    //function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
    template <typename array_type>
    typename std::enable_if<is_symtridiag_matrix_type<array_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type applicative(array_type& res) const
    {
        try
        {
            size_type nnz = m_arr.nnz();
            if(m_alpha != value_type(1.0))
            {
                if(res.buffer() != m_arr.buffer())
                {
                    CALL_AND_HANDLE(backend_type::vector_scalar_product(nnz, m_alpha, m_arr.buffer(), 1, res.buffer(), 1), "Failed to evaluate matrix transpose for symmetric tridiagonal matrices.  Error when calling vector scalar product.");
                }
                else{CALL_AND_HANDLE(backend_type::scal(nnz, m_alpha, res.buffer(), 1), "Failed to evaluate matrix transpose for symmetric tridiagonal matrices.  Error when calling vector scalar product.");}
            }
            else
            {
                if(res.buffer() != m_arr.buffer())
                {
                    CALL_AND_HANDLE(backend_type::copy(m_arr.buffer(), nnz, res.buffer()), "Failed to evaluate matrix transpose for symmetric tridiagonal matrices.  Error when calling copying buffer.");
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to symmetric diagonal matrix transpose expression.");
        }
    }
};


template <typename _mat_type, bool conjugate>
class transpose_expression<_mat_type, conjugate, typename std::enable_if<is_csr_matrix_type<_mat_type>::value, void>::type> : public expression_base<transpose_expression<_mat_type, conjugate, void>, false>, public csr_matrix_type
{
public: 
    using base_type = expression_base<transpose_expression<_mat_type, conjugate, void>, false>;
    using value_type = typename traits<_mat_type>::value_type;
    using backend_type = typename traits<_mat_type>::backend_type;
    using matrix_type = const _mat_type&;

protected:
    matrix_type m_arr;
    value_type m_alpha;

public:
    transpose_expression() = delete;
    transpose_expression(matrix_type arr, value_type alpha = value_type(1.0)) 
    try : base_type(arr.topology()), m_arr(arr), m_alpha(alpha){}
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct transpose expression object.");
    }

    matrix_type matrix() const{return m_arr;}
    value_type coeff() const{return m_alpha;}

    //function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
    template <typename array_type>
    typename std::enable_if<is_csr_matrix_type<_mat_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type applicative(array_type& /* res */) const{RAISE_EXCEPTION("matrix transpose is currently not supported for this sparse matrix type.");}

    template <typename array_type>
    typename std::enable_if<is_dense_matrix<array_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type addition_applicative(array_type& /* res */) const{RAISE_EXCEPTION("matrix transpose is currently not supported for this sparse matrix type.")}

    template <typename array_type>
    typename std::enable_if<is_dense_matrix<array_type>::value && compatible_traits<_mat_type, array_type>::value, void>::type subtraction_applicative(array_type& /* res */) const{RAISE_EXCEPTION("matrix transpose is currently not supported for this sparse matrix type.")}
};

}   //namespace expression_templates
template <typename _mat_type, bool conjugate>
struct traits<expression_templates::transpose_expression<_mat_type, conjugate, void> >
{
    using value_type = typename traits<_mat_type>::value_type;
    using backend_type = typename traits<_mat_type>::backend_type;
    using shape_type = typename expression_templates::result_type<_mat_type>::shape_type;
    using const_shape_reference = typename expression_templates::result_type<_mat_type>::const_shape_reference;
    static constexpr size_t rank = 2;
};
}   //namespace linalg

#endif  //LINALG_ALGEBRA_TRANSPOSE_EXPRESSION_HPP//

