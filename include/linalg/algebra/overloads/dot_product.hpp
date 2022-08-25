#ifndef LINALG_ALGEBRA_OVERLOADS_DOT_PRODUCT_HPP
#define LINALG_ALGEBRA_OVERLOADS_DOT_PRODUCT_HPP

namespace linalg
{

namespace internal
{
template <typename T1, typename T2>
struct is_valid_dot_product : 
    public std::conditional
    < 
        is_dense_tensor<T1>::value && is_dense_tensor<T2>::value,
        typename std::conditional
        <   
            traits<T1>::rank == 1 && traits<T2>::rank == 1 &&
            compatible_traits<T1, T2>::value,
            std::true_type,
            std::false_type
        >::type,
        typename std::conditional
        <
            is_diagonal_matrix_type<T1>::value && is_diagonal_matrix_type<T2>::value && 
            compatible_traits<T1, T2>::value,
            std::true_type,
            std::false_type
        >::type
    >::type
{};
}   //namespace internal

//dot product of two arrays
template <typename T1, typename T2>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const T1& a, const T2& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    
    ASSERT(a.size() == b.size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(false, a.size(), a.buffer(), a.incx(), b.buffer(), b.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return val;
}

//dot product of conjugate expressions
template <typename T1, typename T2>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const conj_type<T1>& a, const T2& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    
    ASSERT(a.obj().size() == b.size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(true, b.size(), a.obj().buffer(), a.obj().incx(), b.buffer(), b.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return val;
}

template <typename T1, typename T2>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const T1& a, const conj_type<T2>& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    
    ASSERT(a.size() == b.obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(true, a.size(), b.obj().buffer(), b.obj().incx(), a.buffer(), a.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return val;
}

template <typename T1, typename T2>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, typename traits<T1>::value_type>::type dot_product(const conj_type<T1>& a, const conj_type<T2>& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    using std::conj;
    
    ASSERT(a.obj().size() == b.obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(false, a.obj().size(), a.obj().buffer(), a.obj().incx(), b.obj().buffer(), b.obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return conj(val);
}

//dot product of scalar times conjugate types
template <typename T1, typename T2, typename T3>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type()*T3())>::type dot_product(const scalconj_type<T3, T1>& a, const T2& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    
    ASSERT(a.right().obj().size() == b.size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(true, b.size(), a.right().obj().buffer(), a.right().obj().incx(), b.buffer(), b.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return val*a.left()();
}

template <typename T1, typename T2, typename T3>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type()*T3())>::type dot_product(const T1& a, const scalconj_type<T3, T2>& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    
    ASSERT(a.size() == b.right().obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(false, a.size(), b.right().obj().buffer(), b.right().obj().incx(), a.buffer(), a.incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return val*b.left()();
}

template <typename T1, typename T2, typename T3, typename T4>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type()*T3()*T4())>::type dot_product(const scalconj_type<T3, T1>& a, const scalconj_type<T4, T2>& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    using std::conj;
    
    ASSERT(a.right().obj().size() == b.right().obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(false, a.right().obj().size(), a.right().obj().buffer(), a.right().obj().incx(), b.right().obj().buffer(), b.right().obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return conj(val)*a.left()()*b.right()();
}

//dot product of scalar*conj and conjugate types
template <typename T1, typename T2, typename T3>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type()*T3())>::type dot_product(const scalconj_type<T3, T1>& a, const conj_type<T2>& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    using std::conj;
    
    ASSERT(a.right().obj().size() == b.obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(false, b.obj().size(), a.right().obj().buffer(), a.right().obj().incx(), b.obj().buffer(), b.obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return conj(val)*a.left()();
}

template <typename T1, typename T2, typename T3>
typename std::enable_if<internal::is_valid_dot_product<T1, T2>::value, decltype(typename traits<T1>::value_type()*T3())>::type dot_product(const conj_type<T1>& a, const scalconj_type<T3, T2>& b)
{
    using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
    using backend_type = typename traits<T1>::backend_type;
    using std::conj;
    
    ASSERT(a.obj().size() == b.right().obj().size(), "Failed to evaluate dot product between two arrays.  The two arrays do not have the same size.");
    value_type val; CALL_AND_HANDLE(val = backend_type::dot(false, a.obj().size(), a.obj().buffer(), a.obj().incx(), b.right().obj().buffer(), b.right().obj().incx()), "Failed to evaluate dot product between two arrays.  Failed to call the dot routine.");
    return conj(val)*b.left()();
}

}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_SCALAR_DOT_PRODUCT_HPP//


