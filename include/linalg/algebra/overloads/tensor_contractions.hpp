#ifndef LINALG_ALGEBRA_TENSOR_CONTRACTION_OVERLOADS_DENSE_HPP
#define LINALG_ALGEBRA_TENSOR_CONTRACTION_OVERLOADS_DENSE_HPP


namespace linalg
{

//contraction rank 2 - rank 3
template <typename T1, typename T2, typename I1, typename I2>
mtc1_return_type<T1, T2> contract(const T1& l, I1 il, const T2& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = mtc1_return_type<T1, T2>;  using value_type = typename rettype::value_type;
    return rettype(static_cast<value_type>(1.0), l, r, il, ir);
}


//contraction rank 3 transpose of rank 2
template <typename T1, typename T2, bool conjugate, typename I1, typename I2>
mtc1_return_type<T1, T2> contract(const T1& l, I1 il, const trans_type<T2, conjugate>& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = mtc1_return_type<T1, T2>;
    CALL_AND_RETHROW(return rettype(r.coeff(), l, r.matrix(), il, (ir == 0 ? 1 : 0), false, conjugate ));
}

template <typename T1, typename T2, bool conjugate, typename I1, typename I2>
mtc1_return_type<T1, T2> contract(const trans_type<T1, conjugate>& l, I1 il, const T2& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = mtc1_return_type<T1, T2>;
    CALL_AND_RETHROW(return rettype(l.coeff(), r, l.matrix(), ir, (il == 0 ? 1 : 0), false, conjugate));
}


//contraction conj rank 3 - rank 2
template <typename T1, typename T2, typename I1, typename I2>
mtc1_return_type<T1, T2> contract(const T1& l, I1 il, const conj_type<T2>& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = mtc1_return_type<T1, T2>;  using value_type = typename rettype::value_type;
    CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l, r.obj(), il, ir, false, true));
}

template <typename T1, typename T2, typename I1, typename I2>
mtc1_return_type<T1, T2> contract(const conj_type<T1>& l, I1 il, const T2& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = mtc1_return_type<T1, T2>;  using value_type = typename rettype::value_type;
    CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l.obj(), r, il, ir, true, false));
}

//contraction conj rank 3 - conj rank 2
template <typename T1, typename T2, typename I1, typename I2>
mtc1_return_type<T1, T2> contract(const conj_type<T1>& l, I1 il, const conj_type<T2>& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = mtc1_return_type<T1, T2>;  using value_type = typename rettype::value_type;
    CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l.obj(), r.obj(), il, ir, true, true));
}


/*
 *  Contractions involving scalar multiples of tensors
 */

//contraction rank 3 - scalar*rank 2
template <typename T1, typename T2, typename T3, typename I1, typename I2>
scal_mtc1_return_type<T1, T2, T3> contract(const scal_type<T1, T2>& l, I1 il, const T3& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = scal_mtc1_return_type<T1, T2, T3>;  using value_type = typename rettype::value_type;
    CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.left()), l.right(), r, il, ir));
}

template <typename T1, typename T2, typename T3, typename I1, typename I2>
scal_mtc1_return_type<T1, T2, T3> contract(const T2& l, I1 il, const scal_type<T1, T3>& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = scal_mtc1_return_type<T1, T2, T3>;  using value_type = typename rettype::value_type;
    CALL_AND_RETHROW(return rettype(static_cast<value_type>(r.left()), l, r.right(), il, ir));
}

//contraction scalar*rank 3 - scalar*rank 2
template <typename T1, typename T2, typename T3, typename T4, typename I1, typename I2>
scal_scal_mtc1_return_type<T1, T2, T3, T4> contract(const scal_type<T1, T2>& l, I1 il, const scal_type<T3,T4>& r, I2 ir)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value, "Failed to instantiate contraction indices are not integral.");
    using rettype = scal_scal_mtc1_return_type<T1, T2, T3, T4>;  using value_type = typename rettype::value_type;
    CALL_AND_RETHROW(return rettype(static_cast<value_type>(l.left()*r.left()), l.right(), r.right(), il, ir));
}




//contraction of rank 3 tensor with rank 3 tensor to form a rank 2 tensor
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4>
ttc2_return_type<T1, T2> contract(const T1& l, I1 il, I2 jl, const T2& r, I3 ir, I4 jr)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");

    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    size_type uncontracted_index = 3 - (il+jl);

    CALL_AND_RETHROW(return rettype(static_cast<value_type>(1.0), l, r, uncontracted_index, false, false));
}

//
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename working_array>
ttc2_return_type<T1, T2> contract(const T1& l, I1 il, I2 jl, const T2& r, I3 ir, I4 jr, working_array& working)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");

    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    size_type uncontracted_index = 3 - (il+jl);

    rettype ret_op;
    try
    {
        ret_op = rettype(static_cast<value_type>(1.0), l, r, uncontracted_index);
        ret_op.bind_working(working);
    }
    catch(...){throw;}
    return ret_op;
}


//contraction rank 3 - conj rank 3
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename conj_array>
ttc2_return_type<T1, T2> contract(const T1& l, I1 il, I2 jl, const conj_type<T2>& r, I3 ir, I4 jr, conj_array& conj)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");

    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    size_type uncontracted_index = 3 - (il+jl);

    rettype ret_op;
    try
    {
        ret_op = rettype(static_cast<value_type>(1.0), l, r.obj(), uncontracted_index, false, true);
        ret_op.bind_conjugate_workspace(conj);
    }
    catch(...){throw;}
    return ret_op;
}
//
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename conj_array, typename working_array>
ttc2_return_type<T1, T2> contract(const T1& l, I1 il, I2 jl, const conj_type<T2>& r, I3 ir, I4 jr, conj_array& conj, working_array& working)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");

    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    size_type uncontracted_index = 3 - (il+jl);

    try
    {
        rettype ret_op(static_cast<value_type>(1.0), l, r.obj(), uncontracted_index, false, true);
        ret_op.bind_conjugate_workspace(conj);
        ret_op.bind_working(working);
        return ret_op;
    }
    catch(...){throw;}
}

//contraction conj rank 3 - rank 3
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename conj_array>
ttc2_return_type<T1, T2> contract(const conj_type<T1>& l, I1 il, I2 jl, const T2& r, I3 ir, I4 jr, conj_array& conj)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");

    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    size_type uncontracted_index = 3 - (il+jl);

   
    try
    {
        rettype ret_op(static_cast<value_type>(1.0), l.obj(), r, uncontracted_index, true, false);
        ret_op.bind_conjugate_workspace(conj);
        return ret_op;
    }
    catch(...){throw;}
}
//
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename conj_array, typename working_array>
ttc2_return_type<T1, T2> contract(const conj_type<T1>& l, I1 il, I2 jl, const T2& r, I3 ir, I4 jr, conj_array& conj, working_array& working)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");

    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    size_type uncontracted_index = 3 - (il+jl);

    try
    {
        rettype ret_op(static_cast<value_type>(1.0), l.obj(), r, uncontracted_index, true, false);
        ret_op.bind_conjugate_workspace(conj);
        ret_op.bind_working(working);
        return ret_op;
    }
    catch(...){throw;}
}

//contraction conj rank 3 - conj rank 3
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename conj_array>
ttc2_return_type<T1, T2> contract(const conj_type<T1>& l, I1 il, I2 jl, const conj_type<T2>& r, I3 ir, I4 jr, conj_array& conj)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");
    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;

    size_type uncontracted_index = 3 - (il+jl);

    try
    {
        rettype ret_op(static_cast<value_type>(1.0), l.obj(), r.obj(), uncontracted_index, true, true);
        ret_op.bind_conjugate_workspace(conj);
        return ret_op;
    }
    catch(...){throw;}
}
//
template <typename T1, typename T2, typename I1, typename I2, typename I3, typename I4, typename conj_array, typename working_array>
ttc2_return_type<T1, T2> contract(const conj_type<T1>& l, I1 il, I2 jl, const conj_type<T2>& r, I3 ir, I4 jr, conj_array& conj, working_array& working)
{
    static_assert(std::is_integral<I1>::value && std::is_integral<I2>::value && std::is_integral<I3>::value && std::is_integral<I4>::value, "Failed to instantiate contraction indices are not integral.");
    ASSERT(il == ir && jl == jr, "Invalid tensor contraction.  Currently contractions of two rank 3 tensors to form a rank 2 tensor require contractions over the same indices.");
    ASSERT(il != jl, "Invalid tensor contraction.  Cannot contract over the same index twice.");
    using rettype = ttc2_return_type<T1, T2>;  using value_type = typename rettype::value_type; using size_type = typename rettype::size_type;
    
    size_type uncontracted_index = 3 - (il+jl);

    try
    {
        rettype ret_op(static_cast<value_type>(1.0), l.obj(), r.obj(), uncontracted_index, true, true);
        ret_op.bind_conjugate_workspace(conj);
        ret_op.bind_working(working);
        return ret_op;
    }
    catch(...){throw;}
}

}   //namespace linalg


#endif  //LINALG_ALGEBRA_TENSOR_CONTRACTION_OVERLOADS_DENSE_HPP//

