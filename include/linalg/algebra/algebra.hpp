#ifndef LINALG_DENSE_ALGEBRA_HPP
#define LINALG_DENSE_ALGEBRA_HPP

//include all the files required to implement the various expression template based operations supported by the code.
#include "../linalg_forward_decl.hpp"
#include "expressions/elemental/expression.hpp"
#include "expressions/elemental/applicative/addition_operation.hpp"
#include "expressions/elemental/applicative/scalar_multiplication_operation.hpp"
#include "expressions/elemental/applicative/complex_conjugation_operation.hpp"
#include "expressions/elemental/applicative/exponential_operation.hpp"
#include "expressions/elemental/applicative/hadamard.hpp"
#include "expressions/elemental/applicative/complex.hpp"
          
#include "expressions/permutations/transpose_expression.hpp"
#include "expressions/permutations/tensor_transpose.hpp"
//#include "expressions/permutations/tensor_permutation.hpp"

#include "expressions/contractions/dense_matrix_vector_multiplication.hpp"
#include "expressions/contractions/csr_matrix_vector_multiplication.hpp"
#include "expressions/contractions/diagonal_matrix_vector_multiplication.hpp"
#include "expressions/contractions/tensor_dot.hpp"

#include "expressions/contractions/dense_dense_matrix_multiplication.hpp"
#include "expressions/contractions/csr_matrix_dense_matrix_multiplication.hpp"
#include "expressions/contractions/diagonal_matrix_dense_matrix_multiplication.hpp"

#include "expressions/contractions/contraction_1_mt.hpp"
#include "expressions/contractions/contraction_332.hpp"

#include "overloads/expression_type_aliases.hpp"
#include "overloads/tensor_addition.hpp"
#include "overloads/hadamard.hpp"
#include "overloads/complex.hpp"
#include "overloads/exponential.hpp"
#include "overloads/real_imag.hpp"
#include "overloads/scalar_multiplication.hpp"
#include "overloads/conjugation.hpp"
#include "overloads/transposition.hpp"

//#include "overloads/tensor_permutation.hpp"
#include "overloads/matrix_vector_product.hpp"
#include "overloads/matrix_matrix_product.hpp"
#include "overloads/tensor_contractions.hpp"
#include "overloads/dot_product.hpp"
#include "overloads/trace.hpp"

#endif  //LINALG_DENSE_ALGEBRA_HPP//

