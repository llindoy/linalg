# linalg

A header library for multidimensional tensor that supports arbitrary rank dense tensor and some sparse matrix types.  This supports a range of linear algebra operations (see linalg/algebra/expressions/overloads for the supported operations) through the use of an expression template based wrapper of BLAS (and cublas for tensors that are stored on CUDA enable GPUs), and through a set of classes wrapping the functionality of various LAPACK routines (see linalg/decompositions and linalg/special_functions for the supported operations).  

Proper documentation and usage examples will be added in future releases.  

At present I can't guarantee that all wrappers to BLAS and LAPACK calls work correctly in all cases, I intend on adding unit tests in a future version.  The functions calls necessary for ML-MCTDH have been tested extensively (and this includes many of the basic BLAS calls, some contractions of rank 3 tensors and the Lapack eigensolvers and singular values decompositions. 
