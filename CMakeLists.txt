cmake_minimum_required(VERSION 3.6) #we require FindBLAS for this to work
project(linalg LANGUAGES CXX)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

option(JUST_INSTALL_LINALG "Don't do anything besides installing the Library" ON)
option(USE_SPARSE_BLAS OFF)
option(WITH_WALL "Compile with '-Wall' C++ compiler flag" OFF)
option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF)
option(USE_GEMM3M "Compile with openmp" OFF)
if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()

add_library(linalg INTERFACE)
target_include_directories(linalg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

option(USE_OPENMP "Compile with openmp" ON)
option(USE_CUDA "Compile with cuda" OFF)

set(LINALG_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "")

if(JUST_INSTALL_LINALG)
    return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    enable_language(CUDA)
    if(COMPILER_SET)
    else()
        set(CMAKE_CUDA_FLAGS "-std=c++11 -Xptxas -O3 -g -expt-extended-lambda ${CMAKE_CUDA_FLAGS}")
        if(USE_OPENMP)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        endif()

        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        set(COMPILER_SET TRUE INTERNAL BOOL "")
    endif()

    set(LINALG_LIBS "cusparse;cublas;cusolver;${LINALG_LIBS}")
else()
    if(COMPILER_SET)
    else()
        if(MSVC)
            message(FATAL_ERROR "Request MSVC compiler.  MSVC is currently not supported.")
        else()
            set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
            if(WITH_WALL)
                set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wall -g -Wextra -Wshadow -pedantic -Wold-style-cast")
            endif()

            if(WITH_WERROR)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
            endif()

            option(CLANG_USE_LIBCPP "Use libc++ for clang compilation" OFF)

            #additional compilation flags for GNU compilers
            if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fwhole-program -fcx-fortran-rules")
                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
                endif()
            #additional compilation flags for inteal compilers
            elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo -O3 -xHost -restrict")
                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
                endif()

            elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo -O3 -xHost")
                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
                endif()

            elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                if(CLANG_USE_LIBCPP)
                    set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
                    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
                endif()
                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
                endif()
            else()
                message(STATUS "${CMAKE_CXX_COMPILER_ID}")
                message(FATAL_ERROR "Compiler not supported.")
            endif()
            

            if(CMAKE_VERSION VERSION_LESS 3.1)
                set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
            else()
              if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD STREQUAL "98")
                set(CMAKE_CXX_STANDARD 11)
              endif()

              if(CMAKE_CXX_STANDARD GREATER 17)
                cmake_minimum_required(VERSION 3.8)
              endif()

              set(CMAKE_CXX_STANDARD_REQUIRED ON)
            endif()
        endif()
        set(COMPILER_SET TRUE INTERNAL BOOL "")
    endif()
endif()

if(DEFINED BLA_VENDOR)
    if(BLA_VENDOR MATCHES "Intel")
        find_package(LAPACK REQUIRED)
        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            set(MKL_INCLUDE_DIRS ${MKLROOT}/include)
            #FIND_PATH(MKL_INCLUDE_DIR "mkl_cblas.h" PATHS ${MKLROOT}/include)
            message(STATUS ${MKL_INCLUDE_DIRS})
            add_compile_definitions(USE_MKL )
            add_compile_definitions(BLAS_HEADER_INCLUDED)
            add_compile_definitions(NOT_ALLOWS_RETURN_TYPE )
            add_compile_definitions(BLAS_NO_TRAILING_UNDERSCORE )
            if(DLA_VENDOR MATCHES "64ilp")
                add_compile_definitons(BLAS_64_BIT)
            endif()
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
            set(USE_SPARSE_BLAS OFF)
        endif()
    elseif(BLA_VENDOR MATCHES "OpenBLAS")   
        add_compile_definitions(USE_OPENBLAS)
        set(USE_DEFAULT_FIND_BLAS ON)
    elseif(BLA_VENDOR MATCHES "FLAME")
        #here we allow for us to handle the multithreaded case.  This is not dealt with by the standard FindBLAS and FindLapack files.  
        #this has been setup to work with the aocl versions of blis and flame
        if(USE_OPENMP)
            #first we attempt to find the standard FLAME LIBRARY
            set(BLA_VENDOR "FLAME")
            include(FindBLAS)
            if(NOT BLAS_FOUND)
                message(FATAL_ERROR "BLAS NOT FOUND")
            endif()
            check_blas_libraries(
              BLAS_LIBRARIES
              BLAS
              sgemm
              ""
              "blis-mt"
              ""
              ""
              ""
              )
            message(STATUS ${BLAS_LIBRARIES})
            set(BLA_VENDOR "FLAME")
            find_package(LAPACK REQUIRED)
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            set(USE_DEFAULT_FIND_BLAS OFF)
        else()
            add_compile_definitions(USE_FLAME)
            set(USE_DEFAULT_FIND_BLAS ON)
        endif()
    else()
        set(USE_DEFAULT_FIND_BLAS ON)
    endif()
else()
    set(USE_DEFAULT_FIND_BLAS ON)
endif()


#if we aren't dealing with one of the vendors we treat specially then we will just go ahead and use the default
#findBLAS and findLAPACK calls
if(USE_DEFAULT_FIND_BLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()


#search for sparse BLAS libraries if requested.  This currently isn't supported but should be in the future
if(USE_SPARSE_BLAS)

endif()


