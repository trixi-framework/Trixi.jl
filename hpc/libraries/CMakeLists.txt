cmake_minimum_required(VERSION 3.24)
project(TrixiLibraries LANGUAGES NONE)

include(ExternalProject)
find_program(MAKE_EXECUTABLE NAMES gmake make REQUIRED)

# The Autotools build gives p4est 2.8 and its libsc distinct library versions.
ExternalProject_Add(p4est
    URL https://p4est.github.io/release/p4est-2.8.tar.gz
    URL_HASH SHA256=6a0586e3abac06c20e31b1018f3a82a564a6a0d9ff6b7f6c772a9e6b0f0cc5e4
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    TLS_VERIFY TRUE
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/p4est"
    CONFIGURE_COMMAND "${CMAKE_COMMAND}" -E env
        "CC=$ENV{EBROOTOPENMPI}/bin/mpicc" "CXX=$ENV{EBROOTOPENMPI}/bin/mpicxx"
        "FC=$ENV{EBROOTOPENMPI}/bin/mpifort" "F77=$ENV{EBROOTOPENMPI}/bin/mpifort"
        "CFLAGS=-O2 -fPIC" "CXXFLAGS=-O2 -fPIC"
        "CPPFLAGS=-I$ENV{EBROOTZLIB}/include"
        "LDFLAGS=-L$ENV{EBROOTZLIB}/lib -Wl,-rpath,$ENV{EBROOTZLIB}/lib -Wl,-rpath,$ENV{EBROOTOPENMPI}/lib"
        <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
        --enable-mpi --enable-shared --disable-static --without-blas
    BUILD_COMMAND "${MAKE_EXECUTABLE}" -j$ENV{CMAKE_BUILD_PARALLEL_LEVEL}
    INSTALL_COMMAND "${MAKE_EXECUTABLE}" install
)

# Full clones keep pinned commits available even after upstream branches advance.
function(t8code_dependency name repository revision)
    ExternalProject_Add(${name}
        GIT_REPOSITORY "${repository}"
        GIT_TAG "${revision}"
        GIT_SUBMODULES ""
        GIT_PROGRESS TRUE
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/sources/${name}"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
endfunction()

t8code_dependency(sc https://github.com/cburstedde/libsc.git
    692841759d9628f15c8eb9b65036aa6a837f1d26)
t8code_dependency(p4est_t8 https://github.com/cburstedde/p4est.git
    2296a990d8b6b54731a63be0ba5bc17b08cd1f3d)
t8code_dependency(googletest https://github.com/DLR-SC/googletest_mpi.git
    98bfff426b057400268a00f97677d749a9e25096)

ExternalProject_Add(t8code
    GIT_REPOSITORY https://github.com/DLR-AMR/t8code.git
    GIT_TAG bfc4e39f2286248ce0bfbe719b0a2d2c9dcac112
    GIT_SUBMODULES ""
    GIT_PROGRESS TRUE
    UPDATE_COMMAND ""
    DEPENDS sc p4est_t8 googletest
    INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/t8code"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR:STRING=lib
        -DCMAKE_C_COMPILER:FILEPATH=$ENV{EBROOTOPENMPI}/bin/mpicc
        -DCMAKE_CXX_COMPILER:FILEPATH=$ENV{EBROOTOPENMPI}/bin/mpicxx
        -DMPI_C_COMPILER:FILEPATH=$ENV{EBROOTOPENMPI}/bin/mpicc
        -DMPIEXEC_EXECUTABLE:FILEPATH=$ENV{EBROOTOPENMPI}/bin/mpiexec
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_CXX_FLAGS:STRING=-std=c++20
        -DFETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON
        -DFETCHCONTENT_SOURCE_DIR_SC:PATH=${CMAKE_CURRENT_BINARY_DIR}/sources/sc
        -DFETCHCONTENT_SOURCE_DIR_P4EST:PATH=${CMAKE_CURRENT_BINARY_DIR}/sources/p4est_t8
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST:PATH=${CMAKE_CURRENT_BINARY_DIR}/sources/googletest
        -DBUILD_TESTING:BOOL=OFF
        -DP4EST_BUILD_TESTING:BOOL=OFF
        -DSC_BUILD_TESTING:BOOL=OFF
        -DINSTALL_GTEST:BOOL=OFF
        -DT8CODE_BUILD_BENCHMARKS:BOOL=OFF
        -DT8CODE_BUILD_DOCUMENTATION:BOOL=OFF
        -DT8CODE_BUILD_EXAMPLES:BOOL=OFF
        -DT8CODE_BUILD_FORTRAN_INTERFACE:BOOL=OFF
        -DT8CODE_BUILD_TESTS:BOOL=OFF
        -DT8CODE_BUILD_TUTORIALS:BOOL=OFF
        -DT8CODE_ENABLE_MPI:BOOL=ON
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel $ENV{CMAKE_BUILD_PARALLEL_LEVEL}
)
