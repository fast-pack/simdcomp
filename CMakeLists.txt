cmake_minimum_required(VERSION 3.14)

project(simdcomp
  VERSION 0.0.3
  DESCRIPTION "SIMD integer compression (bit packing) for x86 SSE/AVX and ARM NEON"
  LANGUAGES C)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SIMDCOMP_BUILD_TESTS "Build the unit tests" ON)
option(SIMDCOMP_BUILD_EXAMPLES "Build the example program" ON)
option(SIMDCOMP_BUILD_BENCHMARKS
       "Build the benchmarks (pulls in lemire/counters, needs C++17)" ON)
option(SIMDCOMP_NATIVE
       "Tune for the building machine (-march=native); enables AVX2/AVX-512 on \
capable x86 hosts" ON)

# Default to an optimized build when the user did not pick one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# The simdcomp library
#
# The core kernels are 128-bit: x86 SSE on Intel/AMD, our own ARM NEON shim
# (include/neon128.h) on AArch64/ARM. The avx*bitpacking.c files are wholly
# guarded behind __AVX2__/__AVX512F__ and contribute nothing unless those are
# enabled (e.g. via -march=native on a capable x86 host).
# ---------------------------------------------------------------------------
set(SIMDCOMP_SOURCES
  src/simdbitpacking.c
  src/simdintegratedbitpacking.c
  src/simdcomputil.c
  src/simdpackedsearch.c
  src/simdpackedselect.c
  src/simdfor.c
  src/avxbitpacking.c
  src/avx512bitpacking.c)

add_library(simdcomp ${SIMDCOMP_SOURCES})
add_library(simdcomp::simdcomp ALIAS simdcomp)

target_include_directories(simdcomp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# The sources are C89; keep that contract but allow compiler extensions, which
# are required for the intrinsics headers (<arm_neon.h>, __attribute__, ...).
set_target_properties(simdcomp PROPERTIES
  C_STANDARD 90
  C_EXTENSIONS ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON)

# -march=native (when requested and supported by the compiler).
include(CheckCCompilerFlag)
set(SIMDCOMP_HAS_MARCH_NATIVE FALSE)
if(SIMDCOMP_NATIVE)
  check_c_compiler_flag("-march=native" SIMDCOMP_HAS_MARCH_NATIVE)
  if(SIMDCOMP_HAS_MARCH_NATIVE)
    target_compile_options(simdcomp PRIVATE -march=native)
  endif()
endif()

# Warnings, mirroring the previous Makefile, on GCC/Clang only.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(simdcomp PRIVATE -Wall -Wextra -Wshadow)
endif()

# A small helper so every consumer below gets -march=native consistently.
function(simdcomp_apply_native target)
  if(SIMDCOMP_NATIVE AND SIMDCOMP_HAS_MARCH_NATIVE)
    target_compile_options(${target} PRIVATE -march=native)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(SIMDCOMP_BUILD_TESTS)
  enable_testing()

  add_executable(unit tests/unit.c)
  target_link_libraries(unit PRIVATE simdcomp)
  simdcomp_apply_native(unit)
  add_test(NAME unit COMMAND unit)

  add_executable(unit_chars tests/unit_chars.c)
  target_link_libraries(unit_chars PRIVATE simdcomp)
  simdcomp_apply_native(unit_chars)
  add_test(NAME unit_chars COMMAND unit_chars)
endif()

# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if(SIMDCOMP_BUILD_EXAMPLES)
  add_executable(example example/example.c)
  target_link_libraries(example PRIVATE simdcomp)
  simdcomp_apply_native(example)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
#
# bitpackingbenchmark uses lemire/counters (C++17) to report CPU cycles per
# integer via hardware performance counters where available. benchmark.c is the
# legacy clock()-based throughput benchmark, kept in C.
# ---------------------------------------------------------------------------
if(SIMDCOMP_BUILD_BENCHMARKS)
  enable_language(CXX)

  include(FetchContent)
  # Do not let counters add its own install/export rules into our project.
  set(COUNTERS_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(counters
    GIT_REPOSITORY https://github.com/lemire/counters.git
    GIT_TAG v4.1.0)
  FetchContent_MakeAvailable(counters)

  add_executable(bitpackingbenchmark benchmarks/bitpackingbenchmark.cpp)
  target_link_libraries(bitpackingbenchmark PRIVATE simdcomp counters::counters)
  set_target_properties(bitpackingbenchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  simdcomp_apply_native(bitpackingbenchmark)

  add_executable(benchmark benchmarks/benchmark.c)
  target_link_libraries(benchmark PRIVATE simdcomp)
  simdcomp_apply_native(benchmark)
endif()

# ---------------------------------------------------------------------------
# Install
#
# Installs the library, the public headers, and a CMake package so downstream
# projects can simply:
#
#     find_package(simdcomp REQUIRED)
#     target_link_libraries(myapp PRIVATE simdcomp::simdcomp)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(SIMDCOMP_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/simdcomp)

install(TARGETS simdcomp
  EXPORT simdcompTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h")

install(EXPORT simdcompTargets
  FILE simdcompTargets.cmake
  NAMESPACE simdcomp::
  DESTINATION ${SIMDCOMP_INSTALL_CMAKEDIR})

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/simdcompConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/simdcompConfig.cmake
  INSTALL_DESTINATION ${SIMDCOMP_INSTALL_CMAKEDIR})

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/simdcompConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/simdcompConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/simdcompConfigVersion.cmake
  DESTINATION ${SIMDCOMP_INSTALL_CMAKEDIR})

# Also generate a pkg-config file for non-CMake consumers.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/simdcomp.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/simdcomp.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/simdcomp.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
