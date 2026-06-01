FROM ubuntu:26.04

# Generic
RUN apt-get update -y && \
    apt-get install -y git

## Pure C++ parts

# Basix C++ dependencies
RUN apt-get install -y build-essential cmake ninja-build libopenblas-dev

# Basix C++ build
RUN git clone --depth 1 https://github.com/FEniCS/basix.git && \
    cd basix/cpp && \
    cmake -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir --prefix ~/fenics

# UFCx header - interface for FFCx generated code
RUN git clone --depth 1 https://github.com/FEniCS/ffcx.git && \
    cd ffcx/cmake && \
    cmake -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir --prefix ~/fenics

# DOLFINx C++ dependencies
# Add libpetsc-real-dev for PETSc
RUN apt-get install -y libboost-dev libhdf5-mpi-dev libopenmpi-dev libpugixml-dev libptscotch-dev libspdlog-dev pkg-config

# DOLFINx C++ build
RUN git clone --depth 1 https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx/cpp && \
    cmake -DDOLFINX_UFCX_PYTHON=OFF -Dufcx_DIR=~/fenics -DDOLFINX_BASIX_PYTHON=OFF -DBasix_DIR=~/fenics -DCMAKE_INSTALL_PREFIX=~/fenics -B build-dir -S . && \
    cmake --build build-dir && \
    cmake --install build-dir --prefix ~/fenics

## Python wrappers to Basix

# System Python dependencies - pypi.org for the rest 
# Add python3-petsc4py for petsc4py
RUN apt-get install -y python3 python3-pip python3-venv && \
    python3 -m venv ~/fenics/ # create virtual environment

# Basix Python wrapper
RUN . ~/fenics/bin/activate && \
    cd basix/python && \
    python -m pip install .

## Pure Python parts

# UFL and FFCx install
RUN . ~/fenics/bin/activate && \
    python -m pip install git+https://github.com/FEniCS/ufl.git && \ 
    python -m pip install git+https://github.com/FEniCS/ffcx.git

## And finally...

# Python wrapper to DOLFINx
# NOTE: Remove mpipy for system petsc4py build
RUN . ~/fenics/bin/activate && \
    cd dolfinx/python && \
    python -m pip install scikit-build-core[pyproject] nanobind mpi4py && \
    python -m pip install --no-build-isolation --check-build-dependencies .
