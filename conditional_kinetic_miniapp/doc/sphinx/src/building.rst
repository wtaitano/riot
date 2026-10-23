..
   This file was made in part with generative AI.

Building
========

Requirements
------------

* A C++20 compiler. The tensor-train branch of Parthenon uses concepts.
* CMake 3.26 or newer.
* HDF5. It is required, and there is no option to build without it. With MPI on (the
  default), HDF5 must be a parallel build.
* MPI (optional, on by default).

Fetching the submodules
-----------------------

The ckm submodules (Parthenon and Catch2) are marked ``update = none`` in
``.gitmodules``, so ``git clone --recursive`` of riot skips them. Fetch them from
anywhere in the repository:

.. code-block:: bash

   conditional_kinetic_miniapp/script/setup.sh

Configuring and building
------------------------

ckm is a standalone CMake project. Always configure it from its own directory, never
from the riot top level:

.. code-block:: bash

   cmake -S conditional_kinetic_miniapp -B build-ckm -DCKM_ENABLE_UNIT_TESTS=ON
   cmake --build build-ckm -j
   ctest --test-dir build-ckm --output-on-failure

For a serial build, add ``-DCKM_ENABLE_MPI=OFF`` and use a serial HDF5.

CMake options
-------------

=========================  =======  =====================================================
Option                     Default  Meaning
=========================  =======  =====================================================
``CKM_ENABLE_MPI``         ``ON``   Build with MPI (requires parallel HDF5)
``CKM_ENABLE_UNIT_TESTS``  ``OFF``  Build the Catch2 unit tests and the driver run test
``CKM_BUILD_CATCH2``       ``ON``   Use the bundled Catch2. ``OFF`` uses a system Catch2 v3
``CKM_ENABLE_CUDA``        ``OFF``  Build for CUDA
``CKM_ENABLE_OPENMP``      ``OFF``  Build with OpenMP
``CKM_ENABLE_WARNINGS``    ``OFF``  Enable compiler warnings
=========================  =======  =====================================================

The ``MACHINE_CFG`` convention is the same as riot's. Pass a CMake file, either with
``-DMACHINE_CFG=...`` or the ``MACHINE_CFG`` environment variable, to set
machine-specific defaults.

Darwin
------

These are the same modules riot's Darwin CI uses:

.. code-block:: bash

   # MPI (default): hdf5/1.12.2 is parallel when loaded with openmpi
   module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3
   cmake -S conditional_kinetic_miniapp -B build-ckm -G Ninja -DCKM_ENABLE_UNIT_TESTS=ON

   # Serial: use the system's serial HDF5
   module load gcc/12.2.0 cmake/3.26.3
   cmake -S conditional_kinetic_miniapp -B build-ckm-serial -G Ninja \
     -DCKM_ENABLE_MPI=OFF -DCKM_ENABLE_UNIT_TESTS=ON -DHDF5_ROOT=/usr

Running
-------

.. code-block:: bash

   build-ckm/src/ckm -i conditional_kinetic_miniapp/inputs/skeleton.in

The skeleton driver fills one tensor-train field on every mesh block, checks the
result, and prints ``ckm: tensor-train check on N blocks: PASS``.
