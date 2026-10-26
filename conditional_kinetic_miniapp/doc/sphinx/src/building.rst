..
   This file was made in part with generative AI.

Building
========

ckm is built as part of riot. Turn it on with ``-DRIOT_BUILD_CKM=ON``, which is off by
default. ckm uses riot's own dependencies from ``external/``. On the ckm feature
branch, ``external/parthenon`` points at the tensor-train branch of Parthenon.

Requirements
------------

* A C++20 compiler. The tensor-train branch of Parthenon uses concepts.
* CMake 3.26 or newer.
* HDF5. ckm refuses to configure with ``RIOT_ENABLE_HDF5=OFF``. When MPI is on (the
  default), HDF5 must be a parallel build.
* MPI. It is optional and on by default.

Fetching the submodules
-----------------------

Run this yourself, in a normal shell, not from Claude Code. Fetching submodules writes
``.git/config``, which the Claude Code sandbox keeps read-only, so it fails there with
``could not lock config file``. Run it once per clone, and again after a submodule pin
changes.

.. code-block:: bash

   git -C .. submodule update --init --recursive \
     external/parthenon external/kokkos-kernels external/singularity-eos external/singularity-opac

Configuring and building
------------------------

Run these from ``conditional_kinetic_miniapp/``:

* **Source.** The CMake source is the riot root (``-S ..``).
* **Build directories.** They go inside the miniapp (``build-mpi/``,
  ``build-serial/``). Riot's ``.gitignore`` rule ``build*/`` already ignores them,
  and they stay writable under the Claude Code sandbox.

.. code-block:: bash

   cmake -S .. -B build-mpi -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON -DRIOT_BUILD_PYTHON=OFF
   cmake --build build-mpi -j --target ckm ckm_unit_tests
   ctest --test-dir build-mpi/conditional_kinetic_miniapp --output-on-failure

If you leave out ``--target``, riot itself is built too. Riot's own tests then run
with ``ctest --test-dir build-mpi``.

.. note::

   Riot's ``riotlib`` does not currently compile with ``RIOT_ENABLE_MPI=OFF``. This is
   a problem on riot ``main`` as well; ckm did not cause it. Serial builds of ckm work
   as long as you build only the ckm targets and ``ckmlib`` does not link
   ``riotlib``.

Riot's options apply to ckm as well. The ones that matter most are:

==========================  =======  ===================================================
Option                      Default  Meaning
==========================  =======  ===================================================
``RIOT_BUILD_CKM``          ``OFF``  Build ckm (``ckmlib``, ``ckm``, and its tests)
``RIOT_ENABLE_MPI``         ``ON``   Build with MPI (requires parallel HDF5)
``RIOT_ENABLE_UNIT_TESTS``  ``OFF``  Also build ckm's Catch2 unit tests and driver test
``RIOT_BUILD_CATCH2``       ``ON``   Use riot's bundled Catch2. ``OFF`` uses a system Catch2
``RIOT_ENABLE_CUDA``        ``OFF``  Build for CUDA
``RIOT_BUILD_PYTHON``       ``ON``   Riot's Python interface. ckm does not need it
==========================  =======  ===================================================

Darwin
------

These are the modules riot's Darwin CI uses:

.. code-block:: bash

   # MPI (default): hdf5/1.12.2 is parallel when loaded with openmpi
   module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3
   cmake -S .. -B build-mpi -G Ninja -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON \
     -DRIOT_BUILD_PYTHON=OFF

   # Serial: use the system's serial HDF5
   module load gcc/12.2.0 cmake/3.26.3
   cmake -S .. -B build-serial -G Ninja -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON \
     -DRIOT_BUILD_PYTHON=OFF -DRIOT_ENABLE_MPI=OFF -DHDF5_ROOT=/usr

Running
-------

.. code-block:: bash

   build-mpi/conditional_kinetic_miniapp/src/ckm -i inputs/skeleton.in

The skeleton driver fills one tensor-train field on every mesh block and checks the
result. On success it prints ``ckm: tensor-train check on N blocks: PASS``.
