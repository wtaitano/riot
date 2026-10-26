..
   This file was made in part with generative AI.

ckm: conditional kinetic miniapp
================================

ckm is a `Parthenon <https://github.com/parthenon-hpc-lab/parthenon>`_ application
for conditional kinetic modelling with block-AMR tensor trains. It lives in the
``conditional_kinetic_miniapp/`` directory of riot and is developed on the
long-lived ``wtaitano/ckm`` feature branch.

* It is built as part of riot with ``-DRIOT_BUILD_CKM=ON``.
* It uses riot's ``external/parthenon``. On the feature branch, that submodule points
  at the tensor-train branch of Parthenon (`PR #1464
  <https://github.com/parthenon-hpc-lab/parthenon/pull/1464>`_).
* It may use riot's utilities, but riot never depends on ckm.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   src/building
   src/contributing
   src/maintainers
