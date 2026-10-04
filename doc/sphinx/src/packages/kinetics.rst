.. _`chap:kinetics`:

Kinetics (BGK neutral gas)
==========================

.. note::

   This page was written with the assistance of generative AI.

The ``kinetics`` package evolves the velocity distribution function
:math:`f(\vec{x}, \vec{v}, t)` of a neutral monatomic gas with the BGK collision
operator, in one to three spatial dimensions and always three velocity dimensions. It
is a dense discrete-velocity solver on a fixed uniform velocity grid and serves as the
reference for later compressed (tensor-train) representations. The hydrodynamics of
Chapter :ref:`chap:hydro` is evolved alongside but is not coupled to the kinetic gas:
the two start from the same state and evolve independently, which allows side-by-side
comparison in the continuum limit.

Governing Equations
-------------------

The distribution function is the number density in phase space,
:math:`[f] = \mathrm{cm^{-3}\,(cm/s)^{-3}}`, and obeys

.. math::

     \frac{\partial f}{\partial t} + \vec{v}\cdot\nabla f = \nu\,\left(M[f] - f\right),

where :math:`M[f]` is the local Maxwellian with the density :math:`n`, velocity
:math:`\vec{u}` and temperature :math:`T` of :math:`f`, and :math:`\nu` is the collision
frequency. With particle mass :math:`m` and velocity-node weight
:math:`w = \Delta v_1\Delta v_2\Delta v_3`, the moments are

.. math::

     n = \sum f\,w, \quad n\vec{u} = \sum \vec{v} f\,w, \quad
     \tfrac{3}{2} n k_B T = \sum \tfrac{1}{2} m |\vec{v}-\vec{u}|^2 f\,w,

and the output also contains the stress
:math:`P_{ij} = m\sum c_i c_j f\,w` and heat flux
:math:`q_i = \tfrac{1}{2} m\sum c_i |\vec{c}|^2 f\,w` (:math:`\vec{c} = \vec{v}-\vec{u}`).

BGK has Prandtl number one. In the continuum limit it reproduces the Navier-Stokes
equations with viscosity :math:`\mu = p/\nu` and conductivity
:math:`\kappa = \tfrac{5}{2}(k_B/m)\,\mu`.

Numerical Method
----------------

**Velocity grid.** Each axis :math:`d` of the box :math:`[v_{d,\min}, v_{d,\max}]` is
split into ``nv``\ :math:`_d` equal cells, with nodes at the cell centers and equal
(midpoint) weights. The distribution of one spatial cell is stored flat, with node index
:math:`n = (i_z N_y + i_y) N_x + i_x`.

**Discrete equilibrium.** :math:`M` is the discrete equilibrium of Mieussens: the
distribution :math:`\exp(\alpha\cdot\phi(\vec{v}))`,
:math:`\phi = (1, \vec{v}, |\vec{v}|^2)`, whose *discrete* moments equal those of
:math:`f`. It is found by a damped Newton iteration in scaled velocities,
:math:`\vec{\xi} = (\vec{v}-\vec{u})/c`. Because the exponent is a sum over axes, every
grid sum factors into three one-dimensional sums, so one iteration costs
:math:`O(N_x+N_y+N_z)`. The relaxation therefore conserves mass, momentum and energy to
roundoff. If Newton fails (temperature not resolved by the grid, or a mean velocity near
the box edge), the sampled continuous Maxwellian is used instead, and the cell is
flagged in ``kinetics.eq_fallback``.

**Collisions.** :math:`M` and :math:`\nu` do not change during a relaxation step,
because relaxation conserves the moments that fix them. Every relaxation step therefore
has the closed form :math:`f \leftarrow f + c\,(M - f)`, with :math:`c` a function of
:math:`\nu h` set by the time integrator.

**Streaming.** Nodal backward semi-Lagrangian interpolation, linear (``sl_order = 1``)
or quadratic (``sl_order = 2``), applied as a tensor product in several dimensions. The
step is capped so that the fastest node moves at most ``cfl`` :math:`\le 1` cells, which
keeps the stencil within one ghost layer. Under that cap, linear SL is first-order
upwind and unlimited quadratic SL is Lax-Wendroff, so both conserve exactly on a periodic
domain. The optional ``minmax`` limiter clips the quadratic value to the bracketing
cells. This keeps :math:`f \ge 0` but **breaks exact conservation**: see the caveat below.

**Time stepping.** Hydrodynamics sets the time step :math:`\Delta t`; kinetics does not
limit it. Inside each hydro step the kinetic package takes :math:`n` substeps
:math:`h = \Delta t / n`, with :math:`n` the smallest count for which every SL step
satisfies the streaming cap. ``kinetics/integrator`` selects the substep:

* ``sl_dirk2`` (default): a characteristic IMEX Runge-Kutta scheme. Streaming is
  explicit (SL); BGK is implicit, with a two-stage, stiffly accurate, L-stable DIRK
  (:math:`\gamma = 1 - 1/\sqrt{2}`). Because the implicit stages have closed forms and
  the explicit stage combination folds into the first relaxation, a substep is

  .. math::

     f \leftarrow \mathrm{SL}(\gamma h) f, \quad f \leftarrow f + c_1 (M - f), \quad
     f \leftarrow \mathrm{SL}((1-\gamma) h) f, \quad f \leftarrow f + c_2 (M - f),

  with :math:`c_1 = (1-\gamma)\nu h/(1+\gamma\nu h)` and
  :math:`c_2 = \gamma\nu h/(1+\gamma\nu h)`. It is second order in time uniformly in
  :math:`\nu h`, including the stiff regime :math:`\nu h \gg 1`, and needs no extra
  storage. Its longest SL step is :math:`(1-\gamma) h`, so :math:`h` is 1.41 times the
  single-step cap, and the cost per unit time is 1.41 times that of ``strang``. For
  :math:`\nu h > 1/(1-2\gamma) \approx 2.41`, :math:`c_1 > 1`: the first stage
  over-relaxes, and :math:`f` can become slightly negative (``kinetics_min_f``).
* ``strang``: SL(:math:`h/2`) BGK(:math:`h`) SL(:math:`h/2`) with exact relaxation,
  :math:`c = 1 - e^{-\nu h}`, which keeps :math:`f \ge 0`. Adjacent half steps are
  merged by default (``merge_half_steps``). Second order while
  :math:`\nu h \lesssim 1`; first order in the stiff regime.

Without collisions (``nu0 = 0``) both reduce to free streaming. With the default
first-order streaming (``sl_order = 1``) the spatial error usually dominates, and the
second-order time accuracy of ``sl_dirk2`` shows only with ``sl_order = 2``.

**Initialization.** :math:`f` is the discrete equilibrium of the hydrodynamic state
(density, velocity, temperature) of the problem generator, or a discrete bi-Maxwellian
with a prescribed temperature anisotropy at the same density and energy.

Requirements
------------

``hydro`` must be enabled, and there must be exactly one material: an ``IdealGas`` with
:math:`\Gamma = 5/3`. The particle mass follows from the specific heat,
:math:`k_B/m = (\Gamma-1)\,C_v`.

Boundary Conditions
-------------------

Each face takes ``kinetics/<face>_bc``, which must be ``periodic`` exactly when the mesh
face is periodic.

* ``outflow``: zero gradient (default on non-periodic faces).
* ``specular``: mirror reflection, which needs a velocity box symmetric about zero along
  the wall normal.
* ``diffuse``: nodes entering the domain carry the discrete equilibrium at the wall
  temperature ``<face>_wall_T`` and tangential velocity ``<face>_wall_u1..3``. The wall
  density is set so the net mass flux through the wall is zero.

Input Parameters
----------------

The package is enabled with ``kinetics = true`` in the ``<physics>`` block.

.. list-table:: Parameters in the ``<kinetics>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - nv1, nv2, nv3
     - int
     - required
     - Velocity nodes per axis.
   * - v1min ... v3max
     - Real
     - required
     - Velocity box (cm/s).
   * - nu_model
     - string
     - ``constant``
     - ``constant`` (``nu0`` [1/s]) or ``power_law``
       (:math:`\nu = p/\mu`, :math:`\mu = \mu_{ref}(T/T_{ref})^\omega`; ``mu_ref``,
       ``T_ref``, ``omega``).
   * - sl_order
     - int
     - ``1``
     - Semi-Lagrangian interpolation order (1 or 2).
   * - sl_limiter
     - string
     - ``minmax``
     - ``minmax`` or ``none`` (quadratic only).
   * - cfl
     - Real
     - ``1.0``
     - Maximum cells per substep moved by the fastest node (:math:`\le 1`).
   * - integrator
     - string
     - ``sl_dirk2``
     - ``sl_dirk2`` (IMEX-RK, second order for any :math:`\nu h`) or ``strang``.
   * - streaming
     - bool
     - ``true``
     - ``false`` keeps only the collisions (0D relaxation tests), with the same
       substeps.
   * - merge_half_steps
     - bool
     - ``true``
     - ``strang`` only: merge adjacent half steps.
   * - init
     - string
     - ``equilibrium``
     - ``equilibrium``, ``bimaxwellian`` (``init_T_ratio``
       :math:`=T_\parallel/T_\perp`, ``init_axis``) or ``two_maxwellian`` (two halves
       drifting by :math:`\mp` ``init_drift`` :math:`\sqrt{k_BT/m}` along
       ``init_axis``, with the hydro density, velocity and energy).
   * - eq_tol, eq_max_iter
     - Real, int
     - ``1e-13``, ``20``
     - Tolerance and iteration limit of the equilibrium solve.
   * - eq_fallback_abort
     - Real
     - ``1e-3``
     - Abort if a larger fraction of cells falls back to the sampled Maxwellian.
   * - edge_mass_warn, edge_mass_abort
     - Real
     - ``1e-10``, ``1e-6``
     - Thresholds on the mass fraction carried by the outermost node layer.
   * - min_vth_over_dv
     - Real
     - ``1.5``
     - Warn if the thermal speed spans fewer velocity cells.
   * - check_every
     - int
     - ``100``
     - Cycles between resolution checks.
   * - <face>_bc
     - string
     - mesh-dependent
     - ``periodic``, ``outflow``, ``specular`` or ``diffuse``.

Tensor-train representation
---------------------------

With ``representation = tt`` the distribution of each cell is stored as a three-core
tensor train over :math:`(v_x, v_y, v_z)`,

.. math::

   f(i, j, k) = \sum_{a < r_1} \sum_{b < r_2} G_1(i, a)\, G_2(a, j, b)\, G_3(b, k),

in the variable ``kinetics.f_tt`` (the ranks :math:`r_1, r_2` followed by the cores,
sized for ``tt_rank_max``, i.e. :math:`2 + N_x r + N_y r^2 + N_z r` reals per cell
instead of :math:`N_x N_y N_z`). The discrete equilibrium is exactly rank 1. Moments
are exact contractions of the cores. The nonlinear diagnostics (entropy, min
:math:`f`) are estimated by cross approximation by default (``tt_diag = cross``, see
below) or computed by decompressing each cell at history cadence (``tt_diag =
exact``); negative mass is always computed by decompression. The
derived output fields are the same as for ``dense``, plus ``kinetics.tt_rank``
:math:`(r_1, r_2)`. Design and status: ``S1_DESIGN.md`` of the kinetic project.

.. warning::

   Work in progress: in this version ``tt`` runs on 1D, 2D and 3D meshes (any 3V grid),
   with both integrators and all four kinetic boundary conditions. On 2D/3D meshes the
   streaming step is a sequence of :math:`x_1, x_2, x_3` sweeps (each sweep covers one
   ghost layer along the directions still to come), which before rounding equals the
   dense single-pass tensor-product update; one ghost exchange per step is kept. Each relaxation forms
   :math:`(1-c)f + cM` (ranks + 1) and rounds it once. Streaming of a cell along
   :math:`x` is two rounded block sums, :math:`\mathrm{round}(w_0 f_i + w_- f_{i-1})`
   then :math:`+\,w_+ f_{i+1}`, where the weights depend only on :math:`v_x` and so
   scale one core. Specular walls reverse the wall-normal core exactly; diffuse walls
   build the ghost as a masked sum of the boundary cell and the wall Maxwellian and
   round it. Mesh refinement and ``sl_limiter = minmax`` (a pointwise limiter has no
   tensor-train form) are not available.

   Every rounding adds an error of up to ``tt_eps`` (relative), and these errors
   accumulate over the steps: a substep of ``sl_dirk2`` rounds 6 times (3 per stage).
   Measured on a smooth wave, the moment error after :math:`N` roundings is
   0.2–0.8 :math:`N\,` ``tt_eps``, so refining the time step at fixed ``tt_eps``
   increases the TT error. To compare with the dense solver at roundoff level use
   ``tt_eps = 1e-14``; with ``1e-10`` the moments of a short 1D Sod run differ by a
   few 1e-9. The time order of both integrators is unchanged in the stiff regime
   (``sl_dirk2`` 1.98/1.99, ``strang`` 1.06/0.98, the same as dense).

.. list-table:: Tensor-train parameters in the ``<kinetics>`` block.
   :class: wraptable
   :header-rows: 1
   :widths: 25 12 18 45

   * - Parameter
     - Type
     - Default
     - Description
   * - representation
     - string
     - ``dense``
     - ``dense`` or ``tt``.
   * - tt_eps
     - Real
     - ``1e-8``
     - Relative Frobenius tolerance of each TT rounding; ``0`` truncates only at
       ``tt_rank_max`` (fixed rank).
   * - tt_rank_max
     - int
     - ``16``
     - Largest TT rank (2 to 64); sets the storage per cell.
   * - tt_diag
     - string
     - ``cross``
     - Nonlinear diagnostics: ``cross`` (DEIM cross approximation, estimates) or
       ``exact`` (decompress every cell).
   * - tt_cross_eps
     - Real
     - ``1e-10``
     - Cross: stop when the max residual on the trial set is below this times the
       max of :math:`|g|` there.
   * - tt_cross_rank_max
     - int
     - ``32``
     - Cross: cap on the ranks of the approximated function (1 to 64).
   * - tt_cross_trial_factor
     - int
     - ``16``
     - Cross: trial nodes per cell = factor :math:`\times` max TT rank :math:`\times`
       max :math:`N_v`.
   * - tt_cross_delta
     - Real
     - ``1e-12``
     - Cross entropy: regularization :math:`\delta` relative to the sampled max
       :math:`|f|`.

Restarts carry ``kinetics.f_tt`` (bitwise round trip), and runs on any number of MPI
ranks give bitwise identical fields. A restart must keep ``representation``,
``nv1..nv3`` and ``tt_rank_max``. Parthenon reads the components of ``f`` by position
before the package can check anything, so a mismatch is caught in two ways: if the new
layout is larger than the stored one, reading ``f`` fails with an HDF5 "selection not
within extent" error; otherwise the stored layout is compared on the first step and the
run aborts with a message naming both layouts. Restart files that do not record the
layout (written by earlier versions of the package) are refused.

The history file adds ``kinetics_tt_ranks_0..3`` (cell count, sum of :math:`r_1`, sum of
:math:`r_2`, cells at ``tt_rank_max``), ``kinetics_tt_round_0..2`` (over the last hydro
step and all cells: sum of the relative discarded norms of the roundings, rank-cap hits,
non-converged SVDs) and ``kinetics_tt_max_rank``.

**Cross diagnostics** (``tt_diag = cross``, ``src/kinetics/tt_cross.hpp``). A pointwise
function :math:`g = \varphi(f)` is approximated as a tensor train from a few node values
of :math:`f`, by two DEIM projections: the :math:`v_x` fibers at a set :math:`J_2` of
:math:`(v_y, v_z)` pairs give the first core and interpolation rows :math:`I_1`; the
:math:`(I_1, v_y)` fibers at a set :math:`J_3` of :math:`v_z` nodes give the second core
and rows :math:`I_2`; the :math:`v_z` fibers at :math:`I_2` are the third core. The
initial :math:`J_2, J_3` are the DEIM points of :math:`f`'s own right bases; the
residual is then measured on a fixed pseudo-random trial set and the worst node's
fibers are added until the residual meets ``tt_cross_eps`` or a set reaches
``tt_cross_rank_max``. Sums of :math:`g` are exact contractions of its cores.

* Entropy is the cross estimate of :math:`\sum s(\ln(s + \delta) - 1)` with
  :math:`s` a :math:`C^1` positive part of :math:`f` of width :math:`\delta` =
  ``tt_cross_delta`` :math:`\times \max|f|` (exact entropy uses :math:`f > 0` only);
  the regularization keeps :math:`g` smooth where rounding noise makes :math:`f`
  change sign.
* Min :math:`f` is a sampled bound: the minimum over the trial set, refined by
  coordinate descent along full fibers from its four smallest nodes. It is an upper
  bound on the true minimum and finds smooth negative lobes, not isolated single-node
  spikes.
* Negative mass is computed by decompression in both modes: its integrand is rounding
  noise at the level of ``tt_eps``, which no low-rank cross resolves cheaply.

The history adds ``kinetics_tt_cross_0..3`` (summed over cells at the output:
evaluations of :math:`\varphi`, :math:`\max(q_1, q_2)`, rank-cap hits, cells whose
cross did not converge).

**Scaling with the velocity grid.** Measured on one core for 0D relaxation and 1D Sod
(128 cells, :math:`\nu_0 = 10^4` and :math:`10`, 37 steps) at
:math:`N_v = 16^3, 32^3, 64^3` (``tt_rank_max`` 8 at :math:`16^3`, 16 otherwise):

* The TT ranks do not grow with :math:`N_v` (Sod: at most 9 at ``tt_eps = 1e-14``, 5 at
  ``1e-8``), so TT storage and cost grow about linearly in :math:`N_v` per direction,
  while dense grows as :math:`N_v^3`.
* At ``tt_eps = 1e-14`` the TT moments match dense to :math:`3\times 10^{-11}` at every
  :math:`N_v`. At the default ``1e-8`` they differ by :math:`4\times 10^{-7}` to
  :math:`10^{-6}`, set by the accumulated rounding error, independent of :math:`N_v`.
* At :math:`64^3` the allocated storage is 14 times smaller than dense and the achieved
  storage 76 to 200 times smaller (peak memory 2.7 GB dense vs 0.28 GB TT). The run is
  15 to 16 times faster at ``tt_eps = 1e-14`` and 50 to 57 times faster at ``1e-8``.
* At :math:`16^3` TT saves little: the allocation is only 3 times smaller, and at
  ``tt_eps = 1e-14`` TT is slower than dense.
* The cross diagnostics evaluate :math:`\varphi` at 2 to 5 % of the nodes at
  :math:`64^3` (30 to 60 % at :math:`16^3`).

Raw data: ``s1/step9/RESULTS.md`` of the kinetic project.

Output
------

The derived fields ``kinetics.rho``, ``kinetics.velocity``, ``kinetics.temperature``,
``kinetics.pressure``, ``kinetics.stress`` (xx, yy, zz, xy, xz, yz),
``kinetics.heat_flux`` and ``kinetics.eq_fallback`` can be listed in an output block.
The distribution ``kinetics.f`` has one component per velocity node and is written only
to restart files. For restart blocks, set ``write_xdmf = false`` and
``hdf5_compression_level = 0``: XDMF describes every component (tens of MB of text at
:math:`32^3` nodes), and gzip compression costs much more than the raw write.

The history file gets ``kinetics_sums_0..11`` (kinetic mass, momentum (3), energy,
entropy :math:`\sum f(\ln f - 1)\,w\,dV`, negative mass, then hydro mass, momentum (3) and
total energy), ``kinetics_min_f``, ``kinetics_eq_fallbacks`` and ``kinetics_substeps``.
To check conservation at the 1e-12 level, set ``data_format = %.17e`` on the history
block.

.. warning::

   Quadratic interpolation (``sl_order = 2``) with the ``minmax`` limiter does not
   conserve mass and energy to roundoff. In a Sod problem at small Knudsen number, about
   0.3% of the mass is lost and the shock is displaced. Keep the default ``sl_order = 1``
   (conservative) for shock problems; ``sl_order = 2`` with ``sl_limiter = none`` is
   suitable for smooth ones.

Examples
--------

Decks are in ``inputs/kinetics``: ``relax0d.py`` (homogeneous relaxation),
``sod.py`` (shock tube), ``freestream.py`` (smooth free streaming) and ``couette.py``
(Couette flow with heat conduction between diffuse walls). A minimal block:

.. code:: python

    riot.input("physics", hydro=True, kinetics=True)
    riot.input("material0", eos_type="IdealGas", Gamma=5.0 / 3.0, Cv=1.5)
    riot.input(
        "kinetics",
        nv1=24, nv2=24, nv3=24,
        v1min=-8.0, v1max=8.0, v2min=-8.0, v2max=8.0, v3min=-8.0, v3max=8.0,
        nu_model="constant", nu0=1.0e4,
        sl_order=1,
    )
