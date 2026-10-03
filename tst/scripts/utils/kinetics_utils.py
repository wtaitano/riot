# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

# Shared helpers for the kinetics regression tests: bitwise comparison of the mesh
# variables of two Parthenon HDF5 dumps, and removal of stale outputs.

import glob
import logging
import os

import numpy as np
from phdf import phdf


def output_names(dump):
    names = dump.fid["Info"].attrs["OutputDatasetNames"]
    return {n.decode() if isinstance(n, bytes) else n for n in names}


def compare_bitwise(file_a, file_b, logger, names=None):
    """Return the list of variables that differ (or are missing) between two dumps.

    With names=None every variable present in both dumps is compared."""
    a, b = phdf(file_a), phdf(file_b)
    if names is None:
        names = sorted(output_names(a) & output_names(b))
    differing = []
    for n in names:
        x = np.asarray(a.Get(n, flatten=False))
        y = np.asarray(b.Get(n, flatten=False))
        if x.shape != y.shape:
            logger.warning(f"{n}: shape {x.shape} vs {y.shape}")
            differing.append(n)
            continue
        nd = int(np.count_nonzero(x != y))
        if nd:
            logger.warning(
                f"{n}: {nd} entries differ, max |d| {np.max(np.abs(x - y)):.3e}"
            )
            differing.append(n)
        else:
            logger.debug(f"{n}: bitwise identical")
    return differing


def clean_outputs(*problem_ids):
    """Remove build/src outputs of earlier runs of these problem ids. History files are
    appended to, so stale outputs (the regression build is kept with --save_build) would
    corrupt the analysis."""
    for p in problem_ids:
        for f in glob.glob(f"build/src/{p}.*"):
            os.remove(f)


def velocity_nodes(nv, vmin, vmax):
    """Cell-centered nodes and spacing of one velocity axis, as in VelocityGrid."""
    dv = (vmax - vmin) / nv
    return vmin + (np.arange(nv) + 0.5) * dv, dv


def discrete_equilibrium(n, u, theta, vs, dvs, tol=1.0e-14, maxit=30):
    """Five-moment discrete (Mieussens) equilibrium, vectorized over points.

    n, theta: arrays (npts); u: array (npts, 3); vs[d], dvs[d]: nodes and spacing of
    axis d. Returns the separable factors g[d] (npts, nv_d), M(v) = prod_d g[d](v_d),
    whose discrete moments equal (n, n u, n (|u|^2 + 3 theta)) to roundoff. Independent
    of the C++ solver (plain Newton on the same scaled system)."""
    npts = n.shape[0]
    c = np.sqrt(theta)
    xis = [(vs[d][None, :] - u[:, d, None]) / c[:, None] for d in range(3)]
    q = np.zeros((npts, 5))
    q[:, 0] = -1.5 * np.log(2 * np.pi)
    q[:, 4] = -0.5
    T = np.zeros((npts, 5))
    T[:, 0] = 1.0
    T[:, 4] = 3.0
    lin = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    sq = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    for _ in range(maxit):
        S = np.zeros((npts, 3, 5))
        for d in range(3):
            xi = xis[d]
            e = np.exp(q[:, 1 + d, None] * xi + q[:, 4, None] * xi**2)
            e = e * (dvs[d] / c)[:, None]
            for k in range(5):
                S[:, d, k] = np.sum(e * xi**k, axis=1)
        E0 = np.exp(q[:, 0])

        def mom(p):
            return E0 * S[:, 0, p[0]] * S[:, 1, p[1]] * S[:, 2, p[2]]

        F = np.zeros((npts, 5))
        J = np.zeros((npts, 5, 5))
        for a in range(4):
            F[:, a] = mom(lin[a])
            for b in range(4):
                J[:, a, b] = mom(add(lin[a], lin[b]))
            J[:, a, 4] = sum(mom(add(lin[a], s)) for s in sq)
            J[:, 4, a] = J[:, a, 4]
        F[:, 4] = sum(mom(s) for s in sq)
        J[:, 4, 4] = sum(mom(add(s, t)) for s in sq for t in sq)
        R = F - T
        if np.max(np.linalg.norm(R, axis=1) / np.linalg.norm(T, axis=1)) < tol:
            break
        q = q - np.linalg.solve(J, R[:, :, None])[:, :, 0]
    g = [
        np.exp(q[:, 1 + d, None] * xis[d] + q[:, 4, None] * xis[d] ** 2)
        for d in range(3)
    ]
    g[0] = g[0] * (n / theta**1.5 * np.exp(q[:, 0]))[:, None]
    return g


def free_streaming_moments(x, t, initial_state, vs, dvs):
    """Exact moments (n, n u_x, n e) at points x of the discrete-velocity free-streaming
    solution f(x, v, t) = M[state(x - v_x t)](v) on a 1D-in-x problem, where
    initial_state(xf) -> (n, u (npts, 3), theta) gives the initial local equilibrium.
    Returns n, u_x and theta = (2/3)(e - |u|^2 / 2) / n... as arrays over x."""
    nvx = vs[0].size
    xf = (x[:, None] - vs[0][None, :] * t).ravel()  # foot points (cell, vx node)
    n0, u0, th0 = initial_state(xf)
    g = discrete_equilibrium(n0, u0, th0, vs, dvs)
    w = dvs[0] * dvs[1] * dvs[2]
    gx = g[0].reshape(x.size, nvx, nvx)
    idx = np.arange(nvx)
    gxx = gx[:, idx, idx]  # factor of the foot point's own vx node
    sy = [
        (g[1] * vs[1][None, :] ** k).sum(axis=1).reshape(x.size, nvx) for k in range(3)
    ]
    sz = [
        (g[2] * vs[2][None, :] ** k).sum(axis=1).reshape(x.size, nvx) for k in range(3)
    ]
    vx = vs[0][None, :]
    n = np.sum(gxx * sy[0] * sz[0], axis=1) * w
    nux = np.sum(vx * gxx * sy[0] * sz[0], axis=1) * w
    nuy = np.sum(gxx * sy[1] * sz[0], axis=1) * w
    nuz = np.sum(gxx * sy[0] * sz[1], axis=1) * w
    nvv = (
        np.sum(vx**2 * gxx * sy[0] * sz[0], axis=1)
        + np.sum(gxx * sy[2] * sz[0], axis=1)
        + np.sum(gxx * sy[0] * sz[2], axis=1)
    ) * w
    ux, uy, uz = nux / n, nuy / n, nuz / n
    theta = (nvv / n - (ux**2 + uy**2 + uz**2)) / 3.0
    return n, ux, theta


def tt_decompress(fname, nv, rcap, var="kinetics.f_tt"):
    """Full f from the tensor-train cores of a restart file (layout of
    src/kinetics/tt_tensor.hpp: [r1, r2, G1 (n0 x r1), G2 (r1 x n1 x r2), G3 (r2 x n2)],
    column-major, slots sized for rcap). Returns f with the dense layout
    (nblocks, nv0 nv1 nv2, nz, ny, nx), flat index (iz nv1 + iy) nv0 + ix, and the list of
    (r1, r2) per cell."""
    import h5py

    with h5py.File(fname, "r") as h:
        c = np.array(h[var])
    n0, n1, n2 = nv
    s1, s2 = 2, 2 + n0 * rcap
    s3 = s2 + rcap * n1 * rcap
    nb, _, nz, ny, nx = c.shape
    out = np.zeros((nb, n0 * n1 * n2, nz, ny, nx))
    ranks = []
    for b in range(nb):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    d = c[b, :, k, j, i]
                    r1, r2 = int(d[0]), int(d[1])
                    ranks.append((r1, r2))
                    G1 = d[s1 : s1 + n0 * r1].reshape(r1, n0).T
                    G2 = (
                        d[s2 : s2 + r1 * n1 * r2].reshape(r2, n1, r1).transpose(2, 1, 0)
                    )
                    G3 = d[s3 : s3 + r2 * n2].reshape(n2, r2).T
                    T = np.einsum("ia,ajb,bk->ijk", G1, G2, G3)
                    out[b, :, k, j, i] = T.transpose(2, 1, 0).reshape(-1)
    return out, ranks


def read_history(problem_id):
    """History array and {column name: index} of build/src/<problem_id>.out2.hst."""
    fname = f"build/src/{problem_id}.out2.hst"
    with open(fname) as fh:
        header = [line for line in fh if line.startswith("# [1]")][0]
    cols = {}
    for tok in header[2:].split():
        if "=" in tok:
            idx, name = tok.split("=")
            cols[name] = int(idx[1:-1]) - 1
    return np.atleast_2d(np.loadtxt(fname)), cols


def read_line(dump, var, component=None):
    """Cell centers (sorted) and values of a variable on a 1D mesh."""
    x = np.asarray(dump.x).ravel()
    order = np.argsort(x)
    a = np.asarray(dump.Get(var, flatten=False))
    if component is not None:
        a = a[:, component]
    return x[order], a.reshape(a.shape[0], -1).ravel()[order]


def sod_exact(
    x, t, gamma=5.0 / 3.0, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), x0=0.5
):
    """Exact Riemann solution (rho, u, p) of the Euler equations at points x, time t.
    left/right = (rho, u, p). Standard two-shock/rarefaction construction (Toro ch. 4).
    """
    from scipy.optimize import brentq

    g = gamma
    rl, ul, pl = left
    rr, ur, pr = right
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def fk(p, rk, pk, ck):
        if p > pk:
            A, B = 2.0 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

    ps = brentq(lambda p: fk(p, rl, pl, cl) + fk(p, rr, pr, cr) + ur - ul, 1e-10, 1e3)
    us = 0.5 * (ul + ur) + 0.5 * (fk(ps, rr, pr, cr) - fk(ps, rl, pl, cl))
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)
    for k, s in enumerate((x - x0) / t):
        if s < us:  # left of the contact
            if ps > pl:  # left shock
                sl = ul - cl * np.sqrt((g + 1) / (2 * g) * ps / pl + (g - 1) / (2 * g))
                state = (
                    (rl, ul, pl)
                    if s < sl
                    else (
                        rl
                        * (ps / pl + (g - 1) / (g + 1))
                        / ((g - 1) / (g + 1) * ps / pl + 1),
                        us,
                        ps,
                    )
                )
            else:  # left rarefaction
                csl = cl * (ps / pl) ** ((g - 1) / (2 * g))
                if s < ul - cl:
                    state = (rl, ul, pl)
                elif s > us - csl:
                    state = (rl * (ps / pl) ** (1 / g), us, ps)
                else:
                    uu = 2 / (g + 1) * (cl + (g - 1) / 2 * ul + s)
                    c = 2 / (g + 1) * (cl + (g - 1) / 2 * (ul - s))
                    state = (
                        rl * (c / cl) ** (2 / (g - 1)),
                        uu,
                        pl * (c / cl) ** (2 * g / (g - 1)),
                    )
        else:  # right of the contact
            if ps > pr:  # right shock
                sr = ur + cr * np.sqrt((g + 1) / (2 * g) * ps / pr + (g - 1) / (2 * g))
                state = (
                    (rr, ur, pr)
                    if s > sr
                    else (
                        rr
                        * (ps / pr + (g - 1) / (g + 1))
                        / ((g - 1) / (g + 1) * ps / pr + 1),
                        us,
                        ps,
                    )
                )
            else:  # right rarefaction
                csr = cr * (ps / pr) ** ((g - 1) / (2 * g))
                if s > ur + cr:
                    state = (rr, ur, pr)
                elif s < us + csr:
                    state = (rr * (ps / pr) ** (1 / g), us, ps)
                else:
                    uu = 2 / (g + 1) * (-cr + (g - 1) / 2 * ur + s)
                    c = 2 / (g + 1) * (cr - (g - 1) / 2 * (ur - s))
                    state = (
                        rr * (c / cr) ** (2 / (g - 1)),
                        uu,
                        pr * (c / cr) ** (2 * g / (g - 1)),
                    )
        rho[k], u[k], p[k] = state
    return rho, u, p
