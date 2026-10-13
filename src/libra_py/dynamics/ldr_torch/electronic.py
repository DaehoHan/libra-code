"""Electronic preparation for LDR models in a fixed orthonormal diabatic basis."""

import torch


def _hermitian_tensor(value, shape, device, name):
    raw = torch.as_tensor(value, device=device)
    tolerance = 1e-6 if raw.dtype in (torch.float32, torch.complex64) else 1e-12
    matrix = raw.to(torch.cdouble)
    if tuple(matrix.shape) != tuple(shape):
        raise ValueError(f"{name} must return shape {tuple(shape)}; received {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must return finite values")
    if not torch.allclose(matrix, matrix.conj().transpose(-2, -1), atol=tolerance, rtol=0):
        raise ValueError(f"{name} must return Hermitian matrices")
    return matrix


def evaluate_potential(q, potential_fn, model_params, nstates):
    """Evaluate V(q); q is (ndof, npoints), output is (npoints, s, s).

    This uses the same coordinate-first, matrix-last model convention as
    exact_torch. Each model value is evaluated in one common diabatic basis.
    """
    if not callable(potential_fn):
        raise ValueError("potential_fn must be callable")
    return _hermitian_tensor(potential_fn(q, model_params),
                             (q.shape[1], nstates, nstates), q.device, "potential_fn")


def diagonalize_potential(q, potential_fn, model_params, nstates):
    """Return ordered energies and column eigenvectors at all supplied points."""
    return torch.linalg.eigh(evaluate_potential(q, potential_fn, model_params, nstates))


def prepare_reference_data(qgrid, q0, nstates, istate, potential_fn, model_params,
                           initial_electronic_state=None):
    """Build local energies, frames, overlaps, and initial electronic overlaps.

    The default initial electronic state is the adiabatic eigenstate istate
    at q0, held fixed across the initial nuclear Gaussian. An explicit
    initial_electronic_state instead supplies a vector in the diabatic basis.
    """
    energies, frames = diagonalize_potential(qgrid.T, potential_fn, model_params, nstates)
    if initial_electronic_state is None:
        if q0.shape != (qgrid.shape[1],):
            raise ValueError("q0 must have shape (ndof,)")
        _, initial_frames = diagonalize_potential(q0[:, None], potential_fn, model_params, nstates)
        initial = initial_frames[0, :, istate]
    else:
        initial = torch.as_tensor(initial_electronic_state, dtype=torch.cdouble, device=qgrid.device)
        if initial.shape != (nstates,) or not torch.isfinite(initial).all():
            raise ValueError("initial_electronic_state must be a finite vector of shape (nstates,)")
        norm = torch.linalg.vector_norm(initial)
        if norm == 0:
            raise ValueError("initial_electronic_state must have nonzero norm")
        initial = initial / norm
    N = len(qgrid)
    overlaps = torch.einsum("ndi,mdj->injm", frames.conj(), frames).reshape(nstates*N, nstates*N)
    amplitudes = torch.einsum("ndi,d->in", frames.conj(), initial)
    return energies.T, frames, overlaps, amplitudes


def compute_energy_gradients(qgrid, frames, potential_fn, model_params):
    """Differentiate the potential at reference geometries with Torch autograd.

    Holding the reference eigenvectors fixed gives the Hellmann-Feynman
    gradients dE_i/dq_a = u_i^dagger (dV/dq_a) u_i. The result has shape
    (nstates, ngrids, ndof) and is detached before propagation.
    Each potential_fn output matrix must depend only on its own geometry.
    """
    s = frames.shape[-1]
    message = ("BAT1 requires potential_fn to preserve the Torch autograd graph "
               "from q to V; avoid detach(), NumPy, or torch.tensor(existing_tensor). "
               "For a constant potential, retain a zero dependence such as 0*q.sum().")
    with torch.enable_grad():
        q = qgrid.detach().T.clone().requires_grad_(True)
        potential = evaluate_potential(q, potential_fn, model_params, s)
        if not potential.requires_grad:
            raise ValueError(message)
        fixed_frames = frames.detach()
        energies = torch.einsum("nri,nrs,nsi->ni", fixed_frames.conj(),
                                potential, fixed_frames).real
        gradients = []
        for i in range(s):
            # Independent geometries let one reverse pass cover all centers.
            gradient = torch.autograd.grad(energies[:, i].sum(), q,
                                           retain_graph=(i < s-1), allow_unused=True)[0]
            if gradient is None:
                raise ValueError(message)
            gradients.append(gradient.T)
        result = torch.stack(gradients).detach()
    if not torch.isfinite(result).all():
        raise ValueError("BAT1 autograd produced nonfinite energy gradients at reference geometries")
    return result


def compute_midpoint_potential(qgrid, alpha, frames, potential_fn, model_params, batch_size=256):
    """Return F[i,n,j,m] = U_n^dagger V(qbar_nm) U_m in bounded batches.

    The complete model electronic space makes this equal to the sum over
    adiabatic midpoint states. Nuclear overlaps are applied by the solver.
    alpha is the normalized (ngrids, ndof) width array from the solver.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("midpoint_batch_size must be a positive integer")
    N, s = len(qgrid), frames.shape[-1]
    result = torch.empty(s, N, s, N, dtype=torch.cdouble, device=qgrid.device)
    for start in range(0, N*N, batch_size):
        pairs = torch.arange(start, min(start+batch_size, N*N), device=qgrid.device)
        n, m = pairs // N, pairs % N
        an, am = alpha[n], alpha[m]
        qbar = (an*qgrid[n] + am*qgrid[m]) / (an+am)
        potential = evaluate_potential(qbar.T, potential_fn, model_params, s)
        result[:, n, :, m] = frames[n].conj().transpose(-2, -1) @ potential @ frames[m]
    return result
