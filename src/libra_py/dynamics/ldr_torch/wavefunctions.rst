Spatial amplitudes and density-matrix placeholders
=================================================

The solver has six wavefunction-related data attributes:

=================  =========  =====================================================
Attribute          Shape      Meaning
=================  =========  =====================================================
``psi_r_dia``      (P, M)     Current diabatic amplitudes
``psi_r_adi``      (P, M)     Current local adiabatic amplitudes
``psi_r_dia_all``  (T, P, M)  Saved diabatic amplitudes
``psi_r_adi_all``  (T, P, M)  Saved local adiabatic amplitudes
``rho_dia_all``    None       Placeholder for diabatic electronic density matrices
``rho_adi_all``    None       Placeholder for adiabatic electronic density matrices
=================  =========  =====================================================

P is ``ngrids``: amplitudes are always evaluated at the given ``qgrid``.
M is ``nstates``, and T is the number of saved steps, including step zero.
The amplitude arrays are ordinary tensors. Amplitude histories are allocated only when requested; otherwise they are ``None``.
Both ``rho_*_all`` attributes are initialized to ``None`` and are not
calculated during propagation. Their calculation is deferred until a
spatial integration rule is chosen.

The existing ``time`` array records snapshot times automatically when any
of these histories is enabled. There is no separate wavefunction time array.

The amplitudes include all reference states and nuclear centers:

.. math::

   \psi^{\mathrm{dia}}_a(q_p) =
   \sum_{i,n} c_{i,n} g_n(q_p)\langle a|\phi_i(q_n)\rangle,
   \qquad
   \psi^{\mathrm{adi}}_k(q_p) =
   \sum_a\langle\phi_k(q_p)|a\rangle\psi^{\mathrm{dia}}_a(q_p).

Here ``C_curr.reshape(M, N)[i, n]`` is the coefficient and
:math:`g_n(q)=\prod_d(2\alpha_{nd}/\pi)^{1/4}
\exp[-\sum_d\alpha_{nd}(q_d-q_{nd})^2]` is a normalized Gaussian.

``update_psi_r()`` explicitly updates both current amplitudes. The solver
calls it during initialization, when saving requested histories, at the end
of propagation, and when writing the final file. Call it yourself after
manually changing ``C_curr``. Reading an attribute performs no computation.
The basis, model, and qgrid are fixed for a run.

Configuration and saving
------------------------

``qgrid`` is the sole spatial grid, with shape ``(ngrids, ndof)``.
``psi_batch_size`` (default 256) bounds reconstruction batches.
Enable amplitude histories with::

    params["properties_to_save"] = [
        "time", "norm", "psi_r_dia", "psi_r_adi"
    ]

Amplitudes retain the physical norm; no implicit renormalization is applied.
The global squared norm is :math:`C^\dagger S C`, and the existing saved
``norm`` is its square root. For normalized spatial densities, divide
``abs(psi)**2`` by the global squared norm.

Spatial densities require no extra attributes::

    state_density_dia = solver.psi_r_dia_all.abs().square()  # (T, P, M)
    state_density_adi = solver.psi_r_adi_all.abs().square()  # (T, P, M)
    total_density = state_density_adi.sum(dim=-1)           # (T, P)

The two total densities agree for a complete electronic model basis.
``save()`` writes all six fields and qgrid, including the ``None`` rho
placeholders. The final amplitudes may be later than the last history entry
when the final step misses the save interval.

The existing ``denmat`` based on Loewdin orthogonalized coefficients remains
available separately from the deferred ``rho_*_all`` calculation.

With only precomputed electronic overlaps and no ``potential_fn``, native
adiabatic amplitudes are available, but the named diabatic frame cannot be
recovered: ``psi_r_dia`` is ``None``. Diabatic histories then require the
model interface.
