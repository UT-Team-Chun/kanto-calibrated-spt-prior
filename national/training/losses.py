"""SVGP losses and regularizers.

Two objectives are exposed:

- :func:`elbo_with_regime_weights` -- SVGP variational lower bound, unbiased
  on shuffled mini-batches, with optional per-sample weights used to up-weight
  rare geological regimes. The KL term is divided by the full dataset size
  ``num_data`` so the per-step gradient magnitude is invariant to batch size
  (matching GPyTorch's ``VariationalELBO`` convention).

- :func:`mmd_regularizer` -- Maximum Mean Discrepancy between the encoder
  output distribution on a batch and a reference distribution (typically a
  fixed unit-Gaussian draw). It penalizes encoder collapse without forcing a
  specific shape on the learned features.
"""

from __future__ import annotations

import gpytorch
import torch


def elbo_with_regime_weights(
    predictive_dist: gpytorch.distributions.MultivariateNormal,
    likelihood: gpytorch.likelihoods.Likelihood,
    y: torch.Tensor,
    *,
    num_data: int,
    model: gpytorch.models.ApproximateGP,
    sample_weights: torch.Tensor | None = None,
    beta: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted SVGP loss (negative ELBO) suitable for an Adam minimizer.

    Mathematically the unbiased mini-batch ELBO is::

        ELBO = (N / B) * sum_b w_b * E_q[ log p(y_b | f_b) ] - beta * KL(q(u) || p(u))

    where ``B`` is the batch size, ``N`` is ``num_data``, ``w_b`` is a per-
    sample weight (default 1), and ``beta`` is the KL temperature. The loss
    returned is ``-ELBO / N`` so the magnitude is comparable across dataset
    sizes.

    Args:
        predictive_dist: ``model(x_batch)``.
        likelihood: Gaussian likelihood used to integrate out ``f``.
        y: target tensor of shape ``(B,)``.
        num_data: total dataset size (NOT the batch size).
        model: the SVGP module -- used to read the variational KL.
        sample_weights: optional ``(B,)`` non-negative weights.
        beta: KL temperature.
    """
    # Gaussian + FixedNoiseGaussian have closed-form expected_log_prob;
    # StudentTLikelihood (and other _OneDimensionalLikelihood subclasses)
    # fall back to Gauss-Hermite quadrature internally, which is a drop-in
    # replacement at this call site. CensoredGaussianLikelihood inherits
    # GaussianLikelihood and overrides expected_log_prob to right-censor
    # at a configurable cap. We allow all four explicitly and reject
    # anything else to flag accidental new-likelihood wiring.
    if not isinstance(
        likelihood,
        (
            gpytorch.likelihoods.GaussianLikelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
            gpytorch.likelihoods.StudentTLikelihood,
        ),
    ):
        raise TypeError(
            "elbo_with_regime_weights supports Gaussian / FixedNoiseGaussian "
            "/ StudentT / CensoredGaussian likelihoods only; got "
            f"{type(likelihood).__name__}."
        )
    if y.dim() != 1:
        raise ValueError(f"y must be 1-D, got shape {tuple(y.shape)}")
    batch_size = y.shape[0]
    if batch_size == 0:
        raise ValueError("Empty batch.")
    if num_data <= 0:
        raise ValueError(f"num_data must be positive, got {num_data}")

    if sample_weights is None:
        weights = y.new_ones(batch_size)
    else:
        if sample_weights.shape != y.shape:
            raise ValueError(
                f"sample_weights shape {tuple(sample_weights.shape)} != y shape "
                f"{tuple(y.shape)}"
            )
        weights = sample_weights.to(dtype=y.dtype, device=y.device)
        if (weights < 0).any():
            raise ValueError("sample_weights must be non-negative.")

    # FixedNoiseGaussianLikelihood needs the per-point noise tensor passed
    # at every call. GaussianLikelihood ignores the kwarg.
    if noise is not None:
        per_point_ll = likelihood.expected_log_prob(y, predictive_dist, noise=noise)
    else:
        per_point_ll = likelihood.expected_log_prob(y, predictive_dist)
    weighted_mean_ll = (weights * per_point_ll).sum() / weights.sum().clamp_min(1e-12)
    data_term = num_data * weighted_mean_ll

    kl = model.variational_strategy.kl_divergence().sum()

    elbo = data_term - beta * kl
    return -elbo / num_data


def mmd_regularizer(
    phi: torch.Tensor,
    phi_ref: torch.Tensor,
    *,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Multi-bandwidth squared MMD between ``phi`` and ``phi_ref``.

    Uses a mixture of RBF kernels with the given bandwidths. The reference
    sample is typically drawn from ``N(0, I)`` of the same dimension as the
    encoder output. The penalty is non-negative and may be minimized directly.

    Args:
        phi: encoder outputs, shape ``(B, D)``.
        phi_ref: reference sample, shape ``(M, D)``.
        sigmas: RBF bandwidths to average over.
    """
    if phi.dim() != 2 or phi_ref.dim() != 2:
        raise ValueError(
            f"phi and phi_ref must be 2-D; got shapes {tuple(phi.shape)} and "
            f"{tuple(phi_ref.shape)}"
        )
    if phi.shape[-1] != phi_ref.shape[-1]:
        raise ValueError(
            f"phi feature dim {phi.shape[-1]} != phi_ref feature dim {phi_ref.shape[-1]}"
        )

    def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return (diff * diff).sum(dim=-1)

    dxx = _pairwise_sq_dist(phi, phi)
    dyy = _pairwise_sq_dist(phi_ref, phi_ref)
    dxy = _pairwise_sq_dist(phi, phi_ref)

    out = phi.new_zeros(())
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        gamma = 1.0 / (2.0 * sigma * sigma)
        out = out + (
            torch.exp(-gamma * dxx).mean()
            + torch.exp(-gamma * dyy).mean()
            - 2.0 * torch.exp(-gamma * dxy).mean()
        )
    return out / len(sigmas)


__all__ = ["elbo_with_regime_weights", "mmd_regularizer"]
