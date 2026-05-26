"""Custom GPyTorch likelihoods for SPT $N$-value modelling.

The operational model in this paper uses a plain
``gpytorch.likelihoods.GaussianLikelihood``. Two custom likelihoods
are provided here as targeted remedies flagged in the limitations
section of the paper, intended for follow-up work:

- :class:`CensoredGaussianLikelihood` — handles the
  $\\Nblow \\leq 100$ upper-tail cap so the optimiser is not
  penalised for predicting above the cap.
"""

from __future__ import annotations

import math

import torch
from gpytorch.likelihoods import GaussianLikelihood


class CensoredGaussianLikelihood(GaussianLikelihood):
    r"""Right-censored Gaussian likelihood.

    For an observation $y$ with cap $c$ (here $c = 100$):

    .. math::

        \log p(y \mid f) =
          \begin{cases}
            \log \mathcal{N}(y \mid f, \sigma^2) & y < c \\
            \log\Phi\!\left(\frac{f - c}{\sigma}\right) & y \geq c
          \end{cases}

    The censored term is the log-survival function of a Gaussian at
    the cap; it does not penalise the model for predicting above the
    cap, which is the desired behaviour for refusal layers logged
    as $N = 100$ in KuniJiban.

    The variational ELBO requires
    :math:`\mathbb{E}_{q(f)}[\log p(y\mid f)]`. The uncensored
    branch has the standard closed form,

    .. math::

        \mathbb{E}_{q(f)}[\log \mathcal{N}(y \mid f, \sigma^2)]
          = -\tfrac{1}{2}\log(2\pi\sigma^2)
            - \frac{(y - \mu_q)^2 + \mathrm{Var}_q[f]}{2\sigma^2},

    where $q(f) = \mathcal{N}(\mu_q, \mathrm{Var}_q[f])$. The censored
    branch :math:`\mathbb{E}_{q(f)}[\log\Phi((f-c)/\sigma)]` does not
    have a closed form; we use the first-order
    "$f$ marginalised into the cdf" approximation,

    .. math::

        \mathbb{E}_{q(f)}\!\left[\log\Phi\!\left(\tfrac{f-c}{\sigma}\right)\right]
          \approx
        \log\Phi\!\left(\tfrac{\mu_q-c}{\sqrt{\sigma^2 + \mathrm{Var}_q[f]}}\right).
    """

    def __init__(self, cap: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.cap = float(cap)

    def expected_log_prob(self, target, input, *params, **kwargs):
        """Compute :math:`E_{q(f)}[\\log p(y \\mid f)]` for a batch of
        :math:`q(f) =` ``input`` and observations :math:`y =` ``target``.

        Args:
            target: ``(n,)`` tensor of observations in raw (or
                standardised — the cap should match) units.
            input: a GPyTorch ``MultivariateNormal`` over the same
                batch (the variational predictive on ``f``).
        """
        mu = input.mean
        var = input.variance
        sigma2 = self.noise  # scalar or (n,) tensor
        eps = 1e-6
        sigma2 = sigma2.clamp_min(eps)

        cap = self.cap
        is_censored = target >= cap

        # Uncensored branch
        log_2pi_sigma2 = torch.log(2 * math.pi * sigma2)
        uncensored = -0.5 * log_2pi_sigma2 - (
            (target - mu).pow(2) + var
        ) / (2.0 * sigma2)

        # Censored branch (first-order approximation)
        sigma_eff = (var + sigma2).clamp_min(eps).sqrt()
        z = (mu - cap) / sigma_eff
        # Numerically stable log Phi(z) = log_ndtr(z)
        log_phi = torch.special.log_ndtr(z)
        censored = log_phi

        return torch.where(is_censored, censored, uncensored)


__all__ = ["CensoredGaussianLikelihood"]
