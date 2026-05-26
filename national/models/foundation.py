"""DKL + SVGP foundation model for nationwide soil property estimation.

The model is a Deep Kernel Learning wrapper around a sparse variational GP.
Raw inputs are mapped through a residual MLP encoder into a 32-dimensional
non-stationary feature space, and the SVGP head operates on those features.

Architectural notes:

- The encoder receives ``(lat, lon, depth_m, *covariates)``. Latitude and
  longitude are passed through random Fourier features so that the spatial
  prior is non-stationary, depth is range-normalized, and covariates are
  expected to be pre-standardized by the covariate registry.
- The SVGP head uses learned inducing locations *in the raw input space*. The
  encoder is applied inside :py:meth:`_DKLApproximateGP.forward`, so the
  variational strategy computes K_uu, K_uX in encoded space. This is the
  standard DKL pattern (Wilson et al. 2016) extended to SVGP.
- The kernel is a scaled Matern over the encoded features. A short-scale
  Matern-3/2 residual term over raw (lat, lon) is added so the model degrades
  gracefully when the encoder underfits an area.
- Regime modulation is FiLM-style: a small embedding lookup produces a
  per-sample bias added to the GP mean and a per-sample log-amplitude added
  to the output scale. Both are differentiable and trained jointly.

The artifact serialized by :py:meth:`FoundationModel.save` is everything
needed to reload and condition on new boring data: encoder weights, SVGP
variational params, kernel hyperparameters, covariate normalization stats,
regime embedding table, and the Hydra config used at training time.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import gpytorch
import torch
from torch import nn


# ----------------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------------


@dataclass
class EncoderSpec:
    """Configuration of the DKL feature encoder (ResMLP)."""

    n_input: int = 24
    n_output: int = 32
    n_layers: int = 6
    hidden: int = 256
    batchnorm: bool = True
    dropout: float = 0.0
    fourier_bands: int = 16
    fourier_scale: float = 4.0
    zero_fourier: bool = False
    """When True, replace the random-Fourier (lat, lon) features with
    zeros in the encoder forward. Used as a covariate-ablation
    counter-test for the feature-importance question 'is the encoder
    just memorising coordinates?'"""


@dataclass
class SVGPSpec:
    """Configuration of the SVGP head."""

    n_inducing: int = 50_000
    learn_inducing: bool = True
    whitened: bool = True
    inducing_init: Literal["random", "kmeans_pp", "kmeans_pp_stratified"] = (
        "kmeans_pp_stratified"
    )
    kernel_type: Literal["matern52", "matern32", "matern12", "rbf"] = "matern52"
    mean_type: Literal["constant", "linear"] = "constant"
    """Mean function over the encoder output.

    ``constant`` is a single learned bias (the historical default). ``linear``
    is ``gpytorch.means.LinearMean`` over the 24/32-D encoder features —
    useful when the target has a strong directional trend in the encoded
    representation (e.g. SPT N increases monotonically with depth).
    """

    likelihood_type: Literal["gaussian", "studentt", "censored"] = "gaussian"
    """Observation likelihood family.

    ``gaussian`` (default) is the historical homoscedastic Gaussian (or,
    when ``NoiseHeadSpec.enabled``, FixedNoiseGaussianLikelihood with
    heteroscedastic per-point noise — that hetero path takes priority
    over likelihood_type).

    ``studentt`` is a Student-t likelihood that natively handles the
    heavy-tail residual structure observed on SPT N (kurtosis ≈ 9.3).
    Conjugacy with the GP prior is lost, so the ELBO's
    ``expected_log_prob`` term falls back to Gauss-Hermite quadrature
    (built-in to gpytorch's ``_OneDimensionalLikelihood`` base).

    ``censored`` uses CensoredGaussianLikelihood from
    ``national.models.likelihoods``, which treats observations at or
    above ``censored_cap`` as right-censored. Targeted at the
    KuniJiban N≤100 cap that creates a structural under-coverage in
    the operational Gaussian model on stiff-layer / refusal data.
    """

    studentt_deg_free: float = 4.0
    """Initial degrees of freedom ν for the Student-t likelihood.

    ν → ∞ recovers Gaussian; ν = 2 has infinite variance (avoid). ν = 4
    is a pragmatic init (kurtosis = 6) — close to the kurtosis ≈ 9 we
    measured on residuals. The parameter is learnable and constrained
    to ν > 2. Ignored when ``likelihood_type != "studentt"``.
    """

    censored_cap: float = 100.0
    """Right-censoring threshold in raw \\Nblow units. Default 100 matches
    the operational SPT cap. Ignored when ``likelihood_type != "censored"``.
    The cap is automatically transformed into standardised target units
    by ``FoundationModel.set_target_stats`` so the likelihood sees the
    cap in the same space as the standardised observations.
    """

    add_residual_geo: bool = True


@dataclass
class NoiseHeadSpec:
    """Heteroscedastic noise head configuration.

    Maps ``(depth_norm, regime_one_hot)`` to ``log_variance``. Adding this
    head turns the model's likelihood from a homoscedastic
    ``GaussianLikelihood`` (single learned noise) into a heteroscedastic
    Gaussian where each query point has its own noise scale. This is the
    right fix for the α=0.50 calibration gap and the per-depth RMSE
    inflation seen in the linear-target runs.
    """

    enabled: bool = False
    hidden: int = 32
    n_layers: int = 2
    """Number of hidden layers in the noise MLP."""
    min_log_var: float = -6.0
    """Clamp on log_variance to keep gradients well-conditioned."""
    max_log_var: float = 3.0


@dataclass
class FoundationSpec:
    """Top-level specification of the foundation model."""

    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    svgp: SVGPSpec = field(default_factory=SVGPSpec)
    regime_dim: int = 8
    depth_scale_m: float = 30.0
    noise_head: NoiseHeadSpec = field(default_factory=NoiseHeadSpec)


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------


class _FourierFeatures(nn.Module):
    """Random Fourier features for the (lat, lon) pair.

    Following NeRF-style positional encoding: each of the two angular inputs
    is expanded to ``2 * fourier_bands`` features (sin and cos of each band).
    The base frequency is set so that one period spans roughly 1/(2^fourier_scale)
    of a degree (~10-100 m at Japanese latitudes).
    """

    def __init__(self, n_bands: int, scale: float) -> None:
        super().__init__()
        # Log-spaced frequencies in [2^0, 2^(n_bands-1)] cycles per scale unit.
        freqs = 2.0 ** torch.linspace(0.0, n_bands - 1, n_bands) * (2.0**scale)
        self.register_buffer("freqs", freqs)

    @property
    def n_features_per_input(self) -> int:
        return int(self.freqs.numel()) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1) or (B,). Output: (B, 2 * n_bands).
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        # broadcast (B, 1) * (n_bands,) -> (B, n_bands)
        scaled = x * self.freqs
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, batchnorm: bool, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.BatchNorm1d(dim) if batchnorm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(dim) if batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        h = self.dropout(h)
        h = self.norm2(self.fc2(h))
        return self.act(x + h)


class ResMLPEncoder(nn.Module):
    """Residual MLP with random-Fourier positional features for (lat, lon).

    Input convention -- the first three columns are special:
        ``x[:, 0]`` = latitude in degrees (EPSG:4326)
        ``x[:, 1]`` = longitude in degrees (EPSG:4326)
        ``x[:, 2]`` = depth in meters from ground surface

    All remaining columns are arbitrary continuous covariates (already
    z-score normalized by the covariate registry).
    """

    def __init__(self, spec: EncoderSpec) -> None:
        super().__init__()
        self.spec = spec
        if spec.n_input < 3:
            raise ValueError(
                f"EncoderSpec.n_input must be >=3 (lat, lon, depth + covariates); "
                f"got {spec.n_input}"
            )

        self.fourier_lat = _FourierFeatures(spec.fourier_bands, spec.fourier_scale)
        self.fourier_lon = _FourierFeatures(spec.fourier_bands, spec.fourier_scale)

        n_covariates = spec.n_input - 3  # lat/lon/depth handled separately
        n_feat = 2 * self.fourier_lat.n_features_per_input + 1 + n_covariates
        self.input_proj = nn.Linear(n_feat, spec.hidden)
        self.input_norm = nn.BatchNorm1d(spec.hidden) if spec.batchnorm else nn.Identity()
        self.input_act = nn.SiLU()

        self.blocks = nn.ModuleList(
            [_ResBlock(spec.hidden, spec.batchnorm, spec.dropout) for _ in range(spec.n_layers)]
        )
        self.output_proj = nn.Linear(spec.hidden, spec.n_output)

        # Set depth scale via spec.fourier_scale neighbor (used in normalize_depth).
        self._depth_scale_m = 30.0  # overridden by FoundationModel constructor

    def set_depth_scale(self, depth_scale_m: float) -> None:
        self._depth_scale_m = float(depth_scale_m)

    def _featurize(self, x: torch.Tensor) -> torch.Tensor:
        # Latitude/longitude in degrees -> radians for Fourier expansion.
        lat_rad = x[:, 0] * (math.pi / 180.0)
        lon_rad = x[:, 1] * (math.pi / 180.0)
        depth_norm = (x[:, 2] / self._depth_scale_m).unsqueeze(-1)

        lat_feat = self.fourier_lat(lat_rad)
        lon_feat = self.fourier_lon(lon_rad)
        if self.spec.zero_fourier:
            lat_feat = torch.zeros_like(lat_feat)
            lon_feat = torch.zeros_like(lon_feat)

        if x.shape[-1] > 3:
            covariates = x[:, 3:]
            feats = torch.cat([lat_feat, lon_feat, depth_norm, covariates], dim=-1)
        else:
            feats = torch.cat([lat_feat, lon_feat, depth_norm], dim=-1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._featurize(x)
        h = self.input_act(self.input_norm(self.input_proj(feats)))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# ----------------------------------------------------------------------------
# SVGP head with DKL
# ----------------------------------------------------------------------------


class _DKLApproximateGP(gpytorch.models.ApproximateGP):
    """SVGP head whose kernel runs on encoder outputs."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        encoder: ResMLPEncoder,
        spec: SVGPSpec,
    ) -> None:
        n_inducing = inducing_points.size(0)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=n_inducing
        )
        strategy_cls = (
            gpytorch.variational.VariationalStrategy
            if spec.whitened
            else gpytorch.variational.UnwhitenedVariationalStrategy
        )
        variational_strategy = strategy_cls(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=spec.learn_inducing,
        )
        super().__init__(variational_strategy)

        self.encoder = encoder
        if spec.mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif spec.mean_type == "linear":
            # LinearMean takes a linear combination of the encoded features,
            # so the bias is captured by the trailing constant of the encoder
            # and the slope is learned per feature dimension.
            self.mean_module = gpytorch.means.LinearMean(input_size=encoder.spec.n_output)
            # Zero-init the weights so training starts equivalent to
            # ConstantMean (bias only). LinearMean defaults to N(0, 1) weights
            # which, multiplied by encoder output norms ~5, produces mean
            # predictions an order of magnitude larger than the standardized
            # target (|y| ~ 1) and pushes the SVGP variational strategy
            # into a non-PSD Cholesky right out of the gate.
            with torch.no_grad():
                self.mean_module.weights.zero_()
        else:
            raise ValueError(f"Unknown mean_type: {spec.mean_type}")

        ard = encoder.spec.n_output
        if spec.kernel_type == "matern52":
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard)
        elif spec.kernel_type == "matern32":
            base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard)
        elif spec.kernel_type == "matern12":
            base = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ard)
        elif spec.kernel_type == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=ard)
        else:
            raise ValueError(f"Unknown kernel_type: {spec.kernel_type}")
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

        if spec.add_residual_geo:
            # Short-scale residual in raw (lat, lon) degrees, kept separate from
            # the DKL kernel so it cannot interfere with encoder learning.
            self.residual_geo = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2)
            )
        else:
            self.residual_geo = None

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        z = self.encoder(x)
        covar = self.covar_module(z)
        if self.residual_geo is not None:
            covar = covar + self.residual_geo(x[..., :2])
        return gpytorch.distributions.MultivariateNormal(self.mean_module(z), covar)


# ----------------------------------------------------------------------------
# Regime modulation (FiLM)
# ----------------------------------------------------------------------------


class _NoiseHead(nn.Module):
    """MLP that predicts per-point log-noise-variance from (depth, regime).

    Input layout matches ``BoringDataset``: the first three encoder input
    columns are ``[lat, lon, depth]`` and (when ``regime_one_hot`` is on)
    the trailing ``regime_dim`` columns are a one-hot indicator. We use
    only depth and the regime one-hot here so the noise head is
    invariant to spatial covariates the encoder already exploits.
    """

    def __init__(self, spec: NoiseHeadSpec, regime_dim: int) -> None:
        super().__init__()
        self.spec = spec
        self.regime_dim = regime_dim
        in_dim = 1 + regime_dim  # depth_norm + regime one-hot
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(spec.n_layers):
            layers.extend([nn.Linear(prev, spec.hidden), nn.SiLU()])
            prev = spec.hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        # Initialise the final layer's bias to 0 (output log_var ~ 0,
        # variance ~ 1) so the initial heteroscedastic noise matches a
        # plain GaussianLikelihood. Keep the final layer's weights at
        # the default Kaiming-style random init: zeroing them too would
        # block all gradient flow through the head (the post-activation
        # path goes weight*input + bias; a zero weight matrix kills the
        # backprop signal to earlier layers).
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-2)

    def forward(
        self,
        depth_norm: torch.Tensor,
        regime_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([depth_norm.unsqueeze(-1), regime_one_hot], dim=-1)
        log_var = self.net(x).squeeze(-1)
        return log_var.clamp(self.spec.min_log_var, self.spec.max_log_var)


class _RegimeFiLM(nn.Module):
    """Per-regime mean bias and log-scale modulation.

    A standard FiLM block would multiply features by a regime-dependent
    affine transform. For a GP we instead modulate the predictive mean and
    standard deviation, which is equivalent on the marginals and keeps the
    SVGP variational distribution clean.
    """

    def __init__(self, n_regimes: int) -> None:
        super().__init__()
        self.bias = nn.Embedding(n_regimes, 1)
        self.log_scale = nn.Embedding(n_regimes, 1)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.log_scale.weight)

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        regime_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.bias(regime_codes).squeeze(-1)
        s = self.log_scale(regime_codes).squeeze(-1).exp()
        return mean + b, std * s


# ----------------------------------------------------------------------------
# Foundation model
# ----------------------------------------------------------------------------


@dataclass
class FoundationPrediction:
    mean: torch.Tensor
    std: torch.Tensor
    encoded: torch.Tensor


class FoundationModel(nn.Module):
    """Top-level foundation model bundling encoder, SVGP, regime FiLM."""

    ARTIFACT_VERSION = 1

    def __init__(self, spec: FoundationSpec, inducing_points: torch.Tensor) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = ResMLPEncoder(spec.encoder)
        self.encoder.set_depth_scale(spec.depth_scale_m)
        self.gp = _DKLApproximateGP(inducing_points, self.encoder, spec.svgp)
        # FixedNoiseGaussianLikelihood expects a noise tensor at every
        # forward call when heteroscedastic; we toggle between that and a
        # plain learned-noise GaussianLikelihood based on the spec.
        if spec.noise_head.enabled:
            # Heteroscedastic Gaussian path. Takes priority over likelihood_type.
            # Dummy single-element noise; replaced per-batch by the trainer.
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.ones(1), learn_additional_noise=True
            )
            self.noise_head: _NoiseHead | None = _NoiseHead(
                spec.noise_head, regime_dim=spec.regime_dim
            )
        elif spec.svgp.likelihood_type == "studentt":
            from gpytorch.constraints import GreaterThan

            self.likelihood = gpytorch.likelihoods.StudentTLikelihood(
                deg_free_constraint=GreaterThan(2.0),
            )
            # Initialize ν at the spec-supplied value (default 4). The
            # constraint above keeps it from collapsing to ≤2 (which would
            # make the marginal variance infinite).
            with torch.no_grad():
                self.likelihood.deg_free = torch.tensor(float(spec.svgp.studentt_deg_free))
            self.noise_head = None
        elif spec.svgp.likelihood_type == "censored":
            from national.models.likelihoods import CensoredGaussianLikelihood

            # Cap initially +inf (uncensored equivalent); the real cap is
            # set by set_target_stats once the trainer knows the
            # standardisation moments.
            self.likelihood = CensoredGaussianLikelihood(cap=float("inf"))
            self.register_buffer(
                "raw_cap", torch.tensor(float(spec.svgp.censored_cap))
            )
            self.noise_head = None
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.noise_head = None
        self.regime_film = _RegimeFiLM(spec.regime_dim)

        # Standardization stats for the target. Set by the trainer at fit start.
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    # -- training-time interface ---------------------------------------------
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.gp(x)

    def predict_noise_variance(self, x: torch.Tensor) -> torch.Tensor | None:
        """Return per-point noise variance, or ``None`` if homoscedastic.

        The trainer feeds this tensor to ``FixedNoiseGaussianLikelihood`` so
        the ELBO accounts for heteroscedastic observation noise.
        ``x`` has the canonical BoringDataset layout
        ``[lat, lon, depth, *features, *regime_one_hot]``; we extract the
        depth column and the trailing regime one-hot block.
        """
        if self.noise_head is None:
            return None
        depth_raw = x[:, 2]
        depth_norm = depth_raw / float(self.spec.depth_scale_m)
        regime_dim = self.spec.regime_dim
        if x.size(-1) < 3 + regime_dim:
            # Encoder isn't using the one-hot extension yet; substitute a
            # uniform regime indicator so the noise head still trains.
            regime_one_hot = (
                torch.ones(x.size(0), regime_dim, device=x.device) / regime_dim
            )
        else:
            regime_one_hot = x[:, -regime_dim:]
        log_var = self.noise_head(depth_norm, regime_one_hot)
        return log_var.exp()

    def set_target_stats(self, mean: float | torch.Tensor, std: float | torch.Tensor) -> None:
        self.target_mean = torch.as_tensor(mean, dtype=self.target_mean.dtype).reshape(1)
        self.target_std = torch.as_tensor(std, dtype=self.target_std.dtype).reshape(1)
        if (self.target_std <= 0).any():
            raise ValueError("target_std must be strictly positive.")
        # Propagate the standardisation onto the censored cap. The
        # CensoredGaussianLikelihood expects ``cap`` in the same space as
        # the observations it receives at expected_log_prob time, and the
        # trainer passes standardised y. ``raw_cap`` is set in __init__
        # only for likelihood_type == "censored".
        from national.models.likelihoods import CensoredGaussianLikelihood

        if hasattr(self, "raw_cap") and isinstance(
            self.likelihood, CensoredGaussianLikelihood
        ):
            self.likelihood.cap = float(
                (float(self.raw_cap) - float(self.target_mean.item()))
                / float(self.target_std.item())
            )

    @torch.no_grad()
    def calibrate_regime_film(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_codes: torch.Tensor,
        *,
        min_samples_per_regime: int = 30,
    ) -> None:
        """Fit the regime FiLM block as a post-hoc calibration step.

        The training loop applies the SVGP loss to the bare GP output
        (``self.gp(x)``); the FiLM block is therefore *not* updated by
        gradients during ``fit``. Run this method after training on the
        same dataset (or a held-out one) to fit the per-regime bias and
        log-scale in closed form:

            bias(g)  = mean_i (y_i - mu_F(x_i)) for regime g
            scale(g) = std_i (y_i - mu_F(x_i)) / mean_i sigma_F(x_i)

        Regimes with fewer than ``min_samples_per_regime`` observations
        keep the identity calibration (bias=0, scale=1) to avoid noisy
        per-regime fits on tiny strata.
        """
        if x.shape[0] != y.shape[0] or x.shape[0] != regime_codes.shape[0]:
            raise ValueError(
                f"shape mismatch: x={x.shape}, y={y.shape}, regime={regime_codes.shape}"
            )
        param = next(self.parameters())
        x_dev = x.to(device=param.device, dtype=param.dtype)
        # Predict without the FiLM modulation (which is identity at init).
        self.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            f = self.gp(x_dev)
            y_pred = self.likelihood(f)
            mean_pred = (y_pred.mean * self.target_std + self.target_mean).cpu()
            std_pred = (y_pred.stddev * self.target_std).cpu()
        y_cpu = y.detach().cpu().float()
        regime_cpu = regime_codes.detach().cpu().long()
        residuals = y_cpu - mean_pred

        n_regimes = int(self.regime_film.bias.num_embeddings)
        new_bias = torch.zeros(n_regimes)
        new_logscale = torch.zeros(n_regimes)
        for g in range(n_regimes):
            mask = regime_cpu == g
            n_g = int(mask.sum().item())
            if n_g < min_samples_per_regime:
                continue
            r_g = residuals[mask]
            sig_g = std_pred[mask]
            new_bias[g] = r_g.mean()
            std_ratio = (r_g.std().clamp_min(1e-3) / sig_g.mean().clamp_min(1e-3)).clamp(0.25, 4.0)
            new_logscale[g] = torch.log(std_ratio)
        # Move FiLM weights to the model's device before writing.
        device = self.regime_film.bias.weight.device
        self.regime_film.bias.weight.data.copy_(new_bias.view(-1, 1).to(device))
        self.regime_film.log_scale.weight.data.copy_(new_logscale.view(-1, 1).to(device))

    # -- inference interface -------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        regime_codes: torch.Tensor | None = None,
    ) -> FoundationPrediction:
        self.eval()
        self.likelihood.eval()
        # Align the query tensor with the model's current device so callers
        # can pass CPU tensors to a CUDA / MPS-resident model without manual
        # ``.to()`` boilerplate.
        param = next(self.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        if regime_codes is not None:
            regime_codes = regime_codes.to(device=param.device)
        with gpytorch.settings.fast_pred_var():
            f = self.gp(x)
            if isinstance(self.likelihood, gpytorch.likelihoods.StudentTLikelihood):
                # Student-t marginal has no closed form combined with a
                # Gaussian q(f). gpytorch returns a quadrature-augmented
                # distribution with shape (num_locs, B), which is not what
                # downstream callers expect. Use the analytic moments:
                #   E[y|x]   = E_q[f]              (GP posterior mean)
                #   Var[y|x] = σ² ν/(ν-2) + Var_q[f]
                # where ν > 2 is enforced by the likelihood's constraint.
                f_mean = f.mean
                f_var = f.variance
                noise = self.likelihood.noise.squeeze()
                df = self.likelihood.deg_free.squeeze()
                t_scale = noise * df / (df - 2.0).clamp_min(1e-3)
                mean = f_mean
                mean_std = (t_scale + f_var).clamp_min(1e-12).sqrt()
            else:
                y = self.likelihood(f)
                mean_std = y.stddev
                mean = y.mean
        encoded = self.encoder(x)
        if regime_codes is not None:
            mean, mean_std = self.regime_film(mean, mean_std, regime_codes)
        # Denormalize and ship results back to CPU for downstream consumption.
        mean = (mean * self.target_std + self.target_mean).cpu()
        std = (mean_std * self.target_std).cpu()
        return FoundationPrediction(mean=mean, std=std, encoded=encoded.cpu())

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.ARTIFACT_VERSION,
                "spec": asdict(self.spec),
                "state_dict": self.state_dict(),
                "inducing_shape": tuple(self.gp.variational_strategy.inducing_points.shape),
            },
            path,
        )
        meta = path.with_suffix(".meta.json")
        meta.write_text(
            json.dumps(
                {
                    "version": self.ARTIFACT_VERSION,
                    "spec": asdict(self.spec),
                    "inducing_shape": list(self.gp.variational_strategy.inducing_points.shape),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path, *, map_location: str | torch.device = "cpu") -> "FoundationModel":
        path = Path(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("version") != cls.ARTIFACT_VERSION:
            raise ValueError(
                f"Incompatible artifact version: {payload.get('version')} "
                f"(expected {cls.ARTIFACT_VERSION})"
            )
        spec_dict = payload["spec"]
        # Older artifacts predate NoiseHeadSpec; default to disabled so they
        # still load correctly.
        noise_dict = spec_dict.get(
            "noise_head", {"enabled": False, "hidden": 32, "n_layers": 2, "min_log_var": -6.0, "max_log_var": 3.0}
        )
        spec = FoundationSpec(
            encoder=EncoderSpec(**spec_dict["encoder"]),
            svgp=SVGPSpec(**spec_dict["svgp"]),
            regime_dim=spec_dict["regime_dim"],
            depth_scale_m=spec_dict["depth_scale_m"],
            noise_head=NoiseHeadSpec(**noise_dict),
        )
        inducing_shape = payload["inducing_shape"]
        inducing_points = torch.zeros(inducing_shape, dtype=torch.float32)
        model = cls(spec, inducing_points)
        model.load_state_dict(payload["state_dict"])
        return model


# ----------------------------------------------------------------------------
# Inducing point initialization
# ----------------------------------------------------------------------------


def init_inducing_points(
    x: torch.Tensor,
    n_inducing: int,
    *,
    method: Literal["random", "kmeans_pp", "kmeans_pp_stratified"] = "kmeans_pp_stratified",
    regime_codes: torch.Tensor | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Pick ``n_inducing`` initial inducing-point locations from ``x``.

    For ``kmeans_pp_stratified``, ``regime_codes`` must be provided and the
    inducing point budget is allocated per-regime proportional to the square
    root of the regime frequency. This prevents the dominant alluvial regime
    from monopolizing the inducing points and starving rare regimes.
    """
    n_data = x.size(0)
    if n_inducing > n_data:
        raise ValueError(
            f"n_inducing ({n_inducing}) must be <= n_data ({n_data})."
        )
    g = torch.Generator(device=x.device).manual_seed(seed)

    if method == "random":
        idx = torch.randperm(n_data, generator=g, device=x.device)[:n_inducing]
        return x[idx].clone()

    if method == "kmeans_pp":
        return _kmeans_pp(x, n_inducing, generator=g)

    if method == "kmeans_pp_stratified":
        if regime_codes is None:
            raise ValueError("kmeans_pp_stratified requires regime_codes.")
        return _kmeans_pp_stratified(x, regime_codes, n_inducing, generator=g)

    raise ValueError(f"Unknown method: {method}")


def _kmeans_pp(
    x: torch.Tensor, n_inducing: int, *, generator: torch.Generator
) -> torch.Tensor:
    n_data, d = x.shape
    chosen = torch.empty((n_inducing, d), dtype=x.dtype, device=x.device)
    first = torch.randint(0, n_data, (1,), generator=generator, device=x.device).item()
    chosen[0] = x[first]
    # Cumulative minimum squared distance to chosen set.
    dist_sq = ((x - chosen[0]) ** 2).sum(dim=-1)
    for i in range(1, n_inducing):
        probs = dist_sq / dist_sq.sum().clamp_min(1e-12)
        pick = torch.multinomial(probs, 1, generator=generator).item()
        chosen[i] = x[pick]
        new_dist = ((x - chosen[i]) ** 2).sum(dim=-1)
        dist_sq = torch.minimum(dist_sq, new_dist)
    return chosen


def _kmeans_pp_stratified(
    x: torch.Tensor,
    regime_codes: torch.Tensor,
    n_inducing: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    unique, counts = torch.unique(regime_codes, return_counts=True)
    # sqrt-tempered allocation -> dominant regimes get fewer, rare regimes more.
    weights = counts.float().sqrt()
    weights = weights / weights.sum()
    budgets = (weights * n_inducing).round().long()
    # Adjust for rounding so total == n_inducing.
    diff = n_inducing - int(budgets.sum().item())
    if diff != 0:
        budgets[0] += diff
    # Clamp budgets to actual regime sizes.
    budgets = torch.minimum(budgets, counts)
    # Fix any deficit caused by clamping by topping up from the largest regime.
    deficit = n_inducing - int(budgets.sum().item())
    if deficit > 0:
        order = counts.argsort(descending=True)
        for idx in order.tolist():
            slack = int((counts[idx] - budgets[idx]).item())
            take = min(slack, deficit)
            budgets[idx] += take
            deficit -= take
            if deficit == 0:
                break

    pieces = []
    for code, budget in zip(unique.tolist(), budgets.tolist(), strict=True):
        if budget == 0:
            continue
        mask = regime_codes == code
        x_r = x[mask]
        pieces.append(_kmeans_pp(x_r, budget, generator=generator))
    out = torch.cat(pieces, dim=0)
    # Shuffle so the variational strategy doesn't see a regime-sorted layout.
    perm = torch.randperm(out.size(0), generator=generator, device=out.device)
    return out[perm]


__all__ = [
    "EncoderSpec",
    "SVGPSpec",
    "FoundationSpec",
    "ResMLPEncoder",
    "FoundationModel",
    "FoundationPrediction",
    "init_inducing_points",
]
