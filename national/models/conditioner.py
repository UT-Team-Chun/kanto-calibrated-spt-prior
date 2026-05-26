"""Online Bayesian conditioning on a trained foundation model.

This is the "foundation model as prior" path that gives the project its
name. The trained :class:`FoundationModel` provides a calibrated global
prior ``p(y | x)`` over soil properties; given a small set of fresh boring
observations from a user (e.g. a new site survey not seen during training),
:class:`FoundationConditioner` returns a refined posterior at any query
point by combining the global prior with a *local* exact GP fit on the
residuals between the prior mean and the new observations.

Mathematical recipe::

    1. Predict foundation marginals at the new borings:
           mu_F(x_i), sigma_F(x_i)^2  for i = 1..n_new
    2. Compute residuals:
           r_i = y_i - mu_F(x_i)
    3. Fit an exact GP on the residuals with a short-scale kernel
       (Matern-3/2 in projected metric coordinates), inflating the
       noise variance by sigma_F(x_i)^2 so the local fit accounts for
       foundation uncertainty (heteroscedastic likelihood).
    4. At each query point x_q, the local GP yields delta(x_q),
       sigma_local(x_q). The refined posterior is
           mu(x_q) = mu_F(x_q) + delta(x_q)
           sigma(x_q)^2 = sigma_F(x_q)^2 + sigma_local(x_q)^2
       (independence approximation -- residual GP is short-scale so its
       covariance with the SVGP prior is small).

The local GP cost is O(n_new^3). We cap ``n_new`` via
``max_local_points`` and a ``local_radius_km`` neighborhood filter so the
runtime stays bounded at scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from national.models.foundation import FoundationModel
from shared.geo.distance import haversine_km


@dataclass
class ConditionResult:
    """Posterior at query points after conditioning on new borings."""

    mean: torch.Tensor  # (N_query,)
    std: torch.Tensor  # (N_query,)
    q05: torch.Tensor
    q95: torch.Tensor
    regime: list[str]


class _ResidualGP(gpytorch.models.ExactGP):
    """Short-scale Matern-3/2 ExactGP over (lat_utm_km, lon_utm_km, depth_m)."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # ARD over (x, y, z) lets the kernel pick a separate length-scale per axis.
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1))
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class FoundationConditioner:
    """Refined posterior = foundation prior + local residual GP."""

    DEFAULT_LOCAL_ITER = 30
    DEFAULT_LR = 0.1

    def __init__(self, model: FoundationModel) -> None:
        self.model = model

    def condition(
        self,
        new_borings: pd.DataFrame,
        query_xyz: torch.Tensor,
        *,
        local_radius_km: float = 5.0,
        max_local_points: int = 2000,
        feature_columns: list[str] | None = None,
        target_column: str = "n_value",
    ) -> ConditionResult:
        """Return the locally-updated posterior at query points.

        Args:
            new_borings: DataFrame with required columns
                ``[latitude_deg, longitude_deg, depth_from_surface, n_value]``
                plus any ``feature_columns`` the foundation encoder expects.
            query_xyz: ``(N_query, 3)`` tensor with columns
                ``[lat, lon, depth]``. Any extra feature columns the encoder
                needs must be supplied via the registry baked into the model.
            local_radius_km: radius around each query point used to select
                the conditioning subset (haversine in km).
            max_local_points: cap on the number of points fed to the local GP.
            feature_columns: optional ordered list of covariate column names
                to feed the foundation encoder. If ``None``, no extra
                features are passed (the encoder must have ``n_input == 3``).
            target_column: name of the regression target in ``new_borings``.

        Returns:
            :class:`ConditionResult` with mean, std, 5/95% quantiles and a
            placeholder regime list (filled later by the API layer).
        """
        if query_xyz.ndim != 2 or query_xyz.size(-1) != 3:
            raise ValueError(
                f"query_xyz must have shape (N_query, 3); got {tuple(query_xyz.shape)}"
            )
        required = {"latitude_deg", "longitude_deg", "depth_from_surface", target_column}
        missing = required - set(new_borings.columns)
        if missing:
            raise KeyError(f"new_borings missing columns: {sorted(missing)!r}")
        feature_columns = list(feature_columns or [])

        device = next(self.model.parameters()).device

        # --- 1. Foundation prediction at the new borings ----------------------
        x_new = self._build_features(new_borings, feature_columns).to(device)
        with torch.no_grad():
            pred_new = self.model.predict(x_new)
        mu_new = pred_new.mean.cpu()
        sigma_new = pred_new.std.cpu()
        y_new = torch.as_tensor(
            new_borings[target_column].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        residuals = y_new - mu_new

        # --- 2. Foundation prediction at the queries --------------------------
        x_query = query_xyz.to(dtype=torch.float32, device=device)
        # If the encoder expects covariates we need a registry / supplied tensor;
        # the API layer will fill these in. For now we accept that condition()
        # is called either with an encoder that requires no covariates beyond
        # (lat, lon, depth) or with a query_xyz that already includes them.
        if x_query.size(-1) != self.model.encoder.spec.n_input:
            if x_query.size(-1) < self.model.encoder.spec.n_input:
                pad = torch.zeros(
                    x_query.size(0),
                    self.model.encoder.spec.n_input - x_query.size(-1),
                    device=device,
                )
                x_query = torch.cat([x_query, pad], dim=-1)
            else:
                raise ValueError(
                    f"query_xyz has more columns ({x_query.size(-1)}) than the "
                    f"encoder expects ({self.model.encoder.spec.n_input})."
                )
        with torch.no_grad():
            pred_q = self.model.predict(x_query)
        mu_q = pred_q.mean.cpu()
        sigma_q = pred_q.std.cpu()

        # --- 3. Local residual GP -------------------------------------------
        keep = self._neighborhood_subset(
            new_borings, query_xyz, local_radius_km, max_local_points
        )
        if keep.size == 0:
            # No nearby observations -- the foundation prior is unchanged.
            delta_mean = torch.zeros(mu_q.shape)
            delta_std = torch.zeros(mu_q.shape)
        else:
            delta_mean, delta_std = self._fit_predict_residual(
                new_borings.iloc[keep],
                residuals[torch.from_numpy(keep)],
                sigma_new[torch.from_numpy(keep)],
                query_xyz,
            )

        # --- 4. Combine -----------------------------------------------------
        # Use the *standardized* foundation std, denormalize later. The
        # foundation already denormalizes inside predict(), so mu_q and
        # sigma_q are in the original target units. The residual GP works
        # on raw residuals, so we add directly.
        post_mean = mu_q + delta_mean
        post_var = (sigma_q.pow(2) + delta_std.pow(2)).clamp_min(1e-12)
        post_std = post_var.sqrt()
        z95 = float(norm.ppf(0.95))
        q05 = post_mean - z95 * post_std
        q95 = post_mean + z95 * post_std
        return ConditionResult(
            mean=post_mean,
            std=post_std,
            q05=q05,
            q95=q95,
            regime=[""] * post_mean.shape[0],
        )

    # ---------------------------------------------------------------- helpers
    def _build_features(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> torch.Tensor:
        cols = [
            df["latitude_deg"].to_numpy(dtype=np.float32),
            df["longitude_deg"].to_numpy(dtype=np.float32),
            df["depth_from_surface"].to_numpy(dtype=np.float32),
        ]
        for c in feature_columns:
            if c not in df.columns:
                raise KeyError(f"feature column {c!r} missing from new_borings")
            cols.append(df[c].to_numpy(dtype=np.float32))
        x = np.stack(cols, axis=1).astype(np.float32)
        # Pad to the encoder's expected width if extra columns are required.
        expected = self.model.encoder.spec.n_input
        if x.shape[1] < expected:
            x = np.concatenate(
                [x, np.zeros((x.shape[0], expected - x.shape[1]), dtype=np.float32)], axis=1
            )
        return torch.from_numpy(x)

    def _neighborhood_subset(
        self,
        new_borings: pd.DataFrame,
        query_xyz: torch.Tensor,
        radius_km: float,
        cap: int,
    ) -> np.ndarray:
        """Return indices into ``new_borings`` of points near any query."""
        lats = new_borings["latitude_deg"].to_numpy(dtype=np.float64)
        lons = new_borings["longitude_deg"].to_numpy(dtype=np.float64)
        q_lat = query_xyz[:, 0].cpu().numpy().astype(np.float64)
        q_lon = query_xyz[:, 1].cpu().numpy().astype(np.float64)
        keep = np.zeros(len(new_borings), dtype=bool)
        for ql, qln in zip(q_lat, q_lon, strict=True):
            d = np.array(
                [haversine_km(la, lo, ql, qln) for la, lo in zip(lats, lons, strict=True)]
            )
            keep |= d <= radius_km
            if keep.sum() >= cap:
                break
        idx = np.where(keep)[0]
        if idx.size > cap:
            # Randomly subsample to stay within the budget (cubic-time GP).
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=cap, replace=False)
            idx.sort()
        return idx

    def _fit_predict_residual(
        self,
        df_local: pd.DataFrame,
        residuals: torch.Tensor,
        foundation_std: torch.Tensor,
        query_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train a short-scale Matern GP on residuals; predict at queries."""
        # Convert (lat, lon, depth) to a metric coordinate system so the
        # kernel length-scale is interpretable in km. Equirectangular is fine
        # for a local fit -- the curvature error over ~5 km is negligible.
        ref_lat = float(df_local["latitude_deg"].mean())
        ref_lon = float(df_local["longitude_deg"].mean())
        train_x = self._to_local_metric(
            df_local["latitude_deg"].to_numpy(),
            df_local["longitude_deg"].to_numpy(),
            df_local["depth_from_surface"].to_numpy(),
            ref_lat,
            ref_lon,
        )
        train_x = torch.from_numpy(train_x).float()
        train_y = residuals.float()

        # Heteroscedastic noise = foundation uncertainty^2. Use FixedNoise.
        noise = foundation_std.float().pow(2).clamp_min(1e-3)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise, learn_additional_noise=True
        )
        model = _ResidualGP(train_x, train_y, likelihood)

        model.train()
        likelihood.train()
        # gpytorch.models.ExactGP stores the likelihood as a submodule, so
        # model.parameters() already covers it -- adding likelihood.parameters()
        # would double-count and trigger Adam's duplicate-parameter warning.
        opt = torch.optim.Adam(list(model.parameters()), lr=self.DEFAULT_LR)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.DEFAULT_LOCAL_ITER):
            opt.zero_grad(set_to_none=True)
            out = model(train_x)
            loss = -mll(out, train_y)
            loss.backward()
            opt.step()

        # Predict at queries.
        q_xyz = self._to_local_metric(
            query_xyz[:, 0].cpu().numpy(),
            query_xyz[:, 1].cpu().numpy(),
            query_xyz[:, 2].cpu().numpy(),
            ref_lat,
            ref_lon,
        )
        q_xyz_t = torch.from_numpy(q_xyz).float()
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # FixedNoise requires noise at prediction points; use 0 (no extra noise on the latent).
            pred_noise = torch.zeros(q_xyz_t.size(0))
            pred = likelihood(model(q_xyz_t), noise=pred_noise)
            return pred.mean, pred.stddev

    @staticmethod
    def _to_local_metric(
        lats: np.ndarray,
        lons: np.ndarray,
        depths: np.ndarray,
        ref_lat: float,
        ref_lon: float,
    ) -> np.ndarray:
        """Convert (lat, lon, depth) to local (km_x, km_y, m_z) around a ref point."""
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * max(0.1, math.cos(math.radians(ref_lat)))
        x = (lons - ref_lon) * km_per_deg_lon
        y = (lats - ref_lat) * km_per_deg_lat
        z = depths * 0.001  # depth in km so the ARD lengthscales share a unit
        return np.stack([x, y, z], axis=1).astype(np.float32)


__all__ = ["ConditionResult", "FoundationConditioner"]
