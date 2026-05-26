"""FastAPI router for national foundation-model endpoints.

The router wraps:

- ``FoundationModel.predict`` -- point and polygon inference.
- ``FoundationConditioner.condition`` -- online Bayesian update.
- A small model registry that loads exactly one artifact per process.

The model is loaded on app startup via :func:`set_model` (typically called
from the FastAPI app factory once the artifact path is known). Tests can
inject a tiny synthetic model the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from national.models.conditioner import FoundationConditioner
from national.models.foundation import FoundationModel
from national.tiling.regime_classifier import Regime

router = APIRouter()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictPointRequest(BaseModel):
    """Point prediction request in WGS84 coordinates."""

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    depth: float = Field(..., ge=0.0)


class PredictPointResponse(BaseModel):
    """Point prediction response with uncertainty and regime metadata."""

    mean: float
    std: float
    q05: float
    q95: float
    regime: str


class PredictBatchRequest(BaseModel):
    """Vectorized point prediction request."""

    lats: list[float] = Field(min_length=1)
    lons: list[float] = Field(min_length=1)
    depths: list[float] = Field(min_length=1)


class PredictBatchResponse(BaseModel):
    mean: list[float]
    std: list[float]
    q05: list[float]
    q95: list[float]


class PredictPolygonRequest(BaseModel):
    """Polygon prediction request for one or more depth slices."""

    geojson: dict[str, Any]
    depths: list[float] = Field(min_length=1)
    resolution_m: float = Field(default=100.0, gt=0.0)


class PredictPolygonResponse(BaseModel):
    """Polygon prediction artifact response."""

    zarr_path: str | None = None
    cog_paths: list[str] = Field(default_factory=list)


class BoringObservation(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    depth: float = Field(..., ge=0.0)
    value: float


class QueryPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    depth: float = Field(..., ge=0.0)


class ConditionRequest(BaseModel):
    new_borings: list[BoringObservation] = Field(min_length=1)
    query_points: list[QueryPoint] = Field(min_length=1)
    local_radius_km: float = Field(default=5.0, gt=0.0)
    max_local_points: int = Field(default=2000, ge=10, le=10_000)


class ConditionResponse(BaseModel):
    mean: list[float]
    std: list[float]
    q05: list[float]
    q95: list[float]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str
    training_date: str
    regimes_covered: list[str]
    encoder_dim: int
    n_inducing: int


# --------------------------------------------------------------------------- #
# Model registry (single-instance per process)
# --------------------------------------------------------------------------- #
@dataclass
class _ServingState:
    model: FoundationModel | None = None
    model_id: str = "unset"
    training_date: str = ""
    artifact_path: Path | None = None


_state = _ServingState()


def set_model(
    model: FoundationModel,
    *,
    model_id: str = "national_v1",
    training_date: str | None = None,
    artifact_path: Path | None = None,
) -> None:
    """Register the served foundation model. Call once at app startup."""
    _state.model = model.eval()
    _state.model_id = model_id
    _state.training_date = training_date or datetime.now(timezone.utc).isoformat()
    _state.artifact_path = artifact_path


def _get_model() -> FoundationModel:
    if _state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Foundation model not loaded. Call set_model() at app startup.",
        )
    return _state.model


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _build_xyz(lats, lons, depths, *, encoder_n_input: int) -> torch.Tensor:
    x = torch.stack(
        [
            torch.as_tensor(lats, dtype=torch.float32),
            torch.as_tensor(lons, dtype=torch.float32),
            torch.as_tensor(depths, dtype=torch.float32),
        ],
        dim=1,
    )
    if x.size(-1) < encoder_n_input:
        pad = torch.zeros(x.size(0), encoder_n_input - x.size(-1))
        x = torch.cat([x, pad], dim=-1)
    return x


def _denorm_quantile(
    mean: torch.Tensor, std: torch.Tensor, alpha: float
) -> torch.Tensor:
    from scipy.stats import norm

    return mean + float(norm.ppf(alpha)) * std


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post("/v1/predict/point", response_model=PredictPointResponse)
async def predict_point(req: PredictPointRequest) -> PredictPointResponse:
    model = _get_model()
    x = _build_xyz([req.lat], [req.lon], [req.depth], encoder_n_input=model.encoder.spec.n_input)
    with torch.no_grad():
        pred = model.predict(x)
    m = float(pred.mean[0])
    s = float(pred.std[0])
    return PredictPointResponse(
        mean=m,
        std=s,
        q05=float(_denorm_quantile(pred.mean, pred.std, 0.05)[0]),
        q95=float(_denorm_quantile(pred.mean, pred.std, 0.95)[0]),
        regime=Regime.UNKNOWN.name.lower(),  # regime classifier wired in Phase B-9b
    )


@router.post("/v1/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    if not (len(req.lats) == len(req.lons) == len(req.depths)):
        raise HTTPException(status_code=400, detail="lats/lons/depths must have equal length")
    model = _get_model()
    x = _build_xyz(req.lats, req.lons, req.depths, encoder_n_input=model.encoder.spec.n_input)
    with torch.no_grad():
        pred = model.predict(x)
    q05 = _denorm_quantile(pred.mean, pred.std, 0.05)
    q95 = _denorm_quantile(pred.mean, pred.std, 0.95)
    return PredictBatchResponse(
        mean=pred.mean.tolist(),
        std=pred.std.tolist(),
        q05=q05.tolist(),
        q95=q95.tolist(),
    )


@router.post("/v1/predict/polygon", response_model=PredictPolygonResponse)
async def predict_polygon(req: PredictPolygonRequest) -> PredictPolygonResponse:
    """Predict over a GeoJSON polygon for the requested depths.

    The endpoint synthesizes a TileBounds from the polygon's bounding box
    and routes through :class:`PredictionEngine` for batched inference.
    For Phase D-2 the response carries a Zarr artifact path; COG slice
    generation is deferred to the offline export pipeline.
    """
    from national.prediction.engine import GridSpec, PredictionEngine
    from national.tiling.tile_manager import TileBounds, TileManager

    try:
        coords = _polygon_coords(req.geojson)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    lat_min, lon_min = min(c[1] for c in coords), min(c[0] for c in coords)
    lat_max, lon_max = max(c[1] for c in coords), max(c[0] for c in coords)
    if lat_max <= lat_min or lon_max <= lon_min:
        raise HTTPException(status_code=400, detail="degenerate polygon bounds")

    model = _get_model()
    tm = TileManager(
        region_bbox=(lat_min, lon_min, lat_max, lon_max),
        tile_size_km=80,
        halo_km=0,
    )
    grid = GridSpec(
        resolution_m=req.resolution_m,
        depths_m=tuple(req.depths),
        batch_size_cells=50_000,
    )
    engine = PredictionEngine(model=model, registry=None, tile_manager=tm, grid=grid)
    out_dir = Path(_state.artifact_path or Path("data/predictions/api")).parent / "polygon"
    out_dir.mkdir(parents=True, exist_ok=True)
    cube_dir = out_dir / f"polygon_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    engine.predict_cube(cube_dir)
    return PredictPolygonResponse(
        zarr_path=str(cube_dir),
        cog_paths=[],  # write_cog_slices is an offline tool; not run inline.
    )


@router.post("/v1/condition", response_model=ConditionResponse)
async def condition(req: ConditionRequest) -> ConditionResponse:
    model = _get_model()
    df = pd.DataFrame(
        {
            "latitude_deg": [b.lat for b in req.new_borings],
            "longitude_deg": [b.lon for b in req.new_borings],
            "depth_from_surface": [b.depth for b in req.new_borings],
            "n_value": [b.value for b in req.new_borings],
        }
    )
    queries = torch.tensor(
        [[q.lat, q.lon, q.depth] for q in req.query_points], dtype=torch.float32
    )
    cond = FoundationConditioner(model)
    result = cond.condition(
        df,
        queries,
        local_radius_km=req.local_radius_km,
        max_local_points=req.max_local_points,
    )
    return ConditionResponse(
        mean=result.mean.tolist(),
        std=result.std.tolist(),
        q05=result.q05.tolist(),
        q95=result.q95.tolist(),
    )


@router.get("/v1/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    model = _get_model()
    inducing_shape = model.gp.variational_strategy.inducing_points.shape
    return ModelInfoResponse(
        model_id=_state.model_id,
        version=str(FoundationModel.ARTIFACT_VERSION),
        training_date=_state.training_date,
        regimes_covered=[r.name.lower() for r in Regime],
        encoder_dim=int(model.encoder.spec.n_output),
        n_inducing=int(inducing_shape[0]),
    )


@router.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": "true" if _state.model is not None else "false"}


# --------------------------------------------------------------------------- #
# GeoJSON helpers
# --------------------------------------------------------------------------- #
def _polygon_coords(geojson: dict[str, Any]) -> list[tuple[float, float]]:
    """Extract a flat list of (lon, lat) coords from a GeoJSON Polygon or Feature."""
    if not isinstance(geojson, dict):
        raise ValueError("geojson must be a dict")
    gtype = geojson.get("type")
    if gtype == "Feature":
        return _polygon_coords(geojson.get("geometry") or {})
    if gtype not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"unsupported geometry type {gtype!r}; expected Polygon or MultiPolygon")
    coords_field = geojson.get("coordinates") or []
    if gtype == "Polygon":
        rings = coords_field
    else:
        rings = [r for poly in coords_field for r in poly]
    flat: list[tuple[float, float]] = []
    for ring in rings:
        for pt in ring:
            if len(pt) < 2:
                raise ValueError(f"invalid polygon vertex {pt!r}")
            flat.append((float(pt[0]), float(pt[1])))
    if not flat:
        raise ValueError("polygon has no coordinates")
    return flat


__all__ = [
    "router",
    "set_model",
    "PredictPointRequest",
    "PredictPointResponse",
    "PredictBatchRequest",
    "PredictBatchResponse",
    "PredictPolygonRequest",
    "PredictPolygonResponse",
    "BoringObservation",
    "QueryPoint",
    "ConditionRequest",
    "ConditionResponse",
    "ModelInfoResponse",
]
