"""Structured Hydra schemas for national-scale training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class BoundsConfig:
    lat_min: float = 24.0
    lat_max: float = 46.0
    lon_min: float = 122.0
    lon_max: float = 146.5


@dataclass
class MeshConfig:
    name: str = "primary"
    fraction_numerator: int = 1
    fraction_denominator: int = 1
    tile_size_km: float = 80.0
    halo_km: float = 5.0


@dataclass
class RegionConfig:
    name: str = "japan"
    crs: str = "EPSG:4326"
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    river_name: Optional[str] = None
    max_river_distance_km: Optional[float] = None


@dataclass
class CovariateSpec:
    name: str = ""
    source: str = ""
    local_path: Optional[str] = None
    dtype: str = "float32"
    normalize: str = "none"
    category: Optional[str] = None


@dataclass
class CovariateGroup:
    name: str = "core"
    feature_cube: str = "${io.features_root}/national_covariates.zarr"
    features: dict[str, CovariateSpec] = field(default_factory=dict)


@dataclass
class EncoderConfig:
    type: str = "resmlp"
    n_layers: int = 6
    hidden: int = 256
    output: int = 32
    batchnorm: bool = True
    dropout: float = 0.0


@dataclass
class SVGPConfig:
    n_inducing: int = 50000
    learn_inducing: bool = True
    whitened: bool = True
    inducing_init: str = "kmeans_pp_stratified"


@dataclass
class KernelConfig:
    name: str = "dkl_depth_additive"
    acf_type: Optional[str] = None
    inputs: list[str] = field(default_factory=list)
    structure: Optional[str] = None
    primary: dict[str, Any] = field(default_factory=dict)
    additive_components: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    type: str = "dkl_svgp"
    encoder: Optional[EncoderConfig] = field(default_factory=EncoderConfig)
    svgp: SVGPConfig = field(default_factory=SVGPConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    outputs: list[str] = field(default_factory=lambda: ["n_value"])
    lmc: Optional[dict[str, Any]] = None


@dataclass
class TrainingConfig:
    device: str = "cuda"
    world_size: int = 1
    node_count: int = 1
    chips_per_node: int = 1
    distributed: bool = False
    nccl_backend: bool = False
    regime_balanced_sampler: bool = False
    n_epochs: int = 1000
    batch_size: int = 8192
    lr: float = 0.005
    lr_schedule: str = "cosine_warmup"
    warmup_steps: int = 500
    grad_accum_steps: int = 1
    mixed_precision: str = "bf16"
    checkpoint_every_min: int = 30
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-5
    early_stopping_patience: int = 50


@dataclass
class PredictionConfig:
    depth_layers: list[float] = field(default_factory=list)
    grid_resolution_m: int = 50
    output_format: str = "zarr"
    compress: str = "zstd"
    compress_level: Optional[int] = 5
    chunks: dict[str, int] = field(
        default_factory=lambda: {"lon": 1024, "lat": 1024, "depth": 1}
    )
    batch_size_cells: int = 200000


@dataclass
class WandbConfig:
    mode: str = "online"
    project: str = "geo-estimation-national"


@dataclass
class IOConfig:
    data_root: str = "${oc.env:GEO_ESTIMATION_DATA,${hydra:runtime.cwd}/data}"
    raw_root: str = "${io.data_root}/raw"
    processed_root: str = "${io.data_root}/processed"
    features_root: str = "${io.data_root}/features"
    predictions_root: str = "${io.data_root}/predictions"
    run_root: str = "${io.data_root}/runs"
    checkpoint_root: str = "${io.run_root}/checkpoints"
    scratch_root: Optional[str] = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class RunConfig:
    name: str = "national_${region.name}_${model.type}_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    output_dir: str = "${io.run_root}/${run.name}"


@dataclass
class NationalConfig:
    run: RunConfig = field(default_factory=RunConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    covariates: CovariateGroup = field(default_factory=CovariateGroup)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    io: IOConfig = field(default_factory=IOConfig)


def register_configs() -> None:
    """Register schemas without shadowing concrete YAML config names."""
    cs = ConfigStore.instance()
    cs.store(name="national_schema", node=NationalConfig)
    cs.store(group="schema", name="national", node=NationalConfig)
    cs.store(group="schema/region", name="base", node=RegionConfig)
    cs.store(group="schema/covariates", name="base", node=CovariateGroup)
    cs.store(group="schema/model", name="base", node=ModelConfig)
    cs.store(group="schema/training", name="base", node=TrainingConfig)
    cs.store(group="schema/prediction", name="base", node=PredictionConfig)
    cs.store(group="schema/io", name="base", node=IOConfig)


register_configs()

