"""Spatial K-fold split using Japanese standard mesh groups.

Standard K-fold cross-validation leaks spatial information: rows from
neighboring boring sites end up in both train and test folds, inflating
the apparent score. We instead group rows by their primary/secondary mesh
code (an exact partition of Japan) and stratify the folds *across mesh
codes*, guaranteeing zero spatial leakage at the chosen resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.geo.tiles import primary_mesh_code, secondary_mesh_code, quarter_mesh_code


def _mesh_code_for_row(lat: float, lon: float, mesh_level: int) -> str:
    if mesh_level == 1:
        return primary_mesh_code(lat, lon)
    if mesh_level == 2:
        return secondary_mesh_code(lat, lon)
    if mesh_level == 4:
        return quarter_mesh_code(lat, lon)
    raise ValueError(f"Unsupported mesh_level: {mesh_level} (use 1, 2, or 4)")


def spatial_kfold_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    mesh_level: int = 1,
    seed: int = 42,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition rows into ``n_folds`` (train, test) splits by mesh code.

    Args:
        df: a DataFrame whose rows are training samples.
        n_folds: number of folds.
        mesh_level: ``1`` for primary mesh (~80 km), ``2`` for secondary
            mesh (~10 km), ``4`` for quarter mesh (~5 km).
        seed: shuffle seed for reproducibility.
        lat_column / lon_column: column names holding the EPSG:4326
            coordinates.

    Returns:
        A list of length ``n_folds``, each entry a tuple
        ``(train_idx, test_idx)`` of integer index arrays into ``df``.

    Raises:
        ValueError: if any column is missing or ``n_folds`` is not in
            ``[2, n_unique_mesh_codes]``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if lat_column not in df.columns or lon_column not in df.columns:
        raise ValueError(
            f"DataFrame must contain {lat_column!r} and {lon_column!r} columns."
        )

    codes = np.array(
        [
            _mesh_code_for_row(lat, lon, mesh_level)
            for lat, lon in zip(
                df[lat_column].to_numpy(), df[lon_column].to_numpy(), strict=True
            )
        ]
    )
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    if unique_codes.size < n_folds:
        raise ValueError(
            f"Only {unique_codes.size} unique mesh codes at level {mesh_level}; "
            f"cannot make {n_folds} folds. Use a finer mesh_level or fewer folds."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_codes.size)

    # Greedy load-balanced bucketing: place each mesh code in the fold with
    # the smallest current row count, preserving roughly equal-sized folds.
    code_to_size = np.bincount(inverse)
    fold_of_code = np.empty(unique_codes.size, dtype=np.int64)
    fold_row_counts = np.zeros(n_folds, dtype=np.int64)
    for code_idx in order:
        target_fold = int(np.argmin(fold_row_counts))
        fold_of_code[code_idx] = target_fold
        fold_row_counts[target_fold] += int(code_to_size[code_idx])

    row_fold = fold_of_code[inverse]
    all_idx = np.arange(len(df), dtype=np.int64)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_idx = all_idx[row_fold == k]
        train_idx = all_idx[row_fold != k]
        if test_idx.size == 0:
            raise RuntimeError(
                f"Fold {k} is empty -- this should be impossible after the "
                f"unique-code check. Likely a bug."
            )
        folds.append((train_idx, test_idx))
    return folds


def spatial_kfold_split_contiguous(
    df: pd.DataFrame,
    n_folds: int = 3,
    mesh_level: int = 2,
    seed: int = 42,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Geographic-block spatial K-fold split.

    Unlike :func:`spatial_kfold_split` (load-balanced random shuffle of
    mesh codes), this clusters mesh centroids with k-means so each fold
    occupies a contiguous geographic region of the study area. The
    consequence for buffered CV is dramatic: a 1-mesh ring around a
    contiguous test region excludes only its perimeter (typically
    <15% of training rows) rather than 95% that random shuffle
    produces. This is the stricter, reviewer-defensible notion of
    spatial K-fold and complements the random-mesh metric.

    Args:
        df, n_folds, mesh_level, seed, lat_column, lon_column: same
            semantics as :func:`spatial_kfold_split`.

    Returns:
        ``[(train_idx, test_idx), ...]`` of length ``n_folds``.
    """
    from sklearn.cluster import KMeans

    from shared.geo.tiles import mesh_bounds

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if lat_column not in df.columns or lon_column not in df.columns:
        raise ValueError(
            f"DataFrame must contain {lat_column!r} and {lon_column!r}."
        )

    codes = np.array(
        [
            _mesh_code_for_row(lat, lon, mesh_level)
            for lat, lon in zip(
                df[lat_column].to_numpy(), df[lon_column].to_numpy(), strict=True
            )
        ]
    )
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    if unique_codes.size < n_folds:
        raise ValueError(
            f"Only {unique_codes.size} unique mesh codes at level "
            f"{mesh_level}; cannot make {n_folds} contiguous folds."
        )

    # Centroid of each unique mesh
    centroids = np.array(
        [
            (
                0.5 * (lat_min + lat_max),
                0.5 * (lon_min + lon_max),
            )
            for code in unique_codes
            for lat_min, lon_min, lat_max, lon_max in [mesh_bounds(code)]
        ]
    )

    # K-means cluster centroids into contiguous fold regions
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    fold_of_code = km.fit_predict(centroids)
    row_fold = fold_of_code[inverse]
    all_idx = np.arange(len(df), dtype=np.int64)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_idx = all_idx[row_fold == k]
        train_idx = all_idx[row_fold != k]
        if test_idx.size == 0:
            raise RuntimeError(
                f"Contiguous fold {k} is empty -- k-means produced an "
                f"empty cluster, which should not happen with random_state "
                f"seeded."
            )
        folds.append((train_idx, test_idx))
    return folds


def spatial_kfold_split_buffered(
    df: pd.DataFrame,
    n_folds: int = 3,
    mesh_level: int = 2,
    buffer_meshes: int = 1,
    seed: int = 42,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
    base_split: str = "random",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Buffered variant of :func:`spatial_kfold_split`.

    Computes the same base partition as :func:`spatial_kfold_split` and
    then, for each test fold, drops from the training set every row
    whose mesh code lies within ``buffer_meshes`` cells of any test
    mesh. This addresses the reviewer concern that near-boundary
    boreholes still neighbour training data through the encoder's
    Fourier features, by enforcing a physical separation of
    ``~10 * buffer_meshes`` km between train and test.

    Only ``mesh_level=2`` (secondary mesh) is supported because the
    adjacency helper assumes 6-digit codes.

    Returns ``[(train_idx, test_idx), ...]`` like the base function;
    test indices are unchanged from the base partition (test set
    cannot grow); train indices are a strict subset of the base
    partition's train indices.
    """
    if mesh_level != 2:
        raise ValueError(
            f"Buffered fold only supports mesh_level=2 (secondary); got {mesh_level}"
        )
    from shared.geo.tiles import adjacent_secondary_mesh_codes

    if base_split == "random":
        base_folds = spatial_kfold_split(
            df, n_folds=n_folds, mesh_level=mesh_level, seed=seed,
            lat_column=lat_column, lon_column=lon_column,
        )
    elif base_split == "contiguous":
        base_folds = spatial_kfold_split_contiguous(
            df, n_folds=n_folds, mesh_level=mesh_level, seed=seed,
            lat_column=lat_column, lon_column=lon_column,
        )
    else:
        raise ValueError(
            f"base_split must be 'random' or 'contiguous'; got {base_split!r}"
        )

    codes = np.array(
        [
            _mesh_code_for_row(lat, lon, mesh_level)
            for lat, lon in zip(
                df[lat_column].to_numpy(), df[lon_column].to_numpy(), strict=True
            )
        ]
    )

    buffered: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in base_folds:
        test_codes = set(codes[test_idx].tolist())
        exclude: set[str] = set(test_codes)
        for c in test_codes:
            exclude.update(adjacent_secondary_mesh_codes(c, ring=buffer_meshes))
        # Train rows: in base train AND mesh code NOT in exclude set
        keep_mask = np.array(
            [code not in exclude for code in codes[train_idx]], dtype=bool
        )
        new_train = train_idx[keep_mask]
        if new_train.size == 0:
            raise RuntimeError(
                "Buffered exclusion emptied the training set for a fold; "
                "consider reducing buffer_meshes or n_folds."
            )
        buffered.append((new_train, test_idx))
    return buffered


__all__ = [
    "spatial_kfold_split",
    "spatial_kfold_split_contiguous",
    "spatial_kfold_split_buffered",
]
