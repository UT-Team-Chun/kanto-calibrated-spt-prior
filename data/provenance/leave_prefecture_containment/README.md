# Leave-prefecture-out (administrative-polygon containment)

Per-prefecture and macro-mean RMSE/MAE for the recommended deployment regressors
(GPBoost, CatBoost) under leave-prefecture-out over all seven Kanto prefectures.
Each borehole is assigned to the administrative prefecture whose boundary polygon
contains it (point-in-polygon; `national/evaluation/prefecture_regions.py` +
`national/evaluation/assets/kanto_prefecture_polygon_assignment.parquet`). The
seven held-out folds are SPT measurement rows summing to 435,732 of the
495,725-row corpus. GPBoost uses the canonical config (num_neighbors=20).

Regenerate with:
    python -m scripts.run_leave_region_out --partition prefecture --model gpboost
    python -m scripts.run_leave_region_out --partition prefecture --model catboost
