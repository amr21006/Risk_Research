# Auto-MCDM Summary

- Validation rows used for weight learning and configuration selection: 40,001
- Test rows used for reporting: 40,000
- Criteria evaluated: 9
- MCDM configurations evaluated: 80
- Selected method: topsis
- Selected normalization: minmax
- Selected weighting: validation_ap
- Selected top-k rate: 20%
- Test PR-AUC for y_cri_high: 0.8275
- Test selected F1 for y_cri_high: 0.5662

Generated tables are in `outputs/auto_mcdm/tables`.
Generated figures are in `outputs/auto_mcdm/figures`.