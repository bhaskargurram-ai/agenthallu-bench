# AgentHallu-Bench Results Tables (Day 8)

## Table 1: Main Results (Baseline vs P2)

| model           |   n_baseline |   n_p2 | baseline_correct_ci   | p2_correct_ci     | p2_eps_ci            |
|:----------------|-------------:|-------:|:----------------------|:------------------|:---------------------|
| deepseek_v3     |          200 |    800 | 66.0 [59.5, 73.0]     | 69.8 [66.4, 73.2] | 0.932 [0.886, 0.984] |
| gemini_20_flash |          200 |    800 | 61.0 [54.0, 68.0]     | 61.5 [58.1, 64.6] | 0.080 [0.056, 0.105] |
| gemini_25_flash |            0 |    200 | 0.0 [0.0, 0.0]        | 63.0 [56.5, 69.5] | 0.825 [0.730, 0.920] |
| gpt35_turbo     |          200 |    800 | 65.5 [59.0, 72.0]     | 67.1 [64.0, 70.6] | 0.669 [0.618, 0.720] |
| gpt41_mini      |          200 |    800 | 69.0 [62.5, 75.0]     | 68.2 [64.9, 71.6] | 0.895 [0.844, 0.945] |
| gpt41_nano      |          200 |    800 | 56.0 [49.0, 62.5]     | 57.0 [53.4, 60.2] | 1.188 [1.142, 1.239] |
| gpt4o           |            0 |    200 | 0.0 [0.0, 0.0]        | 57.5 [51.0, 64.0] | 1.220 [1.125, 1.310] |
| gpt4o_mini      |          200 |    800 | 51.0 [44.0, 57.5]     | 51.0 [47.8, 54.4] | 0.606 [0.551, 0.660] |
| o3_mini         |           41 |    200 | 48.8 [34.1, 63.4]     | 50.0 [43.0, 57.0] | 1.150 [1.040, 1.245] |

## Table 2: Interceptor Performance

| model           | condition         |   n | halluc_ci         | abstain_ci        | correct_ci        | fp_rate   |
|:----------------|:------------------|----:|:------------------|:------------------|:------------------|:----------|
| gemini_20_flash | early_interceptor | 100 | 53.0 [44.0, 63.0] | 1.0 [0.0, 3.0]    | 46.0 [36.0, 55.0] | 0.0%      |
| gemini_20_flash | output_only       | 100 | 51.0 [41.0, 61.0] | 1.0 [0.0, 3.0]    | 48.0 [38.0, 58.0] | 0.0%      |
| gpt4o_mini      | early_interceptor | 100 | 53.0 [43.0, 63.0] | 20.0 [12.0, 28.0] | 27.0 [19.0, 36.0] | 0.0%      |
| gpt4o_mini      | output_only       | 100 | 50.0 [40.0, 61.0] | 18.0 [11.0, 26.0] | 32.0 [22.0, 41.0] | 0.0%      |

## Table 3: Multi-Seed Stability

| model           | condition   |   seeds |   halluc_mean |   halluc_std |   eps_mean |   eps_std |
|:----------------|:------------|--------:|--------------:|-------------:|-----------:|----------:|
| gemini_20_flash | baseline    |       3 |          51.3 |         1.15 |      0     |     0     |
| gemini_20_flash | p2_semantic |       3 |          50.7 |         1.15 |      0.053 |     0.023 |
| gpt4o_mini      | baseline    |       3 |          55.3 |         1.15 |      0     |     0     |
| gpt4o_mini      | p2_semantic |       3 |          57.3 |         5.03 |      0.833 |     0.081 |

## Table 4: Interceptor Ablation

| model           | condition   |   n | halluc            | abstain           | correct           |
|:----------------|:------------|----:|:------------------|:------------------|:------------------|
| gemini_20_flash | l1_l2       |  60 | 51.7 [40.0, 65.0] | 0.0 [0.0, 0.0]    | 48.3 [35.0, 60.0] |
| gemini_20_flash | l1_l2_l3    |  60 | 48.3 [35.0, 60.0] | 1.7 [0.0, 5.0]    | 50.0 [38.3, 63.3] |
| gemini_20_flash | l1_only     |  60 | 48.3 [36.7, 60.0] | 0.0 [0.0, 0.0]    | 51.7 [40.0, 63.3] |
| gpt4o_mini      | l1_l2       |  60 | 46.7 [33.3, 60.0] | 11.7 [5.0, 20.0]  | 41.7 [30.0, 53.3] |
| gpt4o_mini      | l1_l2_l3    |  60 | 41.7 [30.0, 55.0] | 26.7 [15.0, 38.3] | 31.7 [20.0, 43.3] |
| gpt4o_mini      | l1_only     |  60 | 51.7 [40.0, 65.0] | 18.3 [8.3, 28.3]  | 30.0 [18.3, 41.7] |

## Table 5: Self-Consistency Voting (M2)

| model       | condition   |   n | halluc            | abstain           | correct           |   avg_unique_ans |
|:------------|:------------|----:|:------------------|:------------------|:------------------|-----------------:|
| deepseek_v3 | baseline    |  30 | 30.0 [13.3, 46.7] | 46.7 [30.0, 66.7] | 23.3 [10.0, 36.7] |             2.4  |
| deepseek_v3 | p2_semantic |  30 | 23.3 [10.0, 40.0] | 60.0 [43.3, 76.7] | 16.7 [6.7, 30.0]  |             2.53 |
| gpt4o_mini  | baseline    |  30 | 20.0 [6.7, 33.3]  | 70.0 [53.3, 86.7] | 10.0 [0.0, 20.0]  |             2.6  |
| gpt4o_mini  | p2_semantic |  30 | 23.3 [10.0, 40.0] | 56.7 [40.0, 73.3] | 20.0 [6.7, 33.3]  |             2.5  |

## Table 6: Judge-Agreement Ensemble (M1)

| rater           |   kappa |
|:----------------|--------:|
| heuristic       |   0.143 |
| gpt4o           |   0.792 |
| gemini_25_flash |   0.821 |
| gpt4o_mini      |   0.77  |
| _unanimous rate |   0.729 |
| _n              | 960     |

## Table 7: Cross-Benchmark Validation (A2)

| model           |   n_original | halluc_original   |   n_heldout | halluc_heldout    |
|:----------------|-------------:|:------------------|------------:|:------------------|
| deepseek_v3     |          800 | 30.2 [26.8, 33.6] |         150 | 25.3 [18.7, 32.7] |
| gemini_20_flash |          800 | 38.5 [35.4, 41.9] |           0 | —                 |
| gemini_25_flash |          200 | 37.0 [30.5, 43.5] |           0 | —                 |
| gpt35_turbo     |          800 | 32.9 [29.4, 36.0] |           0 | —                 |
| gpt41_mini      |          800 | 31.8 [28.4, 35.1] |           0 | —                 |
| gpt41_nano      |          800 | 43.0 [39.8, 46.6] |           0 | —                 |
| gpt4o           |          200 | 42.5 [36.0, 49.0] |           0 | —                 |
| gpt4o_mini      |          800 | 49.0 [45.6, 52.2] |         150 | 60.0 [52.0, 67.3] |
| o3_mini         |          200 | 50.0 [43.0, 57.0] |           0 | —                 |

## Table 8: Cross-Domain Replication — Retail (Fix 4)

| model           |   n | baseline_correct   | p2_correct        | p2_eps               |
|:----------------|----:|:-------------------|:------------------|:---------------------|
| deepseek_v3     | 100 | 37.0 [27.0, 47.0]  | 43.0 [33.0, 53.0] | 0.850 [0.690, 1.010] |
| gemini_20_flash | 100 | 17.0 [10.0, 25.0]  | 17.0 [10.0, 25.0] | 0.180 [0.080, 0.300] |
| gpt4o_mini      | 100 | 29.0 [20.0, 38.0]  | 33.0 [24.0, 42.0] | 1.050 [0.900, 1.220] |

## Table 9: Tuned Interceptor — Threshold Sweep (Fix 1)

| metric         | value                                                     |
|:---------------|:----------------------------------------------------------|
| best_cfg       | {"l1_threshold": 1, "l2_keywords": 2, "l3_enabled": true} |
| best_F1        | 0.842                                                     |
| best_precision | 0.923                                                     |
| best_recall    | 0.774                                                     |

## Table 10: Human-Annotator Agreement (Fix 3)

_(no data)_


## Table A4: Achieved Power per Claim

| claim                                          |   n_a |   n_b |   p_a |   p_b |    h |   power | underpowered   |
|:-----------------------------------------------|------:|------:|------:|------:|-----:|--------:|:---------------|
| P2 increases error for deepseek_v3             |   450 |   800 | 0.482 | 0.302 | 0.37 |    1    | False          |
| P2 increases error for gemini_20_flash         |   450 |   800 | 0.524 | 0.385 | 0.28 |    1    | False          |
| P2 increases error for gpt35_turbo             |   450 |   800 | 0.496 | 0.329 | 0.34 |    1    | False          |
| P2 increases error for gpt41_mini              |   450 |   800 | 0.453 | 0.318 | 0.28 |    1    | False          |
| P2 increases error for gpt41_nano              |   450 |   800 | 0.547 | 0.43  | 0.23 |    0.98 | False          |
| P2 increases error for gpt4o_mini              |   450 |   800 | 0.6   | 0.49  | 0.22 |    0.96 | False          |
| P2 increases error for o3_mini                 |    50 |   200 | 0.6   | 0.5   | 0.2  |    0.25 | True           |
| Domain calendar differs from others (P2)       |  3684 |  1716 | 0.397 | 0.35  | 0.1  |    0.92 | False          |
| Domain weather differs from others (P2)        |  3444 |  1956 | 0.379 | 0.387 | 0.01 |    0.08 | True           |
| Domain medical differs from others (P2)        |  3672 |  1728 | 0.369 | 0.409 | 0.08 |    0.79 | True           |
| Interceptor reduces halluc for gpt4o_mini      |   100 |   100 | 0.5   | 0.53  | 0.06 |    0.06 | True           |
| Interceptor reduces halluc for gemini_20_flash |   100 |   100 | 0.51  | 0.53  | 0.04 |    0.05 | True           |
