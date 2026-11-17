# Independent ECL one-minute horizon model research

This report compares fixed, modest logistic regression, LightGBM, and XGBoost models for exact-endpoint 10- and 15-minute targets. It uses only generated training/validation datasets through 2026-09-24. The repeatedly used validation period is confirmation, not a pristine holdout. No test data, live API, or paid data were used.

Signal coverage is directional predictions (UP or DOWN) divided by all eligible labeled rows. Eligible-signal accuracy counts a NO_SIGNAL actual as an error if a directional prediction was emitted. Directional accuracy on actual directional rows is also shown separately. Class order is DOWN, NO_SIGNAL, UP.

Thresholds are selected once per horizon/model from pooled chronological training OOF predictions only, maximizing three-class macro-F1 on a fixed grid subject to at least 20% signal coverage and 30 signals. Validation results did not influence model or threshold selection.

## Validation comparison

| horizon_min | model | scope | policy | fold | n | accuracy | balanced_accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall | directional_macro_precision | directional_macro_recall | directional_macro_f1 | signal_coverage_all_eligible | eligible_signal_count | eligible_signal_accuracy | directional_accuracy_on_actual_directional | actual_directional_count | no_signal_prediction_count | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | majority_baseline | validation | raw |  | 543 | 0.4991 | 0.3333 | 0.2219 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 272 | 543 |  |
| 10 | return_10_sign_baseline | validation | raw |  | 543 | 0.2376 | 0.3148 | 0.2100 | 0.2695 | 0.5429 | 0.2031 | 0.4015 | 0.2363 | 0.4722 | 0.3150 | 1.0000 | 543 | 0.2376 | 0.4743 | 272 | 0 |  |
| 10 | logistic_regression | validation | raw_argmax |  | 543 | 0.4236 | 0.4246 | 0.4113 | 0.2941 | 0.3571 | 0.3514 | 0.4924 | 0.3227 | 0.4248 | 0.3663 | 0.6538 | 355 | 0.3239 | 0.4228 | 272 | 188 | 1.1048 |
| 10 | logistic_regression | validation | training_oof_threshold |  | 543 | 0.4678 | 0.3870 | 0.3800 | 0.2750 | 0.1571 | 0.3838 | 0.2879 | 0.3294 | 0.2225 | 0.2645 | 0.3297 | 179 | 0.3352 | 0.2206 | 272 | 364 | 1.1048 |
| 10 | lightgbm | validation | raw_argmax |  | 543 | 0.3702 | 0.3598 | 0.3545 | 0.2434 | 0.3929 | 0.2500 | 0.2879 | 0.2467 | 0.3404 | 0.2841 | 0.6961 | 378 | 0.2460 | 0.3419 | 272 | 165 | 1.1382 |
| 10 | lightgbm | validation | training_oof_threshold |  | 543 | 0.4015 | 0.3671 | 0.3660 | 0.2564 | 0.3571 | 0.2560 | 0.2424 | 0.2562 | 0.2998 | 0.2738 | 0.5893 | 320 | 0.2562 | 0.3015 | 272 | 223 | 1.1382 |
| 10 | xgboost | validation | raw_argmax |  | 543 | 0.3923 | 0.3985 | 0.3837 | 0.2753 | 0.4857 | 0.3285 | 0.3409 | 0.3019 | 0.4133 | 0.3430 | 0.7072 | 384 | 0.2943 | 0.4154 | 272 | 159 | 1.1113 |
| 10 | xgboost | validation | training_oof_threshold |  | 543 | 0.3923 | 0.3961 | 0.3825 | 0.2757 | 0.4786 | 0.3284 | 0.3333 | 0.3020 | 0.4060 | 0.3403 | 0.6943 | 377 | 0.2944 | 0.4081 | 272 | 166 | 1.1113 |
| 15 | majority_baseline | validation | raw |  | 497 | 0.3964 | 0.3333 | 0.1892 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 300 | 497 |  |
| 15 | return_10_sign_baseline | validation | raw |  | 497 | 0.3038 | 0.3373 | 0.2530 | 0.3198 | 0.5563 | 0.2880 | 0.4557 | 0.3039 | 0.5060 | 0.3796 | 1.0000 | 497 | 0.3038 | 0.5033 | 300 | 0 |  |
| 15 | logistic_regression | validation | raw_argmax |  | 497 | 0.3803 | 0.3693 | 0.3716 | 0.2375 | 0.2676 | 0.3688 | 0.3734 | 0.3031 | 0.3205 | 0.3114 | 0.6439 | 320 | 0.3031 | 0.3233 | 300 | 177 | 1.1057 |
| 15 | logistic_regression | validation | training_oof_threshold |  | 497 | 0.3803 | 0.3693 | 0.3716 | 0.2375 | 0.2676 | 0.3688 | 0.3734 | 0.3031 | 0.3205 | 0.3114 | 0.6439 | 320 | 0.3031 | 0.3233 | 300 | 177 | 1.1057 |
| 15 | lightgbm | validation | raw_argmax |  | 497 | 0.3883 | 0.3896 | 0.3881 | 0.2851 | 0.4437 | 0.3741 | 0.3291 | 0.3296 | 0.3864 | 0.3486 | 0.7243 | 360 | 0.3194 | 0.3833 | 300 | 137 | 1.1391 |
| 15 | lightgbm | validation | training_oof_threshold |  | 497 | 0.3823 | 0.3735 | 0.3722 | 0.2737 | 0.3662 | 0.3805 | 0.2722 | 0.3271 | 0.3192 | 0.3153 | 0.6097 | 303 | 0.3135 | 0.3167 | 300 | 194 | 1.1391 |
| 15 | xgboost | validation | raw_argmax |  | 497 | 0.4004 | 0.3978 | 0.3987 | 0.2897 | 0.4366 | 0.3427 | 0.3101 | 0.3162 | 0.3734 | 0.3369 | 0.7183 | 357 | 0.3109 | 0.3700 | 300 | 140 | 1.1179 |
| 15 | xgboost | validation | training_oof_threshold |  | 497 | 0.4024 | 0.3991 | 0.3996 | 0.2925 | 0.4366 | 0.3429 | 0.3038 | 0.3177 | 0.3702 | 0.3362 | 0.7082 | 352 | 0.3125 | 0.3667 | 300 | 145 | 1.1179 |

## Training OOF versus validation raw argmax

| horizon_min | model | scope | n | accuracy_training_oof | balanced_accuracy_training_oof | macro_f1_training_oof | up_precision | up_recall | down_precision | down_recall | directional_macro_precision_training_oof | directional_macro_recall | directional_macro_f1 | signal_coverage_all_eligible_training_oof | eligible_signal_count | eligible_signal_accuracy_training_oof | directional_accuracy_on_actual_directional | actual_directional_count | no_signal_prediction_count | log_loss | accuracy_validation | balanced_accuracy_validation | macro_f1_validation | directional_macro_precision_validation | signal_coverage_all_eligible_validation | eligible_signal_accuracy_validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | lightgbm | training_oof_raw | 1179 | 0.3673 | 0.3700 | 0.3644 | 0.2896 | 0.3902 | 0.3073 | 0.3664 | 0.2984 | 0.3783 | 0.3334 | 0.7116 | 839 | 0.2980 | 0.3782 | 661 | 340 | 1.2185 | 0.3702 | 0.3598 | 0.3545 | 0.2467 | 0.6961 | 0.2460 |
| 10 | logistic_regression | training_oof_raw | 1179 | 0.3588 | 0.3444 | 0.3400 | 0.2418 | 0.2012 | 0.2879 | 0.3994 | 0.2648 | 0.3003 | 0.2771 | 0.6234 | 735 | 0.2707 | 0.3011 | 661 | 444 | 1.1219 | 0.4236 | 0.4246 | 0.4113 | 0.3227 | 0.6538 | 0.3239 |
| 10 | majority_baseline | training_oof_raw | 1179 | 0.4394 | 0.3333 | 0.2035 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 661 | 1179 |  |  |  |  |  |  |  |
| 10 | return_10_sign_baseline | training_oof_raw | 1179 | 0.2604 | 0.3088 | 0.2238 | 0.2526 | 0.4451 | 0.2659 | 0.4775 | 0.2592 | 0.4613 | 0.3319 | 0.9975 | 1176 | 0.2594 | 0.4614 | 661 | 3 |  |  |  |  |  |  |  |
| 10 | xgboost | training_oof_raw | 1179 | 0.3673 | 0.3732 | 0.3659 | 0.2851 | 0.3963 | 0.3123 | 0.3874 | 0.2987 | 0.3919 | 0.3387 | 0.7371 | 869 | 0.2980 | 0.3918 | 661 | 310 | 1.1647 | 0.3923 | 0.3985 | 0.3837 | 0.3019 | 0.7072 | 0.2943 |
| 15 | lightgbm | training_oof_raw | 1070 | 0.3720 | 0.3723 | 0.3715 | 0.2791 | 0.3909 | 0.3822 | 0.3488 | 0.3306 | 0.3699 | 0.3452 | 0.6953 | 744 | 0.3226 | 0.3687 | 651 | 326 | 1.2099 | 0.3883 | 0.3896 | 0.3881 | 0.3296 | 0.7243 | 0.3194 |
| 15 | logistic_regression | training_oof_raw | 1070 | 0.3692 | 0.3572 | 0.3558 | 0.3021 | 0.2834 | 0.3187 | 0.2965 | 0.3104 | 0.2899 | 0.2998 | 0.5682 | 608 | 0.3109 | 0.2903 | 651 | 462 | 1.1279 | 0.3803 | 0.3693 | 0.3716 | 0.3031 | 0.6439 | 0.3031 |
| 15 | majority_baseline | training_oof_raw | 1070 | 0.3056 | 0.3217 | 0.2379 | 0.2746 | 0.6906 | 0.0000 | 0.0000 | 0.1373 | 0.3453 | 0.1965 | 0.7215 | 772 | 0.2746 | 0.3257 | 651 | 298 |  |  |  |  |  |  |  |
| 15 | return_10_sign_baseline | training_oof_raw | 1070 | 0.2710 | 0.2948 | 0.2252 | 0.2425 | 0.3941 | 0.2935 | 0.4855 | 0.2680 | 0.4398 | 0.3330 | 0.9981 | 1068 | 0.2697 | 0.4424 | 651 | 2 |  |  |  |  |  |  |  |
| 15 | xgboost | training_oof_raw | 1070 | 0.3654 | 0.3641 | 0.3635 | 0.2786 | 0.3811 | 0.3514 | 0.3198 | 0.3150 | 0.3504 | 0.3284 | 0.6850 | 733 | 0.3097 | 0.3487 | 651 | 337 | 1.1541 | 0.4004 | 0.3978 | 0.3987 | 0.3162 | 0.7183 | 0.3109 |

Training values are pooled expanding-window OOF predictions only. Each OOF row was predicted by a model fitted on earlier training dates. Validation is the later, already-reused period.

## Paired validation timestamps

Both horizons have 422 common eligible validation timestamps.

| horizon_min | model | policy | n_common | n | accuracy | balanced_accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall | directional_macro_precision | directional_macro_recall | directional_macro_f1 | signal_coverage_all_eligible | eligible_signal_count | eligible_signal_accuracy | directional_accuracy_on_actual_directional | actual_directional_count | no_signal_prediction_count | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | majority_baseline | raw | 422 | 422 | 0.5024 | 0.3333 | 0.2229 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 210 | 422 |  |
| 10 | return_10_sign_baseline | raw | 422 | 422 | 0.2464 | 0.3286 | 0.2183 | 0.2844 | 0.5741 | 0.2059 | 0.4118 | 0.2451 | 0.4929 | 0.3274 | 1.0000 | 422 | 0.2464 | 0.4952 | 210 | 0 |  |
| 10 | logistic_regression | raw_argmax | 422 | 422 | 0.4218 | 0.4175 | 0.4059 | 0.2812 | 0.3333 | 0.3451 | 0.4804 | 0.3132 | 0.4069 | 0.3534 | 0.6398 | 270 | 0.3148 | 0.4048 | 210 | 152 | 1.0996 |
| 10 | logistic_regression | training_oof_threshold | 422 | 422 | 0.4692 | 0.3913 | 0.3848 | 0.2615 | 0.1574 | 0.3765 | 0.3137 | 0.3190 | 0.2356 | 0.2694 | 0.3555 | 150 | 0.3267 | 0.2333 | 210 | 272 | 1.0996 |
| 10 | lightgbm | raw_argmax | 422 | 422 | 0.3910 | 0.3793 | 0.3745 | 0.2469 | 0.3704 | 0.2800 | 0.3431 | 0.2635 | 0.3568 | 0.3023 | 0.6801 | 287 | 0.2613 | 0.3571 | 210 | 135 | 1.1269 |
| 10 | lightgbm | training_oof_threshold | 422 | 422 | 0.4147 | 0.3773 | 0.3777 | 0.2500 | 0.3241 | 0.2788 | 0.2843 | 0.2644 | 0.3042 | 0.2819 | 0.5782 | 244 | 0.2623 | 0.3048 | 210 | 178 | 1.1269 |
| 10 | xgboost | raw_argmax | 422 | 422 | 0.3957 | 0.4014 | 0.3879 | 0.2682 | 0.4444 | 0.3305 | 0.3824 | 0.2993 | 0.4134 | 0.3445 | 0.7038 | 297 | 0.2929 | 0.4143 | 210 | 125 | 1.1004 |
| 10 | xgboost | training_oof_threshold | 422 | 422 | 0.3981 | 0.4013 | 0.3888 | 0.2727 | 0.4444 | 0.3304 | 0.3725 | 0.3016 | 0.4085 | 0.3441 | 0.6896 | 291 | 0.2955 | 0.4095 | 210 | 131 | 1.1004 |
| 15 | majority_baseline | raw | 422 | 422 | 0.3910 | 0.3333 | 0.1874 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 257 | 422 |  |
| 15 | return_10_sign_baseline | raw | 422 | 422 | 0.3081 | 0.3389 | 0.2551 | 0.3303 | 0.5806 | 0.2843 | 0.4361 | 0.3073 | 0.5084 | 0.3826 | 1.0000 | 422 | 0.3081 | 0.5058 | 257 | 0 |  |
| 15 | logistic_regression | raw_argmax | 422 | 422 | 0.3768 | 0.3692 | 0.3709 | 0.2500 | 0.2742 | 0.3611 | 0.3910 | 0.3056 | 0.3326 | 0.3185 | 0.6635 | 280 | 0.3071 | 0.3346 | 257 | 142 | 1.1106 |
| 15 | logistic_regression | training_oof_threshold | 422 | 422 | 0.3768 | 0.3692 | 0.3709 | 0.2500 | 0.2742 | 0.3611 | 0.3910 | 0.3056 | 0.3326 | 0.3185 | 0.6635 | 280 | 0.3071 | 0.3346 | 257 | 142 | 1.1106 |
| 15 | lightgbm | raw_argmax | 422 | 422 | 0.3910 | 0.3926 | 0.3919 | 0.2947 | 0.4516 | 0.3629 | 0.3383 | 0.3288 | 0.3950 | 0.3534 | 0.7441 | 314 | 0.3217 | 0.3930 | 257 | 108 | 1.1337 |
| 15 | lightgbm | training_oof_threshold | 422 | 422 | 0.3863 | 0.3785 | 0.3782 | 0.2822 | 0.3710 | 0.3762 | 0.2857 | 0.3292 | 0.3283 | 0.3227 | 0.6256 | 264 | 0.3182 | 0.3268 | 257 | 158 | 1.1337 |
| 15 | xgboost | raw_argmax | 422 | 422 | 0.4052 | 0.4039 | 0.4050 | 0.3065 | 0.4597 | 0.3360 | 0.3158 | 0.3212 | 0.3877 | 0.3467 | 0.7370 | 311 | 0.3183 | 0.3852 | 257 | 111 | 1.1176 |
| 15 | xgboost | training_oof_threshold | 422 | 422 | 0.4076 | 0.4055 | 0.4064 | 0.3081 | 0.4597 | 0.3361 | 0.3083 | 0.3221 | 0.3840 | 0.3453 | 0.7275 | 307 | 0.3192 | 0.3813 | 257 | 115 | 1.1176 |

## Missing-data exposure

| horizon_min | dataset_rows | feature_ready_rows | warmup_or_missing_feature_rows | unavailable_target_among_feature_ready | missing_endpoint | session_or_boundary |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 49347 | 4085 | 45262 | 1585 | 459 | 1126 |
| 15 | 49347 | 4085 | 45262 | 1834 | 416 | 1418 |

## Training OOF folds

| horizon_min | model | fold | fold_start_date | n | accuracy | balanced_accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall | directional_macro_precision | directional_macro_recall | directional_macro_f1 | signal_coverage_all_eligible | eligible_signal_count | eligible_signal_accuracy | directional_accuracy_on_actual_directional | actual_directional_count | no_signal_prediction_count | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | majority_baseline | 1 | 2026-04-20 | 398 | 0.4874 | 0.3333 | 0.2185 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 204 | 398 |  |
| 10 | return_10_sign_baseline | 1 | 2026-04-20 | 398 | 0.2714 | 0.3487 | 0.2416 | 0.2513 | 0.4896 | 0.2823 | 0.5463 | 0.2668 | 0.5179 | 0.3522 | 0.9950 | 396 | 0.2677 | 0.5196 | 204 | 2 |  |
| 10 | majority_baseline | 2 | 2026-05-22 | 544 | 0.4228 | 0.3333 | 0.1981 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 314 | 544 |  |
| 10 | return_10_sign_baseline | 2 | 2026-05-22 | 544 | 0.2426 | 0.2796 | 0.2046 | 0.2358 | 0.3791 | 0.2492 | 0.4596 | 0.2425 | 0.4194 | 0.3069 | 0.9982 | 543 | 0.2431 | 0.4204 | 314 | 1 |  |
| 10 | majority_baseline | 3 | 2026-06-30 | 237 | 0.3966 | 0.3333 | 0.1893 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 143 | 237 |  |
| 10 | return_10_sign_baseline | 3 | 2026-06-30 | 237 | 0.2827 | 0.3084 | 0.2331 | 0.2828 | 0.5190 | 0.2826 | 0.4062 | 0.2827 | 0.4626 | 0.3497 | 1.0000 | 237 | 0.2827 | 0.4685 | 143 | 0 |  |
| 10 | logistic_regression | 1 | 2026-04-20 | 398 | 0.3392 | 0.3585 | 0.3341 | 0.2586 | 0.3125 | 0.2967 | 0.5000 | 0.2777 | 0.4062 | 0.3277 | 0.7487 | 298 | 0.2819 | 0.4118 | 204 | 100 | 1.1166 |
| 10 | logistic_regression | 2 | 2026-05-22 | 544 | 0.3732 | 0.3472 | 0.3423 | 0.2137 | 0.1634 | 0.2947 | 0.3478 | 0.2542 | 0.2556 | 0.2521 | 0.5643 | 307 | 0.2638 | 0.2580 | 314 | 237 | 1.1239 |
| 10 | logistic_regression | 3 | 2026-06-30 | 237 | 0.3586 | 0.3471 | 0.3303 | 0.2750 | 0.1392 | 0.2556 | 0.3594 | 0.2653 | 0.2493 | 0.2418 | 0.5485 | 130 | 0.2615 | 0.2378 | 143 | 107 | 1.1264 |
| 10 | lightgbm | 1 | 2026-04-20 | 398 | 0.3166 | 0.3846 | 0.2876 | 0.2774 | 0.8438 | 0.3455 | 0.1759 | 0.3114 | 0.5098 | 0.3253 | 0.8719 | 347 | 0.2882 | 0.4902 | 204 | 51 | 1.4293 |
| 10 | lightgbm | 2 | 2026-05-22 | 544 | 0.3787 | 0.3538 | 0.3464 | 0.2323 | 0.1503 | 0.2844 | 0.3851 | 0.2584 | 0.2677 | 0.2549 | 0.5827 | 317 | 0.2681 | 0.2707 | 314 | 227 | 1.1191 |
| 10 | lightgbm | 3 | 2026-06-30 | 237 | 0.4262 | 0.4425 | 0.4223 | 0.4706 | 0.3038 | 0.3306 | 0.6406 | 0.4006 | 0.4722 | 0.4027 | 0.7384 | 175 | 0.3714 | 0.4545 | 143 | 62 | 1.0927 |
| 10 | xgboost | 1 | 2026-04-20 | 398 | 0.2864 | 0.3567 | 0.2498 | 0.2667 | 0.8333 | 0.2830 | 0.1389 | 0.2748 | 0.4861 | 0.2952 | 0.8869 | 353 | 0.2691 | 0.4657 | 204 | 45 | 1.3152 |
| 10 | xgboost | 2 | 2026-05-22 | 544 | 0.3934 | 0.3779 | 0.3709 | 0.2788 | 0.1895 | 0.3061 | 0.4658 | 0.2925 | 0.3277 | 0.2976 | 0.6415 | 349 | 0.2980 | 0.3312 | 314 | 195 | 1.0967 |
| 10 | xgboost | 3 | 2026-06-30 | 237 | 0.4430 | 0.4513 | 0.4350 | 0.4038 | 0.2658 | 0.3391 | 0.6094 | 0.3715 | 0.4376 | 0.3782 | 0.7046 | 167 | 0.3593 | 0.4196 | 143 | 70 | 1.0680 |
| 15 | majority_baseline | 1 | 2026-04-15 | 382 | 0.2932 | 0.3333 | 0.1511 | 0.2932 | 1.0000 | 0.0000 | 0.0000 | 0.1466 | 0.5000 | 0.2267 | 1.0000 | 382 | 0.2932 | 0.4571 | 245 | 0 |  |
| 15 | return_10_sign_baseline | 1 | 2026-04-15 | 382 | 0.3168 | 0.3247 | 0.2602 | 0.2822 | 0.4107 | 0.3364 | 0.5489 | 0.3093 | 0.4798 | 0.3758 | 0.9948 | 380 | 0.3132 | 0.4857 | 245 | 2 |  |
| 15 | majority_baseline | 2 | 2026-05-20 | 390 | 0.2564 | 0.3333 | 0.1361 | 0.2564 | 1.0000 | 0.0000 | 0.0000 | 0.1282 | 0.5000 | 0.2041 | 1.0000 | 390 | 0.2564 | 0.4484 | 223 | 0 |  |
| 15 | return_10_sign_baseline | 2 | 2026-05-20 | 390 | 0.2564 | 0.2953 | 0.2150 | 0.2167 | 0.3900 | 0.2905 | 0.4959 | 0.2536 | 0.4430 | 0.3225 | 1.0000 | 390 | 0.2564 | 0.4484 | 223 | 0 |  |
| 15 | majority_baseline | 3 | 2026-06-25 | 298 | 0.3859 | 0.3333 | 0.1856 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | 183 | 298 |  |
| 15 | return_10_sign_baseline | 3 | 2026-06-25 | 298 | 0.2315 | 0.2513 | 0.1913 | 0.2308 | 0.3789 | 0.2324 | 0.3750 | 0.2316 | 0.3770 | 0.2869 | 1.0000 | 298 | 0.2315 | 0.3770 | 183 | 0 |  |
| 15 | logistic_regression | 1 | 2026-04-15 | 382 | 0.3272 | 0.3266 | 0.3265 | 0.3182 | 0.3125 | 0.3431 | 0.3534 | 0.3306 | 0.3329 | 0.3317 | 0.6466 | 247 | 0.3320 | 0.3347 | 245 | 135 | 1.1572 |
| 15 | logistic_regression | 2 | 2026-05-20 | 390 | 0.3974 | 0.3702 | 0.3649 | 0.2727 | 0.3000 | 0.3494 | 0.2358 | 0.3111 | 0.2679 | 0.2836 | 0.4949 | 193 | 0.3057 | 0.2646 | 223 | 197 | 1.1180 |
| 15 | logistic_regression | 3 | 2026-06-25 | 298 | 0.3859 | 0.3699 | 0.3645 | 0.3235 | 0.2316 | 0.2600 | 0.2955 | 0.2918 | 0.2635 | 0.2733 | 0.5638 | 168 | 0.2857 | 0.2623 | 183 | 130 | 1.1032 |
| 15 | lightgbm | 1 | 2026-04-15 | 382 | 0.3377 | 0.3541 | 0.3237 | 0.2991 | 0.6250 | 0.4545 | 0.2256 | 0.3768 | 0.4253 | 0.3531 | 0.7853 | 300 | 0.3333 | 0.4082 | 245 | 82 | 1.3339 |
| 15 | lightgbm | 2 | 2026-05-20 | 390 | 0.3821 | 0.3685 | 0.3719 | 0.2015 | 0.2700 | 0.4016 | 0.3984 | 0.3016 | 0.3342 | 0.3154 | 0.6564 | 256 | 0.2969 | 0.3408 | 223 | 134 | 1.1543 |
| 15 | lightgbm | 3 | 2026-06-25 | 298 | 0.4027 | 0.3983 | 0.3913 | 0.3710 | 0.2421 | 0.3254 | 0.4659 | 0.3482 | 0.3540 | 0.3381 | 0.6309 | 188 | 0.3404 | 0.3497 | 183 | 110 | 1.1237 |
| 15 | xgboost | 1 | 2026-04-15 | 382 | 0.3272 | 0.3461 | 0.3054 | 0.3033 | 0.6607 | 0.4286 | 0.1805 | 0.3659 | 0.4206 | 0.3348 | 0.7853 | 300 | 0.3267 | 0.4000 | 245 | 82 | 1.2359 |
| 15 | xgboost | 2 | 2026-05-20 | 390 | 0.3795 | 0.3617 | 0.3636 | 0.2119 | 0.2500 | 0.3651 | 0.3740 | 0.2885 | 0.3120 | 0.2994 | 0.6256 | 244 | 0.2910 | 0.3184 | 223 | 146 | 1.1104 |
| 15 | xgboost | 3 | 2026-06-25 | 298 | 0.3960 | 0.3886 | 0.3788 | 0.3103 | 0.1895 | 0.3053 | 0.4545 | 0.3078 | 0.3220 | 0.3003 | 0.6342 | 189 | 0.3069 | 0.3169 | 183 | 109 | 1.1064 |

## Date stability

| horizon_min | model | policy | date_count | dates_with_10plus | min_date_n | max_date_n | median_daily_accuracy | min_daily_accuracy | max_daily_accuracy | median_daily_macro_f1 | median_daily_directional_macro_precision | median_daily_signal_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | lightgbm | oof_selected | 25 | 12 | 1 | 79 | 0.4000 | 0.0000 | 1.0000 | 0.2917 | 0.0625 | 0.6000 |
| 10 | lightgbm | raw | 25 | 12 | 1 | 79 | 0.3810 | 0.0000 | 1.0000 | 0.2917 | 0.1000 | 0.7292 |
| 10 | logistic_regression | oof_selected | 25 | 12 | 1 | 79 | 0.4304 | 0.0000 | 1.0000 | 0.2500 | 0.0000 | 0.3000 |
| 10 | logistic_regression | raw | 25 | 12 | 1 | 79 | 0.4167 | 0.0000 | 1.0000 | 0.3157 | 0.3143 | 0.7143 |
| 10 | majority_baseline | raw | 25 | 12 | 1 | 79 | 0.5696 | 0.0000 | 1.0000 | 0.2419 | 0.0000 | 0.0000 |
| 10 | return_10_sign_baseline | raw | 25 | 12 | 1 | 79 | 0.1087 | 0.0000 | 0.5556 | 0.0794 | 0.1000 | 1.0000 |
| 10 | xgboost | oof_selected | 25 | 12 | 1 | 79 | 0.3816 | 0.0000 | 1.0000 | 0.3333 | 0.2448 | 0.7609 |
| 10 | xgboost | raw | 25 | 12 | 1 | 79 | 0.3542 | 0.0000 | 1.0000 | 0.3162 | 0.2788 | 0.8000 |
| 15 | lightgbm | oof_selected | 22 | 11 | 1 | 83 | 0.3855 | 0.0000 | 1.0000 | 0.2910 | 0.1254 | 0.6198 |
| 15 | lightgbm | raw | 22 | 11 | 1 | 83 | 0.4530 | 0.0000 | 1.0000 | 0.3333 | 0.1904 | 0.7112 |
| 15 | logistic_regression | oof_selected | 22 | 11 | 1 | 83 | 0.4170 | 0.0000 | 1.0000 | 0.2899 | 0.1326 | 0.5651 |
| 15 | logistic_regression | raw | 22 | 11 | 1 | 83 | 0.4170 | 0.0000 | 1.0000 | 0.2899 | 0.1326 | 0.5651 |
| 15 | majority_baseline | raw | 22 | 11 | 1 | 83 | 0.4073 | 0.0000 | 1.0000 | 0.1928 | 0.0000 | 0.0000 |
| 15 | return_10_sign_baseline | raw | 22 | 11 | 1 | 83 | 0.2257 | 0.0000 | 1.0000 | 0.1599 | 0.2323 | 1.0000 |
| 15 | xgboost | oof_selected | 22 | 11 | 1 | 83 | 0.4210 | 0.0000 | 1.0000 | 0.2988 | 0.3066 | 0.7340 |
| 15 | xgboost | raw | 22 | 11 | 1 | 83 | 0.4339 | 0.0000 | 1.0000 | 0.2988 | 0.3066 | 0.7461 |

Per-date detail is saved in `date_stability.csv`. Small per-date samples are noisy and should not be interpreted as independent tests.

## Limitations

- Alpaca IEX minute bars are sparse; samples are available only when the exact future endpoint candle exists, and features reset at every missing-minute gap.
- The eligible sample set is a small subset of all source rows after causal feature warm-up and exact-endpoint eligibility.
- Validation has been used in prior research, so this is not independent final confirmation.
- Classification performance is not evidence of trading profitability; no trading simulation is performed here.
- No candidate is promoted by this experiment.
