# Publication Refactor Change Log

This change log documents the cleaned Python script and Jupyter notebook produced for a public research repository or journal replication package. The scientific methodology, parameter values, numerical estimands, and generated output filenames were preserved. Textual analysis labels were revised where they used unclear implementation-specific wording.

## Returned files

- `finite_superlevel_topology_pipeline.py`: cleaned command-line replication script.
- `finite_superlevel_topology_pipeline.ipynb`: cleaned notebook entry point that calls the same script.
- `publication_refactor_changelog.md`: this change log.

## Removed unused code

| Removed name | Reason |
|---|---|
| `resample_points_by_arm` | `removed (unused helper; no call sites)` |
| `weighted_quantile` | `removed (unused helper; no call sites)` |

## Renamed constants and module-level configuration

| Before | After |
|---|---|
| `BG_STRUCT` | `BACKGROUND_CONNECTIVITY` |
| `DEFAULT_EICU_DATA_DIR` | `DEFAULT_EICU_DIRECTORY` |
| `FG_STRUCT` | `FOREGROUND_CONNECTIVITY` |
| `FIXED_BOUNDS` | `DEFAULT_TOPOLOGY_BOUNDS` |
| `JOBS_II_ICPSR_DOI` | `JOBS_II_ARCHIVE_DOI` |
| `JOBS_II_ICPSR_URL` | `JOBS_II_ARCHIVE_URL` |
| `JOBS_II_RDATSETS_URL` | `JOBS_II_RDATASETS_CSV_URL` |
| `RNG_SEED` | `DEFAULT_RANDOM_SEED` |

## Renamed functions

| Before | After |
|---|---|
| `_standardize_columns` | `_standardize_nonconstant_columns` |
| `analyze_dataset` | `analyze_structural_benchmark` |
| `analyze_observational_dataset` | `analyze_observational_application` |
| `analyze_randomized_dataset` | `analyze_randomized_application` |
| `assign_synthetic_treatment` | `assign_structural_benchmark_treatment` |
| `benjamini_hochberg_q_values` | `compute_benjamini_hochberg_q_values` |
| `betti_curves_from_density` | `compute_superlevel_betti_curves` |
| `binary_effect_sensitivity_table` | `compute_binary_outcome_sensitivity` |
| `bool_tex` | `format_bool_for_latex` |
| `bootstrap_sensitivity` | `run_structural_bootstrap_sensitivity` |
| `bootstrap_sensitivity_observational` | `run_observational_bootstrap_sensitivity` |
| `bootstrap_sensitivity_randomized` | `run_randomized_bootstrap_sensitivity` |
| `causal_benchmark_metric_table` | `summarize_structural_benchmark_mean_effects` |
| `causal_benchmark_score_bin_effects` | `summarize_structural_benchmark_score_bin_effects` |
| `compact_float` | `format_compact_float` |
| `compute_design_diagnostics` | `compute_design_and_sensitivity_diagnostics` |
| `count_connected_components` | `count_foreground_components` |
| `covariate_balance_diagnostics` | `compute_covariate_balance_diagnostics` |
| `covariate_balance_summary` | `summarize_covariate_balance` |
| `covariate_design_matrix` | `build_covariate_design_matrix` |
| `cross_fitted_aipw_mean_contrasts` | `estimate_cross_fitted_aipw_mean_contrasts` |
| `default_level_fracs` | `default_superlevel_fractions` |
| `density_grid` | `estimate_smoothed_density_grid` |
| `diagnostic_topology_parameters` | `default_diagnostic_topology_parameters` |
| `e_value_from_risk_ratio` | `compute_e_value_for_risk_ratio` |
| `ect_coordinate_index` | `decode_sliced_euler_coordinate` |
| `ect_effect_from_details` | `sliced_euler_effect_from_details` |
| `ect_signature_from_density` | `compute_sliced_euler_signature_from_density` |
| `ect_slice_chamber_detail_table` | `build_sliced_euler_coordinate_check_table` |
| `ect_slice_chamber_detail_table_for_bins` | `build_binned_sliced_euler_coordinate_check_table` |
| `ect_slice_chamber_stats` | `compare_sliced_euler_effect_to_reference` |
| `effective_sample_size` | `compute_effective_sample_size` |
| `euler_characteristic` | `compute_euler_characteristic` |
| `finite_signature_chamber_diagnostics` | `compute_structural_signature_chamber_checks` |
| `finite_signature_tuning_sensitivity` | `run_structural_signature_tuning_sensitivity` |
| `finite_signature_tuning_sensitivity_without_oracle` | `run_empirical_signature_tuning_sensitivity` |
| `fit_propensity_model` | `estimate_propensity_scores` |
| `generate_connected_bar_outcomes` | `generate_single_component_outcomes` |
| `generate_two_cluster_outcomes` | `generate_two_component_outcomes` |
| `inverse_probability_weights` | `compute_inverse_probability_weights` |
| `latent_confounding_diagnostics` | `compute_latent_variable_diagnostics` |
| `level_margin_from_density` | `compute_level_grid_margin` |
| `log_progress` | `print_progress` |
| `metric_value` | `extract_metric_value` |
| `norm_l1` | `mean_absolute_signature_difference` |
| `norm_l2` | `root_mean_square_signature_difference` |
| `observational_metric_table` | `summarize_observational_mean_differences` |
| `observational_score_bin_effects` | `summarize_observational_score_bin_differences` |
| `odds_tilt_sensitivity` | `run_structural_odds_tilt_sensitivity` |
| `odds_tilt_sensitivity_observational` | `run_observational_odds_tilt_sensitivity` |
| `odds_tilt_sensitivity_randomized` | `run_randomized_odds_tilt_sensitivity` |
| `odds_tilted_propensity` | `apply_odds_ratio_propensity_tilt` |
| `omnibus_balance_permutation_test` | `run_omnibus_balance_permutation_test` |
| `omnibus_balance_statistic` | `compute_omnibus_balance_statistic` |
| `outcome_distribution_permutation_tests` | `run_outcome_distribution_permutation_tests` |
| `overlap_diagnostics` | `summarize_propensity_overlap` |
| `overlap_trim_sensitivity` | `compute_overlap_trim_sensitivity` |
| `overlap_weights` | `compute_overlap_weights` |
| `propensity_overlap_diagnostics` | `compute_propensity_overlap_diagnostics` |
| `quantile_bins` | `assign_quantile_bins` |
| `randomized_metric_table` | `summarize_randomized_mean_differences` |
| `randomized_score_bin_effects` | `summarize_randomized_score_bin_differences` |
| `remove_small_components` | `remove_small_foreground_components` |
| `rows_for_cate_topology_table` | `build_score_bin_topology_comparison_table` |
| `sampled_energy_distance` | `estimate_sampled_energy_distance` |
| `sampled_rbf_mmd2` | `estimate_sampled_rbf_mmd2` |
| `save_observational_summary_figure` | `save_empirical_summary_figure` |
| `save_summary_figure` | `save_structural_summary_figure` |
| `score_bin_topology_contrasts_for_causal_benchmark` | `compute_structural_benchmark_binned_topology_contrasts` |
| `score_bin_topology_contrasts_without_oracle` | `compute_empirical_binned_topology_contrasts` |
| `standardized_mean_difference_table` | `compute_standardized_mean_differences` |
| `summarize_bootstrap` | `summarize_bootstrap_results` |
| `superlevel_signature_from_density` | `compute_superlevel_topological_signature` |
| `topology_contrast_superlevel` | `compute_superlevel_topology_contrast` |
| `topology_contrasts_for_causal_benchmark` | `compute_structural_benchmark_topology_contrasts` |
| `topology_contrasts_without_oracle_outcomes` | `compute_empirical_topology_contrasts` |
| `topology_permutation_tests` | `run_topological_signature_permutation_tests` |
| `topology_stability_against_reference` | `compare_topology_contrasts_to_reference` |
| `variance_ratio_or_inverse` | `compute_symmetric_variance_ratio` |
| `weighted_ks_statistic` | `compute_weighted_ks_statistic` |
| `weighted_mean` | `compute_weighted_mean` |
| `weighted_mean_difference` | `compute_weighted_mean_difference` |
| `weighted_variance` | `compute_weighted_variance` |
| `write_tex_summary` | `write_latex_summary` |

## Renamed variables and local data objects

| Before | After |
|---|---|
| `X_design` | `covariate_design` |
| `X_imp` | `imputed_covariate_design` |
| `X_numeric` | `numeric_covariates` |
| `Xdf` | `covariates` |
| `Xs` | `standardized_covariates` |
| `a` | `treatment` |
| `aipw_df` | `aipw_table` |
| `bal_diag` | `balance_diagnostic_table` |
| `bal_omni` | `omnibus_balance_table` |
| `bal_summary` | `balance_summary_table` |
| `base` | `dataset_output_dir` |
| `binary_sensitivity_df` | `binary_sensitivity_table` |
| `boot_df` | `bootstrap_table` |
| `boot_summary_df` | `bootstrap_summary_table` |
| `bundle` | `dataset_bundle` |
| `bundles` | `dataset_bundles` |
| `cate_df` | `score_bin_effect_table` |
| `chamber_bin_agg_df` | `binned_signature_chamber_summary_table` |
| `chamber_bin_df` | `binned_signature_chamber_table` |
| `chamber_df` | `signature_chamber_table` |
| `data_dir` | `input_data_dir` |
| `details` | `topology_details` |
| `distribution_tests` | `distribution_test_table` |
| `e_true` | `true_propensity_score` |
| `ect_slice_bin_df` | `binned_sliced_euler_coordinate_table` |
| `ect_slice_df` | `sliced_euler_coordinate_table` |
| `ehat` | `propensity_score` |
| `energy_df` | `energy_distance_table` |
| `eta` | `structural_score` |
| `eta_raw` | `raw_structural_score` |
| `eta_true` | `structural_score` |
| `faster_topo_kwargs` | `sensitivity_topology_parameters` |
| `gamma_df` | `odds_tilt_table` |
| `main_topo_kwargs` | `primary_topology_parameters` |
| `method_note` | `method_description` |
| `metric_df` | `mean_effect_table` |
| `n_boot` | `n_bootstrap` |
| `n_perm` | `n_permutations` |
| `nbins` | `n_bins` |
| `outdir` | `output_dir` |
| `overlap_df` | `overlap_summary_table` |
| `overlap_ext` | `overlap_diagnostic_table` |
| `pred` | `prediction_table` |
| `pred_cols` | `prediction_columns` |
| `pred_path` | `prediction_file` |
| `raw_smd` | `raw_smd_table` |
| `result_cols` | `result_columns` |
| `result_path` | `result_file` |
| `score_topo_details` | `binned_topology_details` |
| `score_topo_df` | `binned_topology_table` |
| `score_topo_std_df` | `standardized_binned_topology_table` |
| `settings_grid` | `signature_parameter_grid` |
| `topo` | `topology_tests` |
| `topo_kwargs` | `topology_parameters` |
| `topo_perm` | `topology_permutations` |
| `topology_df` | `topology_contrast_table` |
| `topology_tests` | `topology_test_table` |
| `trim_df` | `trim_sensitivity_table` |
| `tuning_df` | `signature_tuning_table` |
| `u` | `unobserved_group` |
| `w_arr` | `weight_array` |
| `w_ate` | `ate_weights` |
| `w_overlap` | `overlap_weight_values` |
| `wt_smd` | `weighted_smd_table` |
| `y0` | `control_potential_outcomes` |
| `y1` | `treated_potential_outcomes` |
| `y_binary` | `binary_outcome` |
| `y_obs` | `observed_outcomes` |
| `y_raw` | `raw_outcomes` |

## Renamed analysis labels and entry-point files

| Before | After |
|---|---|
| `topological_ignorability_superlevel_pipeline.py` | `finite_superlevel_topology_pipeline.py` |
| `topological_ignorability_superlevel_pipeline(5).ipynb` | `finite_superlevel_topology_pipeline.ipynb` |
| `observed_raw` | `unadjusted_observed` |
| `IPW_observed_Z_weighted_density` | `ipw_adjusted_observed_density` |
| `overlap_weighted_observed_Z_density` | `overlap_weighted_observed_density` |
| `IPW_observed_Z_diff_mean_x/y` | `ipw_observed_covariate_diff_mean_x/y` |
| `overlap_weighted_observed_Z_diff_mean_x/y` | `overlap_weighted_observed_covariate_diff_mean_x/y` |

## Command-line options

No command-line option names were removed or renamed. Help text and documentation were revised for clarity and reproducibility.

| Option | Status |
|---|---|
| `--outdir` | `retained; default remains relative path `superlevel_outputs`` |
| `--bootstrap` | `retained; help text clarified` |
| `--diagnostic-permutations` | `retained; help text revised from publication wording to design-diagnostic wording` |
| `--eicu-dir` | `retained; default remains relative path `data/eicu`` |
| `--jobs-csv` | `retained; help text clarifies local archival replication` |
| `--jobs-url` | `retained; default Rdatasets URL unchanged` |
| `--skip-jobs` | `retained` |
| `--skip-eicu` | `retained` |
| `--no-progress` | `retained` |

## Generated output files

Generated output filenames were retained to preserve downstream reproducibility. Files are written under `--outdir/<dataset-name>/` unless noted as root outputs. The script also writes `output_manifest.csv` in the root output directory.

- `balance_raw_smd.csv`
- `balance_summary.csv`
- `balance_weighted_smd.csv`
- `balancing_bin_chamber_check_summary.csv`
- `balancing_bin_chamber_checks.csv`
- `balancing_bin_ect_slice_chamber_checks.csv`
- `balancing_bin_ect_slice_stability_checks.csv`
- `bootstrap_sensitivity.csv`
- `bootstrap_sensitivity_summary.csv`
- `causal_benchmark_metric_targets.csv`
- `causal_benchmark_score_bin_effects.csv`
- `covariate_balance_diagnostics.csv`
- `design_balance_omnibus_tests.csv`
- `design_binary_evalue_sensitivity.csv`
- `design_crossfit_aipw_mean_checks.csv`
- `design_distribution_permutation_tests.csv`
- `design_overlap_trim_sensitivity.csv`
- `design_overlap_weight_diagnostics.csv`
- `design_topology_permutation_tests.csv`
- `ect_slice_chamber_checks.csv`
- `ect_slice_stability_checks.csv`
- `energy_distance_checks.csv`
- `finite_signature_chamber_checks.csv`
- `finite_signature_tuning_sensitivity.csv`
- `latent_confounding_by_score_bins.csv`
- `observational_cohort_metadata.csv`
- `observational_metric_diffs.csv`
- `observational_score_bin_effects.csv`
- `observational_topology_stability_checks.csv`
- `odds_tilt_sensitivity.csv`
- `output_manifest.csv`
- `overlap_diagnostics.csv`
- `randomized_application_metadata.csv`
- `randomized_metric_diffs.csv`
- `randomized_score_bin_effects.csv`
- `randomized_topology_stability_checks.csv`
- `superlevel_results_summary.tex`
- `topology_balancing_bin_contrasts.csv`
- `topology_balancing_bin_standardized.csv`
- `topology_superlevel_contrasts.csv`

## Documentation and wording changes

- Replaced development-history wording with neutral methodological documentation.
- Replaced “observed-Z” wording with “observed-covariate” or descriptive analysis labels.
- Replaced “hidden-bias” wording with “unmeasured-confounding sensitivity” and “hidden U” wording with “unobserved U”.
- Revised report text so diagnostics are described as empirical design and stability checks, not as proof of exchangeability, conditional topological identifying restrictions, or other assumptions.
- Distinguished causal structural benchmarks, randomized empirical applications, observational analyses, and descriptive topological illustrations in the script docstring, generated report text, and notebook Markdown.
- Added concise installation, command-line, optional eICU, local JOBS II, and output-location instructions to the script and notebook.

## Validation performed

- `python -m py_compile finite_superlevel_topology_pipeline.py` completed successfully.
- The cleaned module imported successfully.
- A reduced synthetic end-to-end smoke run completed, wrote CSV/PNG outputs, and generated `superlevel_results_summary.tex`.
- A deterministic equivalence check against the original script on a reduced synthetic benchmark found identical generated arrays, fitted propensities, weights, and selected topology numeric outputs after mapping renamed analysis labels; maximum absolute numerical difference was `0.0` for the checked quantities.
- The full default archival run was not completed during the refactor because it is computationally long; the reduced end-to-end and deterministic equivalence checks were used to verify that refactoring did not alter the computational logic.

## Detected methodological or computational issues

1. No result-changing programming error was identified during the refactor. The removed helpers had no call sites and therefore could not affect analyses.
2. The weighted omnibus balance permutation diagnostic keeps fitted weights fixed while permuting treatment labels. This is acceptable as a descriptive design stress diagnostic under the implemented fixed-weight null, but it should not be described as a formal randomization test that re-estimates the propensity model under every permutation. The code was not changed to preserve reported results.
3. The default JOBS II path reads a public Rdatasets URL. For archival reproducibility without network dependence, use `--jobs-csv` with a local copy. The default was retained to preserve the existing behavior.
4. The optional eICU branch is observational and cannot identify causal effects without additional assumptions. The language was revised to present it as a descriptive sensitivity analysis only.
5. Sliced Euler coordinates remain descriptive unless the finite coordinate agreement checks pass for the relevant comparison. The report text was revised to avoid claiming that these diagnostics prove identifying assumptions.
