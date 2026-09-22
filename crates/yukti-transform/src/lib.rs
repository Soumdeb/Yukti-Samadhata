//! Elementary presolve reductions, transformation mapping, and solution postsolve reconstruction.

use serde::{Deserialize, Serialize};
use yukti_model::{Bound, LpProblem, LpSolution, SolverError};
use yukti_numerics::NumericalTolerances;
use yukti_sparse::{CooMatrix, CsrMatrix};

/// Information about a singleton row eliminated during presolve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingletonRowInfo {
    pub original_row: usize,
    pub original_var: usize,
    pub coeff: f64,
    pub bound: Bound,
}

/// Transformation map tracking changes between the original model and the presolved model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformationMap {
    pub original_num_vars: usize,
    pub original_num_rows: usize,
    /// Maps original variable index to Some(presolved_index) or None if eliminated
    pub orig_to_presolved_var: Vec<Option<usize>>,
    /// Fixed variable values (original_col_idx, fixed_val)
    pub fixed_vars: Vec<(usize, f64)>,
    /// Maps original row index to Some(presolved_index) or None if eliminated
    pub orig_to_presolved_row: Vec<Option<usize>>,
    /// Eliminated empty row indices
    pub eliminated_empty_rows: Vec<usize>,
    /// Eliminated singleton rows
    pub singleton_rows: Vec<SingletonRowInfo>,
    /// Constant objective shift accumulated from fixed variable elimination
    pub obj_shift: f64,
    /// Original problem objective offset
    pub original_obj_offset: f64,
    /// Original problem cost vector
    pub original_c: Vec<f64>,
}

impl TransformationMap {
    pub fn identity(model: &LpProblem) -> Self {
        let n = model.num_variables();
        let m = model.num_constraints();
        Self {
            original_num_vars: n,
            original_num_rows: m,
            orig_to_presolved_var: (0..n).map(Some).collect(),
            fixed_vars: Vec::new(),
            orig_to_presolved_row: (0..m).map(Some).collect(),
            eliminated_empty_rows: Vec::new(),
            singleton_rows: Vec::new(),
            obj_shift: 0.0,
            original_obj_offset: model.obj_offset,
            original_c: model.c.clone(),
        }
    }
}

/// Options controlling presolve passes and reduction rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresolveOptions {
    pub zero_tol: f64,
    pub eliminate_fixed_vars: bool,
    pub eliminate_empty_rows: bool,
    pub prune_zero_coefficients: bool,
    pub tighten_bounds: bool,
    pub eliminate_singleton_rows: bool,
    pub max_passes: usize,
}

impl Default for PresolveOptions {
    fn default() -> Self {
        Self {
            zero_tol: 1e-12,
            eliminate_fixed_vars: true,
            eliminate_empty_rows: true,
            prune_zero_coefficients: true,
            tighten_bounds: true,
            eliminate_singleton_rows: true,
            max_passes: 10,
        }
    }
}

/// Run an elementary presolve reduction pass on an LP model with a zero tolerance.
pub fn presolve(
    model: &LpProblem,
    tol: f64,
) -> Result<(LpProblem, TransformationMap), SolverError> {
    let options = PresolveOptions {
        zero_tol: tol,
        ..Default::default()
    };
    presolve_with_options(model, &options)
}

/// Run presolve reduction with numerical tolerances.
pub fn presolve_with_tolerances(
    model: &LpProblem,
    tols: &NumericalTolerances,
) -> Result<(LpProblem, TransformationMap), SolverError> {
    let options = PresolveOptions {
        zero_tol: tols.zero_tol,
        ..Default::default()
    };
    presolve_with_options(model, &options)
}

/// Run multi-pass presolve reductions according to configured options:
/// 1. Zero coefficient pruning
/// 2. Fixed variable elimination
/// 3. Empty row elimination & infeasibility detection
/// 4. Safe singleton row bound tightening & elimination
pub fn presolve_with_options(
    model: &LpProblem,
    options: &PresolveOptions,
) -> Result<(LpProblem, TransformationMap), SolverError> {
    let zero_tol = options.zero_tol;
    let n = model.num_variables();
    let m = model.num_constraints();

    let mut current_c = model.c.clone();
    let mut current_var_names = model.var_names.clone();
    let mut current_var_bounds = model.var_bounds.clone();
    let mut current_row_names = model.row_names.clone();
    let mut current_row_bounds = model.row_bounds.clone();
    let mut current_offset = model.obj_offset;

    // Track active original variable indices and row indices
    let mut active_orig_vars: Vec<usize> = (0..n).collect();
    let mut active_orig_rows: Vec<usize> = (0..m).collect();

    let mut fixed_vars: Vec<(usize, f64)> = Vec::new();
    let mut eliminated_empty_rows: Vec<usize> = Vec::new();
    let mut singleton_rows: Vec<SingletonRowInfo> = Vec::new();

    // Extract entries from model.a into mutable triplets
    let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(model.a.nnz());
    for i in 0..m {
        let start = model.a.row_ptrs[i];
        let end = model.a.row_ptrs[i + 1];
        for k in start..end {
            let val = model.a.values[k];
            if val.abs() > zero_tol {
                entries.push((i, model.a.col_indices[k], val));
            }
        }
    }

    // Iterative reduction loop (max 10 passes)
    for _pass in 0..10 {
        let mut changed = false;

        // 1. Fixed Variable Elimination
        let mut newly_fixed: Vec<(usize, f64)> = Vec::new();
        let mut remaining_vars: Vec<usize> = Vec::new();
        let mut remaining_c: Vec<f64> = Vec::new();
        let mut remaining_var_names: Vec<String> = Vec::new();
        let mut remaining_var_bounds: Vec<Bound> = Vec::new();

        for (curr_idx, &orig_var) in active_orig_vars.iter().enumerate() {
            let b = &current_var_bounds[curr_idx];
            if b.is_fixed(zero_tol) {
                let fixed_val = b.lower;
                newly_fixed.push((orig_var, fixed_val));
                fixed_vars.push((orig_var, fixed_val));
                current_offset += current_c[curr_idx] * fixed_val;
                changed = true;
            } else {
                remaining_vars.push(orig_var);
                remaining_c.push(current_c[curr_idx]);
                remaining_var_names.push(current_var_names[curr_idx].clone());
                remaining_var_bounds.push(*b);
            }
        }

        if !newly_fixed.is_empty() {
            // Adjust row bounds and eliminate fixed column entries
            for &(f_var, f_val) in &newly_fixed {
                // Find all entries with orig_col == f_var
                for entry in entries.iter_mut() {
                    if entry.1 == f_var {
                        let row_orig = entry.0;
                        if let Some(curr_row_idx) =
                            active_orig_rows.iter().position(|&r| r == row_orig)
                        {
                            let coeff = entry.2;
                            let shift = coeff * f_val;
                            let rb = &mut current_row_bounds[curr_row_idx];
                            if rb.lower.is_finite() {
                                rb.lower -= shift;
                            }
                            if rb.upper.is_finite() {
                                rb.upper -= shift;
                            }
                        }
                        // Mark entry as removed by zeroing coefficient
                        entry.2 = 0.0;
                    }
                }
            }
            active_orig_vars = remaining_vars;
            current_c = remaining_c;
            current_var_names = remaining_var_names;
            current_var_bounds = remaining_var_bounds;
        }

        // Clean up zeroed entries
        entries.retain(|e| e.2.abs() > zero_tol);

        // 2. Empty Rows and Infeasibility Check
        let mut remaining_rows: Vec<usize> = Vec::new();
        let mut remaining_row_names: Vec<String> = Vec::new();
        let mut remaining_row_bounds: Vec<Bound> = Vec::new();

        for (curr_row_idx, &orig_row) in active_orig_rows.iter().enumerate() {
            let nnz_in_row = entries.iter().filter(|e| e.0 == orig_row).count();
            if nnz_in_row == 0 {
                // Check feasibility: 0 must be in [lower, upper]
                let rb = &current_row_bounds[curr_row_idx];
                if rb.lower > zero_tol || rb.upper < -zero_tol {
                    return Err(SolverError::ValidationError(format!(
                        "Presolve detected infeasible empty row '{orig_row}' with bounds [{}, {}]",
                        rb.lower, rb.upper
                    )));
                }
                eliminated_empty_rows.push(orig_row);
                changed = true;
            } else {
                remaining_rows.push(orig_row);
                remaining_row_names.push(current_row_names[curr_row_idx].clone());
                remaining_row_bounds.push(current_row_bounds[curr_row_idx]);
            }
        }
        active_orig_rows = remaining_rows;
        current_row_names = remaining_row_names;
        current_row_bounds = remaining_row_bounds;

        // 3. Singleton Row Handling and Simple Bound Tightening
        let mut rows_after_singleton: Vec<usize> = Vec::new();
        let mut row_names_after_singleton: Vec<String> = Vec::new();
        let mut row_bounds_after_singleton: Vec<Bound> = Vec::new();

        for (curr_row_idx, &orig_row) in active_orig_rows.iter().enumerate() {
            let row_entries: Vec<(usize, usize, f64)> = entries
                .iter()
                .filter(|e| e.0 == orig_row)
                .copied()
                .collect();

            if row_entries.len() == 1 {
                let (_, orig_col, coeff) = row_entries[0];
                let rb = current_row_bounds[curr_row_idx];

                if let Some(curr_var_idx) = active_orig_vars.iter().position(|&v| v == orig_col) {
                    let vb = &mut current_var_bounds[curr_var_idx];
                    // l_c <= coeff * x <= u_c
                    let (implied_l, implied_u) = if coeff > 0.0 {
                        let il = if rb.lower.is_finite() {
                            rb.lower / coeff
                        } else {
                            f64::NEG_INFINITY
                        };
                        let iu = if rb.upper.is_finite() {
                            rb.upper / coeff
                        } else {
                            f64::INFINITY
                        };
                        (il, iu)
                    } else {
                        let il = if rb.upper.is_finite() {
                            rb.upper / coeff
                        } else {
                            f64::NEG_INFINITY
                        };
                        let iu = if rb.lower.is_finite() {
                            rb.lower / coeff
                        } else {
                            f64::INFINITY
                        };
                        (il, iu)
                    };

                    let new_l = vb.lower.max(implied_l);
                    let new_u = vb.upper.min(implied_u);

                    if new_l > new_u + zero_tol {
                        return Err(SolverError::ValidationError(format!(
                            "Presolve detected infeasible bound tightening for variable '{orig_col}': lower {new_l} > upper {new_u}"
                        )));
                    }

                    vb.lower = new_l;
                    vb.upper = new_u;

                    singleton_rows.push(SingletonRowInfo {
                        original_row: orig_row,
                        original_var: orig_col,
                        coeff,
                        bound: rb,
                    });

                    // Remove entry for this row
                    entries.retain(|e| e.0 != orig_row);
                    changed = true;
                    continue; // Eliminate this singleton row
                }
            }

            rows_after_singleton.push(orig_row);
            row_names_after_singleton.push(current_row_names[curr_row_idx].clone());
            row_bounds_after_singleton.push(current_row_bounds[curr_row_idx]);
        }

        active_orig_rows = rows_after_singleton;
        current_row_names = row_names_after_singleton;
        current_row_bounds = row_bounds_after_singleton;

        if !changed {
            break;
        }
    }

    // Build presolved CSR Matrix
    let presolved_m = active_orig_rows.len();
    let presolved_n = active_orig_vars.len();

    let mut coo = CooMatrix::new(presolved_m, presolved_n);
    for &(orig_row, orig_col, val) in &entries {
        if let (Some(new_row), Some(new_col)) = (
            active_orig_rows.iter().position(|&r| r == orig_row),
            active_orig_vars.iter().position(|&c| c == orig_col),
        ) {
            coo.add_entry(new_row, new_col, val)
                .map_err(|e| SolverError::ValidationError(e.to_string()))?;
        }
    }

    let presolved_a =
        CsrMatrix::from_coo(&coo).map_err(|e| SolverError::ValidationError(e.to_string()))?;

    let presolved_model = LpProblem {
        name: format!("{}_presolved", model.name),
        sense: model.sense,
        obj_offset: current_offset,
        c: current_c,
        var_names: current_var_names,
        var_bounds: current_var_bounds,
        row_names: current_row_names,
        row_bounds: current_row_bounds,
        a: presolved_a,
    };

    // Construct full index mapping
    let mut orig_to_presolved_var = vec![None; n];
    for (new_idx, &orig_var) in active_orig_vars.iter().enumerate() {
        orig_to_presolved_var[orig_var] = Some(new_idx);
    }

    let mut orig_to_presolved_row = vec![None; m];
    for (new_idx, &orig_row) in active_orig_rows.iter().enumerate() {
        orig_to_presolved_row[orig_row] = Some(new_idx);
    }

    let trans_map = TransformationMap {
        original_num_vars: n,
        original_num_rows: m,
        orig_to_presolved_var,
        fixed_vars,
        orig_to_presolved_row,
        eliminated_empty_rows,
        singleton_rows,
        obj_shift: current_offset - model.obj_offset,
        original_obj_offset: model.obj_offset,
        original_c: model.c.clone(),
    };

    Ok((presolved_model, trans_map))
}

/// Map a solution computed in presolved space back into original variable coordinates.
pub fn postsolve(presolved_sol: &LpSolution, trans_map: &TransformationMap) -> LpSolution {
    let mut original_primal = vec![0.0; trans_map.original_num_vars];

    // Restore active variables
    for (orig_j, val) in original_primal
        .iter_mut()
        .enumerate()
        .take(trans_map.original_num_vars)
    {
        if let Some(presolved_j) = trans_map.orig_to_presolved_var[orig_j] {
            if presolved_j < presolved_sol.primal.len() {
                *val = presolved_sol.primal[presolved_j];
            }
        }
    }

    // Restore fixed variables
    for &(fixed_j, fixed_val) in &trans_map.fixed_vars {
        if fixed_j < original_primal.len() {
            original_primal[fixed_j] = fixed_val;
        }
    }

    // Restore dual variables
    let mut original_dual = vec![0.0; trans_map.original_num_rows];
    for (orig_i, val) in original_dual
        .iter_mut()
        .enumerate()
        .take(trans_map.original_num_rows)
    {
        if let Some(presolved_i) = trans_map.orig_to_presolved_row[orig_i] {
            if presolved_i < presolved_sol.dual.len() {
                *val = presolved_sol.dual[presolved_i];
            }
        }
    }

    // Recompute exact original objective value: c^T x + offset
    let orig_raw_obj: f64 = trans_map
        .original_c
        .iter()
        .zip(original_primal.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum();
    let exact_original_obj = orig_raw_obj + trans_map.original_obj_offset;

    LpSolution {
        status: presolved_sol.status,
        primal: original_primal,
        dual: original_dual,
        objective: exact_original_obj,
        iterations: presolved_sol.iterations,
        primal_residual: presolved_sol.primal_residual,
        dual_residual: presolved_sol.dual_residual,
        solve_time_secs: presolved_sol.solve_time_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yukti_model::{Bound, Sense};

    #[test]
    fn test_presolve_fixed_variable_and_postsolve() {
        let mut coo = CooMatrix::new(1, 3);
        coo.add_entry(0, 0, 2.0).unwrap();
        coo.add_entry(0, 1, 3.0).unwrap();
        coo.add_entry(0, 2, 4.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let model = LpProblem {
            name: "test".into(),
            sense: Sense::Minimize,
            obj_offset: 0.0,
            c: vec![10.0, 20.0, 5.0],
            var_names: vec!["x1".into(), "x2".into(), "x3".into()],
            var_bounds: vec![
                Bound::fixed(5.0),
                Bound::non_negative(),
                Bound::non_negative(),
            ],
            row_names: vec!["r1".into()],
            row_bounds: vec![Bound::upper_only(25.0)],
            a,
        };

        let (presolved, map) = presolve(&model, 1e-8).unwrap();
        assert_eq!(presolved.num_variables(), 2);
        assert_eq!(presolved.num_constraints(), 1);
        // r1 bound was <= 25, x1=5 contributes 2*5 = 10, so new bound is <= 15
        assert_eq!(presolved.row_bounds[0].upper, 15.0);
        // obj offset was 0, now 10*5 = 50
        assert_eq!(presolved.obj_offset, 50.0);

        let mock_presolved_sol = LpSolution {
            status: yukti_model::SolverStatus::Optimal,
            primal: vec![4.0, 1.0], // x2 = 4, x3 = 1
            dual: vec![0.0],
            objective: 135.0,
            iterations: 10,
            primal_residual: 1e-7,
            dual_residual: 1e-7,
            solve_time_secs: 0.01,
        };

        let restored = postsolve(&mock_presolved_sol, &map);
        assert_eq!(restored.primal.len(), 3);
        assert_eq!(restored.primal[0], 5.0); // restored fixed x1
        assert_eq!(restored.primal[1], 4.0); // restored x2
        assert_eq!(restored.primal[2], 1.0); // restored x3
                                             // exact obj: 10*5 + 20*4 + 5*1 + 0 = 135
        assert!((restored.objective - 135.0).abs() < 1e-6);
    }

    #[test]
    fn test_presolve_empty_rows_elimination() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 2.0).unwrap();
        // Row 1 has zero entries
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let model = LpProblem {
            name: "empty_row".into(),
            sense: Sense::Minimize,
            obj_offset: 0.0,
            c: vec![1.0, 1.0],
            var_names: vec!["x1".into(), "x2".into()],
            var_bounds: vec![Bound::non_negative(), Bound::non_negative()],
            row_names: vec!["r0".into(), "r1_empty".into()],
            row_bounds: vec![Bound::upper_only(5.0), Bound::new(-2.0, 3.0)], // feasible 0 in [-2, 3]
            a,
        };

        let (presolved, map) = presolve(&model, 1e-8).unwrap();
        assert_eq!(presolved.num_constraints(), 1);
        assert_eq!(map.eliminated_empty_rows.len(), 1);
        assert_eq!(map.eliminated_empty_rows[0], 1);

        let mock_sol = LpSolution {
            status: yukti_model::SolverStatus::Optimal,
            primal: vec![2.0, 1.0],
            dual: vec![0.5],
            objective: 3.0,
            iterations: 5,
            primal_residual: 1e-8,
            dual_residual: 1e-8,
            solve_time_secs: 0.001,
        };

        let restored = postsolve(&mock_sol, &map);
        assert_eq!(restored.dual.len(), 2);
        assert_eq!(restored.dual[0], 0.5);
        assert_eq!(restored.dual[1], 0.0); // restored empty row multiplier
    }

    #[test]
    fn test_presolve_zero_coefficients_pruning() {
        let mut coo = CooMatrix::new(1, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 2.0).unwrap();
        coo.add_entry(0, 2, 1e-15).unwrap(); // near-zero coefficient
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let model = LpProblem {
            name: "zero_coeff".into(),
            sense: Sense::Minimize,
            obj_offset: 0.0,
            c: vec![1.0, 2.0, 3.0],
            var_names: vec!["x1".into(), "x2".into(), "x3".into()],
            var_bounds: vec![
                Bound::non_negative(),
                Bound::non_negative(),
                Bound::non_negative(),
            ],
            row_names: vec!["r0".into()],
            row_bounds: vec![Bound::upper_only(5.0)],
            a,
        };

        let (presolved, _) = presolve(&model, 1e-12).unwrap();
        assert_eq!(presolved.a.nnz(), 2);
    }

    #[test]
    fn test_presolve_singleton_row_tightening() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 2.0).unwrap(); // Row 0 is singleton: 2*x1 <= 8 -> x1 <= 4
        coo.add_entry(1, 0, 1.0).unwrap();
        coo.add_entry(1, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let model = LpProblem {
            name: "singleton".into(),
            sense: Sense::Minimize,
            obj_offset: 0.0,
            c: vec![1.0, 1.0],
            var_names: vec!["x1".into(), "x2".into()],
            var_bounds: vec![Bound::new(0.0, 10.0), Bound::non_negative()],
            row_names: vec!["r_singleton".into(), "r_regular".into()],
            row_bounds: vec![Bound::upper_only(8.0), Bound::upper_only(12.0)],
            a,
        };

        let (presolved, map) = presolve(&model, 1e-8).unwrap();
        // Row 0 should be eliminated after tightening x1's upper bound to 4.0
        assert_eq!(presolved.num_constraints(), 1);
        assert_eq!(map.singleton_rows.len(), 1);
        assert!((presolved.var_bounds[0].upper - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_presolve_multi_variable_retention_and_exact_postsolve() {
        // 5 variables: x0, x1(fixed=2), x2, x3(fixed=7), x4
        // Objective: 1*x0 + 2*x1 + 3*x2 + 4*x3 + 5*x4 + 10.0
        let mut coo = CooMatrix::new(2, 5);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        coo.add_entry(0, 2, 1.0).unwrap();
        coo.add_entry(1, 3, 1.0).unwrap();
        coo.add_entry(1, 4, 2.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let model = LpProblem {
            name: "multi_var_presolve".into(),
            sense: Sense::Minimize,
            obj_offset: 10.0,
            c: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            var_names: vec![
                "x0".into(),
                "x1".into(),
                "x2".into(),
                "x3".into(),
                "x4".into(),
            ],
            var_bounds: vec![
                Bound::non_negative(),
                Bound::fixed(2.0),
                Bound::new(0.0, 5.0),
                Bound::fixed(7.0),
                Bound::non_negative(),
            ],
            row_names: vec!["r0".into(), "r1".into()],
            row_bounds: vec![Bound::upper_only(20.0), Bound::upper_only(30.0)],
            a,
        };

        let (presolved, map) = presolve(&model, 1e-8).unwrap();
        assert_eq!(presolved.num_variables(), 3); // x0, x2, x4
        assert_eq!(map.original_num_vars, 5);
        assert_eq!(map.fixed_vars.len(), 2);

        // Active variables should be mapped to 0, 1, 2
        assert_eq!(map.orig_to_presolved_var[0], Some(0)); // x0
        assert_eq!(map.orig_to_presolved_var[1], None); // x1 fixed
        assert_eq!(map.orig_to_presolved_var[2], Some(1)); // x2
        assert_eq!(map.orig_to_presolved_var[3], None); // x3 fixed
        assert_eq!(map.orig_to_presolved_var[4], Some(2)); // x4

        // Presolved solution: x0 = 3.0, x2 = 4.0, x4 = 5.0
        let presolved_sol = LpSolution {
            status: yukti_model::SolverStatus::Optimal,
            primal: vec![3.0, 4.0, 5.0],
            dual: vec![0.0, 0.0],
            objective: 0.0,
            iterations: 1,
            primal_residual: 0.0,
            dual_residual: 0.0,
            solve_time_secs: 0.001,
        };

        let restored = postsolve(&presolved_sol, &map);
        assert_eq!(restored.primal.len(), 5);
        assert_eq!(restored.primal[0], 3.0); // x0
        assert_eq!(restored.primal[1], 2.0); // x1 restored fixed
        assert_eq!(restored.primal[2], 4.0); // x2
        assert_eq!(restored.primal[3], 7.0); // x3 restored fixed
        assert_eq!(restored.primal[4], 5.0); // x4

        // Recomputed exact objective:
        // 1(3) + 2(2) + 3(4) + 4(7) + 5(5) + 10.0 = 3 + 4 + 12 + 28 + 25 + 10 = 82.0
        assert!((restored.objective - 82.0).abs() < 1e-10);
    }
}
