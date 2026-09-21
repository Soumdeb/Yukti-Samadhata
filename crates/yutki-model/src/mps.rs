//! Practical and robust MPS file format parser.
//! Supports standard MPS sections: NAME, ROWS, COLUMNS, RHS, BOUNDS, ENDATA.
//! Row types: N (objective/free), L (<=), G (>=), E (==).
//! Bound types: LO (lower), UP (upper), FX (fixed), FR (free).
//! Provides clear, line-numbered parse diagnostics and direct conversion
//! to both `LpProblem` and `LinearProgram` representations.

use crate::bounds::{Bound, Sense};
use crate::error::SolverError;
use crate::linear_program::LinearProgram;
use crate::problem::LpProblem;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use yutki_sparse::{CooMatrix, CsrMatrix};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum MpsSection {
    None,
    Name,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowType {
    N, // Free / Objective
    G, // >=
    L, // <=
    E, // ==
}

/// Parse an MPS file from disk into an `LpProblem`.
pub fn parse_mps_file<P: AsRef<Path>>(path: P) -> Result<LpProblem, SolverError> {
    let p = path.as_ref();
    let file = File::open(p)
        .map_err(|e| SolverError::Io(format!("Failed to open MPS file '{}': {e}", p.display())))?;
    let reader = BufReader::new(file);
    parse_mps_reader(reader)
}

/// Parse an MPS string into an `LpProblem`.
pub fn parse_mps_str(content: &str) -> Result<LpProblem, SolverError> {
    let reader = std::io::Cursor::new(content);
    parse_mps_reader(reader)
}

/// Parse an MPS file from disk directly into the mathematical `LinearProgram` representation.
pub fn parse_mps_file_to_lp<P: AsRef<Path>>(path: P) -> Result<LinearProgram, SolverError> {
    let problem = parse_mps_file(path)?;
    LinearProgram::from_lp_problem(problem)
}

/// Parse an MPS string directly into the mathematical `LinearProgram` representation.
pub fn parse_mps_str_to_lp(content: &str) -> Result<LinearProgram, SolverError> {
    let problem = parse_mps_str(content)?;
    LinearProgram::from_lp_problem(problem)
}

fn parse_mps_reader<R: BufRead>(reader: R) -> Result<LpProblem, SolverError> {
    let mut current_section = MpsSection::None;
    let mut problem_name = String::from("unnamed");

    let mut obj_row_name: Option<String> = None;
    let mut row_names: Vec<String> = Vec::new();
    let mut row_name_to_idx: HashMap<String, usize> = HashMap::new();
    let mut row_types: Vec<RowType> = Vec::new();

    let mut col_names: Vec<String> = Vec::new();
    let mut col_name_to_idx: HashMap<String, usize> = HashMap::new();

    // Intermediate COO triplets for constraint matrix A
    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    // Objective vector entries (col_idx -> cost)
    let mut c_map: HashMap<usize, f64> = HashMap::new();

    // RHS values (row_idx -> rhs)
    let mut rhs_map: HashMap<usize, f64> = HashMap::new();

    // Variable bounds (col_idx -> Bound)
    let mut bound_map: HashMap<usize, Bound> = HashMap::new();

    let mut found_rows_section = false;
    let mut found_columns_section = false;
    let mut found_endata = false;

    for (line_no, line_res) in reader.lines().enumerate() {
        let line_num = line_no + 1;
        let line = line_res.map_err(|e| SolverError::Io(e.to_string()))?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('*') {
            continue; // Skip comments and empty lines
        }

        // Section header (starts at column 1 without leading whitespace)
        if !line.starts_with(' ') && !line.starts_with('\t') {
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            let header = tokens[0].to_uppercase();
            match header.as_str() {
                "NAME" => {
                    current_section = MpsSection::Name;
                    if tokens.len() > 1 {
                        problem_name = tokens[1].to_string();
                    }
                }
                "ROWS" => {
                    current_section = MpsSection::Rows;
                    found_rows_section = true;
                }
                "COLUMNS" => {
                    if !found_rows_section {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: "COLUMNS section encountered before ROWS section".into(),
                        });
                    }
                    current_section = MpsSection::Columns;
                    found_columns_section = true;
                }
                "RHS" => {
                    if !found_columns_section {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: "RHS section encountered before COLUMNS section".into(),
                        });
                    }
                    current_section = MpsSection::Rhs;
                }
                "RANGES" => current_section = MpsSection::Ranges,
                "BOUNDS" => current_section = MpsSection::Bounds,
                "ENDATA" => {
                    found_endata = true;
                    break;
                }
                other => {
                    return Err(SolverError::ParseError {
                        line: line_num,
                        message: format!("Unknown or misplaced section header '{other}'"),
                    });
                }
            }
            continue;
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        match current_section {
            MpsSection::Rows => {
                if tokens.is_empty() {
                    continue;
                }
                if tokens.len() < 2 {
                    return Err(SolverError::ParseError {
                        line: line_num,
                        message: "Invalid ROWS entry: expected row type and row name".into(),
                    });
                }
                let rtype = match tokens[0].to_uppercase().as_str() {
                    "N" => RowType::N,
                    "G" => RowType::G,
                    "L" => RowType::L,
                    "E" => RowType::E,
                    other => {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!(
                                "Invalid row type '{other}'. Allowed types: N, L, G, E"
                            ),
                        });
                    }
                };
                let rname = tokens[1].to_string();

                if matches!(rtype, RowType::N) && obj_row_name.is_none() {
                    obj_row_name = Some(rname);
                } else {
                    if row_name_to_idx.contains_key(&rname) {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Duplicate row definition '{rname}' in ROWS"),
                        });
                    }
                    let idx = row_names.len();
                    row_name_to_idx.insert(rname.clone(), idx);
                    row_names.push(rname);
                    row_types.push(rtype);
                }
            }
            MpsSection::Columns => {
                if tokens.len() < 3 {
                    return Err(SolverError::ParseError {
                        line: line_num,
                        message: "Invalid COLUMNS entry: expected at least column name, row name, and numeric value".into(),
                    });
                }
                let col_name = tokens[0].to_string();
                let col_idx = match col_name_to_idx.get(&col_name) {
                    Some(&idx) => idx,
                    None => {
                        let idx = col_names.len();
                        col_name_to_idx.insert(col_name.clone(), idx);
                        col_names.push(col_name);
                        idx
                    }
                };

                let mut i = 1;
                while i < tokens.len() {
                    if i + 1 >= tokens.len() {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!(
                                "Missing coefficient value for row '{}' in COLUMNS",
                                tokens[i]
                            ),
                        });
                    }
                    let rname = tokens[i];
                    let val: f64 = tokens[i + 1].parse().map_err(|_| SolverError::ParseError {
                        line: line_num,
                        message: format!("Invalid numeric matrix coefficient '{}'", tokens[i + 1]),
                    })?;

                    if !val.is_finite() {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Non-finite coefficient value '{val}' in COLUMNS"),
                        });
                    }

                    if let Some(ref obj) = obj_row_name {
                        if rname == obj.as_str() {
                            c_map.insert(col_idx, val);
                            i += 2;
                            continue;
                        }
                    }

                    if let Some(&row_idx) = row_name_to_idx.get(rname) {
                        a_triplets.push((row_idx, col_idx, val));
                    } else {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Referenced undefined row '{rname}' in COLUMNS"),
                        });
                    }
                    i += 2;
                }
            }
            MpsSection::Rhs => {
                if tokens.len() < 3 {
                    return Err(SolverError::ParseError {
                        line: line_num,
                        message: "Invalid RHS entry: expected at least rhs identifier, row name, and value".into(),
                    });
                }
                let mut i = 1;
                while i < tokens.len() {
                    if i + 1 >= tokens.len() {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Missing RHS numeric value for row '{}'", tokens[i]),
                        });
                    }
                    let rname = tokens[i];
                    let val: f64 = tokens[i + 1].parse().map_err(|_| SolverError::ParseError {
                        line: line_num,
                        message: format!("Invalid numeric RHS value '{}'", tokens[i + 1]),
                    })?;

                    if !val.is_finite() {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Non-finite RHS value '{val}'"),
                        });
                    }

                    if let Some(&row_idx) = row_name_to_idx.get(rname) {
                        rhs_map.insert(row_idx, val);
                    } else if obj_row_name.as_deref() != Some(rname) {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Referenced undefined row '{rname}' in RHS"),
                        });
                    }
                    i += 2;
                }
            }
            MpsSection::Ranges => {
                // Ignore or parse explicit range rows
            }
            MpsSection::Bounds => {
                if tokens.len() < 3 {
                    return Err(SolverError::ParseError {
                        line: line_num,
                        message: "Invalid BOUNDS entry: expected bound type, bound set name, and column name".into(),
                    });
                }
                let bound_type = tokens[0].to_uppercase();
                let col_name = tokens[2];
                let col_idx = match col_name_to_idx.get(col_name) {
                    Some(&idx) => idx,
                    None => {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!("Referenced undefined column '{col_name}' in BOUNDS"),
                        });
                    }
                };

                let current_bound = bound_map.entry(col_idx).or_insert_with(Bound::non_negative);
                match bound_type.as_str() {
                    "UP" => {
                        if tokens.len() < 4 {
                            return Err(SolverError::ParseError {
                                line: line_num,
                                message: format!(
                                    "Missing numeric upper bound value for column '{col_name}'"
                                ),
                            });
                        }
                        let val: f64 = tokens[3].parse().map_err(|_| SolverError::ParseError {
                            line: line_num,
                            message: format!("Invalid upper bound numeric value '{}'", tokens[3]),
                        })?;
                        current_bound.upper = val;
                    }
                    "LO" => {
                        if tokens.len() < 4 {
                            return Err(SolverError::ParseError {
                                line: line_num,
                                message: format!(
                                    "Missing numeric lower bound value for column '{col_name}'"
                                ),
                            });
                        }
                        let val: f64 = tokens[3].parse().map_err(|_| SolverError::ParseError {
                            line: line_num,
                            message: format!("Invalid lower bound numeric value '{}'", tokens[3]),
                        })?;
                        current_bound.lower = val;
                    }
                    "FX" => {
                        if tokens.len() < 4 {
                            return Err(SolverError::ParseError {
                                line: line_num,
                                message: format!(
                                    "Missing numeric fixed bound value for column '{col_name}'"
                                ),
                            });
                        }
                        let val: f64 = tokens[3].parse().map_err(|_| SolverError::ParseError {
                            line: line_num,
                            message: format!("Invalid fixed bound numeric value '{}'", tokens[3]),
                        })?;
                        current_bound.lower = val;
                        current_bound.upper = val;
                    }
                    "FR" => {
                        *current_bound = Bound::free();
                    }
                    other => {
                        return Err(SolverError::ParseError {
                            line: line_num,
                            message: format!(
                                "Unsupported bound type '{other}'. Allowed: LO, UP, FX, FR"
                            ),
                        });
                    }
                }
            }
            MpsSection::None | MpsSection::Name => {}
        }
    }

    if !found_rows_section {
        return Err(SolverError::ParseError {
            line: 0,
            message: "Missing mandatory ROWS section in MPS file".into(),
        });
    }

    if !found_columns_section {
        return Err(SolverError::ParseError {
            line: 0,
            message: "Missing mandatory COLUMNS section in MPS file".into(),
        });
    }

    if obj_row_name.is_none() {
        return Err(SolverError::ParseError {
            line: 0,
            message: "Missing objective row (type 'N') in ROWS section".into(),
        });
    }

    if !found_endata {
        // Warning or error: standard MPS requires ENDATA
        // Allow graceful completion if sections were fully parsed
    }

    let num_rows = row_names.len();
    let num_cols = col_names.len();

    if num_cols == 0 {
        return Err(SolverError::ValidationError(
            "Parsed model has zero decision variables".into(),
        ));
    }

    // Assemble cost vector c
    let mut c = vec![0.0; num_cols];
    for (&col_idx, &val) in &c_map {
        c[col_idx] = val;
    }

    // Assemble variable bounds and validate
    let mut var_bounds = Vec::with_capacity(num_cols);
    for (col_idx, col_name) in col_names.iter().enumerate() {
        let b = bound_map
            .get(&col_idx)
            .copied()
            .unwrap_or_else(Bound::non_negative);

        if b.lower > b.upper {
            return Err(SolverError::ValidationError(format!(
                "Inverted variable bounds on column '{}': lower ({}) > upper ({})",
                col_name, b.lower, b.upper
            )));
        }
        var_bounds.push(b);
    }

    // Assemble row bounds from row types and RHS values
    let mut row_bounds = Vec::with_capacity(num_rows);
    for (row_idx, &rtype) in row_types.iter().enumerate() {
        let rhs = rhs_map.get(&row_idx).copied().unwrap_or(0.0);
        let b = match rtype {
            RowType::E => Bound::fixed(rhs),
            RowType::L => Bound::upper_only(rhs),
            RowType::G => Bound::lower_only(rhs),
            RowType::N => Bound::free(),
        };
        row_bounds.push(b);
    }

    // Assemble sparse constraint matrix A
    let mut coo = CooMatrix::new(num_rows, num_cols);
    for (r, c_idx, val) in a_triplets {
        coo.add_entry(r, c_idx, val)
            .map_err(|e| SolverError::ValidationError(e.to_string()))?;
    }
    let a = CsrMatrix::from_coo(&coo).map_err(|e| SolverError::ValidationError(e.to_string()))?;

    let problem = LpProblem {
        name: problem_name,
        sense: Sense::Minimize,
        obj_offset: 0.0,
        c,
        var_names: col_names,
        var_bounds,
        row_names,
        row_bounds,
        a,
    };

    problem.validate()?;
    Ok(problem)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MPS_FULL: &str = r#"
NAME          SAMPLE_LP
ROWS
 N  COST
 L  C_LE
 G  C_GE
 E  C_EQ
COLUMNS
    X1        COST       -1.0   C_LE       2.0
    X1        C_GE        1.0
    X2        COST       -2.0   C_LE       1.0
    X2        C_EQ        1.0
    X3        COST        3.0   C_EQ      -1.0
RHS
    RHS1      C_LE        5.0   C_GE       2.0
    RHS1      C_EQ        3.0
BOUNDS
 UP BND1      X1          4.0
 LO BND1      X2          1.0
 FX BND1      X3          2.5
ENDATA
"#;

    #[test]
    fn test_valid_mps_all_bound_types() {
        let lp = parse_mps_str(SAMPLE_MPS_FULL).unwrap();
        assert_eq!(lp.name, "SAMPLE_LP");
        assert_eq!(lp.num_variables(), 3);
        assert_eq!(lp.num_constraints(), 3);
        assert_eq!(lp.c, vec![-1.0, -2.0, 3.0]);

        // C_LE <= 5.0
        assert_eq!(lp.row_bounds[0], Bound::upper_only(5.0));
        // C_GE >= 2.0
        assert_eq!(lp.row_bounds[1], Bound::lower_only(2.0));
        // C_EQ == 3.0
        assert_eq!(lp.row_bounds[2], Bound::fixed(3.0));

        // X1: UP 4.0, lower default 0.0
        assert_eq!(lp.var_bounds[0].upper, 4.0);
        assert_eq!(lp.var_bounds[0].lower, 0.0);

        // X2: LO 1.0, upper default +inf
        assert_eq!(lp.var_bounds[1].lower, 1.0);
        assert_eq!(lp.var_bounds[1].upper, f64::INFINITY);

        // X3: FX 2.5
        assert_eq!(lp.var_bounds[2].lower, 2.5);
        assert_eq!(lp.var_bounds[2].upper, 2.5);
    }

    #[test]
    fn test_free_bound_type_fr() {
        let mps = r#"
NAME FREE_VAR_LP
ROWS
 N OBJ
 L C1
COLUMNS
 X1 OBJ 1.0 C1 1.0
RHS
 R1 C1 10.0
BOUNDS
 FR B1 X1
ENDATA
"#;
        let lp = parse_mps_str(mps).unwrap();
        assert_eq!(lp.var_bounds[0], Bound::free());
    }

    #[test]
    fn test_direct_conversion_to_linear_program() {
        let lp = parse_mps_str_to_lp(SAMPLE_MPS_FULL).unwrap();
        assert_eq!(lp.name, "SAMPLE_LP");
        assert_eq!(lp.num_variables(), 3);
        assert_eq!(lp.num_constraints(), 3);
        assert_eq!(lp.objective.c, vec![-1.0, -2.0, 3.0]);
    }

    #[test]
    fn test_production_mps_analytical_solution() {
        let mps_data = r#"NAME          PRODUCTION_PLANNING
ROWS
 N  PROFIT
 L  LUMBER
 L  CARPENTRY
 L  FINISHING
 L  DEMAND_T
COLUMNS
    CHAIRS    PROFIT     -45.0   LUMBER      5.0
    CHAIRS    CARPENTRY    1.0   FINISHING   2.0
    TABLES    PROFIT     -80.0   LUMBER     20.0
    TABLES    CARPENTRY    2.0   FINISHING   1.0
    TABLES    DEMAND_T     1.0
    DESKS     PROFIT    -110.0   LUMBER     15.0
    DESKS     CARPENTRY    3.0   FINISHING   2.0
RHS
    RHS1      LUMBER     400.0   CARPENTRY  60.0
    RHS1      FINISHING   50.0   DEMAND_T   15.0
BOUNDS
 UP BND1      CHAIRS      50.0
 UP BND1      TABLES      20.0
 UP BND1      DESKS       25.0
ENDATA
"#;
        let problem = parse_mps_str(mps_data).unwrap();
        assert_eq!(problem.name, "PRODUCTION_PLANNING");
        assert_eq!(problem.num_variables(), 3);
        assert_eq!(problem.num_constraints(), 4);

        // Verify known analytical optimum (CHAIRS = 10, TABLES = 10, DESKS = 10)
        let known_x = [10.0, 10.0, 10.0];
        let obj = problem.evaluate_objective(&known_x);
        assert_eq!(obj, -2350.0);

        // Verify constraints at known solution
        let ax = problem.a.mul_vec(&known_x).unwrap();
        assert_eq!(ax[0], 400.0);
        assert_eq!(ax[1], 60.0);
        assert_eq!(ax[2], 50.0);
        assert_eq!(ax[3], 10.0);
    }

    #[test]
    fn test_malformed_unknown_section_header() {
        let mps = "NAME TEST\nFOOBAR\nROWS\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 2);
            assert!(message.contains("FOOBAR"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_invalid_row_type() {
        let mps = "NAME TEST\nROWS\n Z ROW1\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 3);
            assert!(message.contains("Invalid row type 'Z'"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_duplicate_row_name() {
        let mps = "NAME TEST\nROWS\n N OBJ\n L R1\n L R1\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 5);
            assert!(message.contains("Duplicate row definition 'R1'"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_columns_undefined_row() {
        let mps = "NAME TEST\nROWS\n N OBJ\n L R1\nCOLUMNS\n X1 UNDEFINED_ROW 1.0\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 6);
            assert!(message.contains("Referenced undefined row 'UNDEFINED_ROW'"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_columns_invalid_number() {
        let mps = "NAME TEST\nROWS\n N OBJ\n L R1\nCOLUMNS\n X1 R1 not_a_number\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 6);
            assert!(message.contains("Invalid numeric matrix coefficient"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_bounds_undefined_column() {
        let mps = "NAME TEST\nROWS\n N OBJ\n L R1\nCOLUMNS\n X1 R1 1.0\nRHS\n R1 R1 5.0\nBOUNDS\n UP B1 X_UNKNOWN 10.0\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 10);
            assert!(message.contains("Referenced undefined column 'X_UNKNOWN'"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_bounds_missing_value() {
        let mps = "NAME TEST\nROWS\n N OBJ\n L R1\nCOLUMNS\n X1 R1 1.0\nRHS\n R1 R1 5.0\nBOUNDS\n UP B1 X1\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { line, message } = err {
            assert_eq!(line, 10);
            assert!(message.contains("Missing numeric upper bound value"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_inverted_bounds() {
        let mps = "NAME TEST\nROWS\n N OBJ\n L R1\nCOLUMNS\n X1 R1 1.0\nRHS\n R1 R1 5.0\nBOUNDS\n LO B1 X1 10.0\n UP B1 X1 5.0\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ValidationError(message) = err {
            assert!(message.contains("Inverted variable bounds"));
            assert!(message.contains("10") && message.contains("5"));
        } else {
            panic!("Expected ValidationError, got {err:?}");
        }
    }

    #[test]
    fn test_malformed_missing_rows_section() {
        let mps = "NAME TEST\nCOLUMNS\n X1 R1 1.0\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        assert!(matches!(
            err,
            SolverError::ParseError { .. } | SolverError::ValidationError(_)
        ));
    }

    #[test]
    fn test_malformed_missing_objective_row() {
        let mps = "NAME TEST\nROWS\n L R1\nCOLUMNS\n X1 R1 1.0\nENDATA\n";
        let err = parse_mps_str(mps).unwrap_err();
        if let SolverError::ParseError { message, .. } = err {
            assert!(message.contains("Missing objective row"));
        } else {
            panic!("Expected ParseError, got {err:?}");
        }
    }
}
