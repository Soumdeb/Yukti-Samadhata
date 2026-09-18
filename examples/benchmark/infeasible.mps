* Deliberately Infeasible Linear Program for Benchmark Robustness Testing
* min x1 + x2
* s.t.
*   SUM_LE: x1 + x2 <= 2
*   SUM_GE: x1 + x2 >= 5
*   x1 >= 0, x2 >= 0
NAME          INFEASIBLE
ROWS
 N  COST
 L  SUM_LE
 G  SUM_GE
COLUMNS
    X1        COST                1.0   SUM_LE              1.0
    X1        SUM_GE              1.0
    X2        COST                1.0   SUM_LE              1.0
    X2        SUM_GE              1.0
RHS
    RHS1      SUM_LE              2.0   SUM_GE              5.0
BOUNDS
 LO BND       X1                  0.0
 LO BND       X2                  0.0
ENDATA
