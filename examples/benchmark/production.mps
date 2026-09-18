NAME          PRODUCTION_PLANNING
* ==============================================================================
* Yutki-Samadhata: Sovereign Optimization Engine Demonstration LP
* Known exact analytical optimal solution: profit = 2350.0, min obj = -2350.0
* ==============================================================================
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
    LIMITS    LUMBER     400.0   CARPENTRY  60.0
    LIMITS    FINISHING   50.0   DEMAND_T   15.0
BOUNDS
 UP BND       CHAIRS      50.0
 UP BND       TABLES      20.0
 UP BND       DESKS       25.0
ENDATA
