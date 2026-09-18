NAME          PRODUCTION_PLANNING
* ==============================================================================
* Yutki-Samadhata: Sovereign Optimization Engine Demonstration LP
* Problem Type: Continuous Linear Programming (LP)
* Reference Instance: Multi-Product Production & Resource Allocation
*
* MATHEMATICAL FORMULATION:
* Decision Variables:
*   CHAIRS (x1) : Number of chairs produced, 0 <= x1 <= 50
*   TABLES (x2) : Number of tables produced, 0 <= x2 <= 20
*   DESKS  (x3) : Number of desks produced,  0 <= x3 <= 25
*
* Objective:
*   Maximize Profit:  45 * CHAIRS + 80 * TABLES + 110 * DESKS
*   Expressed as standard minimization:
*   Minimize:        -45 * CHAIRS - 80 * TABLES - 110 * DESKS
*
* Constraints:
*   LUMBER:     5 * CHAIRS + 20 * TABLES + 15 * DESKS <= 400
*   CARPENTRY:  1 * CHAIRS +  2 * TABLES +  3 * DESKS <=  60
*   FINISHING:  2 * CHAIRS +  1 * TABLES +  2 * DESKS <=  50
*   DEMAND_T:                 1 * TABLES              <=  15
*
* KNOWN EXACT ANALYTICAL OPTIMAL SOLUTION:
*   Primal Variables:
*     CHAIRS (x1) = 10.0
*     TABLES (x2) = 10.0
*     DESKS  (x3) = 10.0
*
*   Active Constraints at Optimum:
*     LUMBER:    5(10) + 20(10) + 15(10) = 400.0 (Slack = 0.0)
*     CARPENTRY: 1(10) +  2(10) +  3(10) =  60.0 (Slack = 0.0)
*     FINISHING: 2(10) +  1(10) +  2(10) =  50.0 (Slack = 0.0)
*     DEMAND_T:  10.0 <= 15.0                    (Slack = 5.0)
*
*   Optimal Objective Value:
*     Max Profit = 45(10) + 80(10) + 110(10) = 2350.0
*     Min Objective = -2350.0
*
*   Dual Multipliers (Shadow Prices):
*     y_LUMBER    = 0.875
*     y_CARPENTRY = 28.125
*     y_FINISHING = 6.250
*     y_DEMAND_T  = 0.000
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
