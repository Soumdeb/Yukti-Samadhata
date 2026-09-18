# Yutki-Samadhata: Mathematical Foundation of PDHG / PDLP

## 1. Mathematical Formulation

### 1.1 General Primal Linear Program
Yutki-Samadhata targets the general continuous Linear Program:
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^T x \\
\text{subject to} \quad & l_c \le A x \le u_c \\
& l_v \le x \le u_v
\end{aligned}
$$
where:
- $c \in \mathbb{R}^n$ is the objective cost vector.
- $A \in \mathbb{R}^{m \times n}$ is the sparse constraint matrix.
- $l_c \in (\mathbb{R} \cup \{-\infty\})^m$ and $u_c \in (\mathbb{R} \cup \{\infty\})^m$ are row lower and upper bounds.
- $l_v \in (\mathbb{R} \cup \{-\infty\})^n$ and $u_v \in (\mathbb{R} \cup \{\infty\})^n$ are variable lower and upper bounds.

### 1.2 Saddle-Point Reformulation
We express the general LP as an unconstrained minimax problem using convex indicator functions $\mathcal{I}_{\mathcal{X}}$ and $\mathcal{I}_{\mathcal{Y}}$:
$$
\min_{x \in \mathbb{R}^n} \max_{y \in \mathbb{R}^m} \mathcal{L}(x, y) = c^T x + y^T (A x) - g^*(y) + \mathcal{I}_{\mathcal{X}}(x)
$$
where:
- $\mathcal{X} = \{ x \in \mathbb{R}^n \mid l_v \le x \le u_v \}$ is the primal box constraint set.
- $g(u) = \mathcal{I}_{[l_c, u_c]}(u)$ is the indicator function of the constraint range.
- $g^*(y) = \sup_{u \in [l_c, u_c]} \{ y^T u \}$ is the convex conjugate (support function) of $g$:
  $$
  g^*(y) = \sum_{i=1}^m \phi_i(y_i), \quad \phi_i(y_i) = \begin{cases}
  y_i u_{c, i} & \text{if } y_i > 0 \\
  y_i l_{c, i} & \text{if } y_i < 0 \\
  0 & \text{if } y_i = 0
  \end{cases}
  $$
  (with standard limit conventions when $l_{c, i} = -\infty$ or $u_{c, i} = \infty$).

Alternatively, by introducing constraint slack variables $s \in \mathbb{R}^m$ such that:
$$
A x - s = 0, \quad l_c \le s \le u_c, \quad l_v \le x \le u_v
$$
the Lagrangian dual multiplier $y \in \mathbb{R}^m$ associates with $A x - s = 0$:
$$
\max_{y \in \mathbb{R}^m} \min_{\substack{x \in [l_v, u_v] \\ s \in [l_c, u_c]}} c^T x + y^T (s - A x)
$$
This yields symmetric separable subproblems for primal coordinates $(x, s)$ and dual multiplier $y$.

---

## 2. Variables and Coordinates

- **Primal Variables:**
  - $x \in \mathbb{R}^n$: decision variables bounded in $[l_v, u_v]$.
  - $s \in \mathbb{R}^m$: auxiliary constraint slacks bounded in $[l_c, u_c]$.
- **Dual Variables:**
  - $y \in \mathbb{R}^m$: Lagrange multipliers for row constraints.
  - Reduced cost vector: $r = c - A^T y$.

---

## 3. Primal-Dual Hybrid Gradient (PDHG) Algorithm

### 3.1 Step-Size Matrices and Preconditioning
Let $D_x = \text{diag}(\tau_1, \dots, \tau_n) > 0$ and $D_y = \text{diag}(\sigma_1, \dots, \sigma_m) > 0$ represent positive diagonal primal and dual step-size matrices.
For standard Chambolle-Pock iteration, convergence is guaranteed if:
$$
\| D_y^{1/2} A D_x^{1/2} \|_2 < 1
$$
In uniform step-size mode:
$$
\tau \cdot \sigma < \frac{1}{\|A\|_2^2}
$$
where $\|A\|_2$ is the spectral norm of $A$, efficiently bounded or estimated via Power Iteration:
$$
\|A\|_2 \le \sqrt{\|A\|_1 \cdot \|A\|_\infty}
$$

### 3.2 Iteration Scheme (Chambolle-Pock / PDLP Variant)
At iteration $k \ge 0$, given current state $(x^k, s^k, y^k)$ and extrapolated primal state $(\bar{x}^k, \bar{s}^k)$:

1. **Dual Proximal Step:**
   $$
   \tilde{y}^{k+1} = y^k + \sigma \left( A \bar{x}^k - \bar{s}^k \right)
   $$
   Depending on row types:
   - For equality constraint $i$ ($l_{c, i} = u_{c, i} = b_i$): $y_i^{k+1} = \tilde{y}_i^{k+1}$ (unconstrained multiplier).
   - For upper inequality $A_i x \le u_{c, i}$ ($l_{c, i} = -\infty$): $y_i^{k+1} = \min(0, \tilde{y}_i^{k+1})$ or non-positive sign convention depending on formulation.
   - Using the explicit slack formulation:
     $$
     y^{k+1} = y^k + \sigma \left( A \bar{x}^k - s^k \right)
     $$
     and slack projection:
     $$
     s^{k+1} = \text{proj}_{[l_c, u_c]} \left( s^k + \sigma_s y^{k+1} \right)
     $$

2. **Primal Proximal Step:**
   $$
   \tilde{x}^{k+1} = x^k - \tau \left( c - A^T y^{k+1} \right)
   $$
   $$
   x^{k+1} = \text{proj}_{[l_v, u_v]} \left( \tilde{x}^{k+1} \right)
   $$
   where coordinate-wise projection is:
   $$
   \text{proj}_{[l, u]}(v) = \max(l, \min(u, v))
   $$

3. **Extrapolation Step:**
   $$
   \bar{x}^{k+1} = 2 x^{k+1} - x^k
   $$
   $$
   \bar{s}^{k+1} = 2 s^{k+1} - s^k
   $$

---

## 4. Diagonal Preconditioning (Ruiz & Pock-Chambolle Scaling)
Before starting PDHG iterations, the matrix $A$ undergoes diagonal equivalence scaling:
$$
\tilde{A} = R A C
$$
where $R = \text{diag}(r_1, \dots, r_m)$ and $C = \text{diag}(c_1, \dots, c_n)$.
- **Ruiz Equilibration:** Iteratively computes $r_i = 1 / \sqrt{\|A_{i, :}\|_\infty}$ and $c_j = 1 / \sqrt{\|A_{:, j}\|_\infty}$ for $5 \sim 10$ passes until rows and columns have balanced unit $L_\infty$ norms.
- **Pock-Chambolle Preconditioning:**
  $$
  \sigma_i = \frac{1}{\sum_{j=1}^n |A_{ij}|^{2-\alpha}}, \quad \tau_j = \frac{1}{\sum_{i=1}^m |A_{ij}|^\alpha}, \quad \alpha \in [0, 2] \text{ (typically } \alpha = 1\text{)}
  $$
  This ensures $\| D_y^{1/2} A D_x^{1/2} \|_2 \le 1$ without requiring expensive global eigenvalue solves.

---

## 5. Termination Criteria & Residuals

### 5.1 Primal Infeasibility Residual
Let $s^k = \text{proj}_{[l_c, u_c]}(A x^k)$. The absolute primal residual is:
$$
p_{\text{res}} = \| A x^k - s^k \|_2
$$
The relative primal residual:
$$
\epsilon_p = \frac{\| A x^k - s^k \|_2}{1 + \|s^k\|_2}
$$

### 5.2 Dual Infeasibility Residual
Let dual slack / reduced gradient be:
$$
d^k = c - A^T y^k - z^k
$$
where $z^k$ satisfies complementary slackness with respect to $[l_v, u_v]$:
$$
z_j^k = \begin{cases}
\min(0, c_j - (A^T y^k)_j) & \text{if } x_j = u_{v, j} > l_{v, j} \\
\max(0, c_j - (A^T y^k)_j) & \text{if } x_j = l_{v, j} < u_{v, j} \\
0 & \text{if } l_{v, j} < x_j < u_{v, j} \\
c_j - (A^T y^k)_j & \text{if } l_{v, j} = u_{v, j} \text{ (fixed variable)}
\end{cases}
$$
The relative dual residual:
$$
\epsilon_d = \frac{\| c - A^T y^k - z^k \|_2}{1 + \|c\|_2}
$$

### 5.3 Relative Duality Gap
$$
\epsilon_g = \frac{| c^T x^k - (y^k)^T s^k - (z^k)^T x^k |}{1 + |c^T x^k| + |(y^k)^T s^k + (z^k)^T x^k|}
$$

### 5.4 Convergence Check
The solver declares optimality when, for user-configured tolerance $\epsilon_{\text{tol}}$ (e.g., $10^{-6}$):
$$
\epsilon_p \le \epsilon_{\text{tol}}, \quad \epsilon_d \le \epsilon_{\text{tol}}, \quad \epsilon_g \le \epsilon_{\text{tol}}
$$

---

## 6. Adaptive Restarts & Ergodic Averaging
PDHG exhibits $O(1/k)$ ergodic convergence for saddle-point objectives.
- **Iterate Averaging:** The solver tracks running averages:
  $$
  x_{\text{avg}}^K = \frac{1}{\sum w_k} \sum_{k=1}^K w_k x^k, \quad y_{\text{avg}}^K = \frac{1}{\sum w_k} \sum_{k=1}^K w_k y^k
  $$
- **Restart Condition:** Every $K_{\text{check}}$ iterations (e.g., 40 iterations), evaluate the normalized KKT error:
  $$
  \text{err}(x, y) = \max(\epsilon_p, \epsilon_d, \epsilon_g)
  $$
  If $\text{err}(x_{\text{avg}}, y_{\text{avg}}) \le \beta \cdot \text{err}(x_{\text{last\_restart}}, y_{\text{last\_restart}})$ with $\beta \approx 0.8$:
  - Reset current point $(x^k, y^k) \leftarrow (x_{\text{avg}}, y_{\text{avg}})$.
  - Reset ergodic accumulation.
  - Update base error.
  This restart schedule restores linear convergence rates on typical linear programming polyhedra.
