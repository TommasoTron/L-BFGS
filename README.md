## Problem statement

We consider the classical unconstrained non-linear optimization problem

\[
\min_{x \in \mathbb{R}^n} f(x),
\]

where \( f : \mathbb{R}^n \to \mathbb{R} \) is assumed to be sufficiently smooth (typically at least twice continuously differentiable).
A standard second-order method for this problem is **Newton’s method**, whose iteration can be written as

\[
H_k \Delta_k = -\nabla f(x_k), \quad x_{k+1} = x_k + \Delta_k,
\]

where \(H_k\) is the Hessian of \(f\) at \(x_k\) and \(\nabla f(x_k)\) is the gradient. Newton’s method can achieve quadratic convergence near the solution but it has some drawbacks:

- computing and storing the full Hessian \(H_k \in \mathbb{R}^{n \times n}\) is expensive for large \(n\);
- solving a linear system with \(H_k\) at each iteration is also costly;
- global convergence typically requires safeguards, such as **line-search** or **trust-region** strategies.

For moderately large or large-scale problems, forming and storing the full Hessian quickly becomes prohibitive, and more efficient approaches are required. **Quasi-Newton methods** address this by building and updating an approximation of the Hessian (or its inverse) using only gradient information, while still retaining superlinear convergence under suitable conditions.

This project focuses on two such methods: **BFGS** and **L-BFGS**.

---

## Algorithms

This project implements/analyzes Quasi-Newton methods for unconstrained non-linear optimization.

### BFGS (Broyden–Fletcher–Goldfarb–Shanno)

BFGS is an iterative method for solving unconstrained non-linear optimization problems. As a Quasi-Newton method, it approximates the Hessian matrix of the objective function using gradient evaluations.

- **Mechanism:** It maintains a full dense approximation of the (inverse) Hessian matrix and updates it at each step so that the **secant equation** is satisfied. The search direction is obtained by solving a linear system involving this matrix.
- **Performance:** It offers superlinear convergence under standard assumptions and is generally robust in practice.
- **Memory cost:** \(O(n^2)\), where \(n\) is the number of variables. This makes it suitable for small to medium-sized problems, where storing an \(n \times n\) matrix is still feasible.

### L-BFGS (Limited-memory BFGS)

L-BFGS is an optimization algorithm in the family of Quasi-Newton methods that approximates the BFGS algorithm using a limited amount of memory.

- **Mechanism:** Instead of storing the full \(n \times n\) inverse Hessian matrix, it stores only a small number of vector pairs \((s_k, y_k)\) that implicitly represent the quasi-Newton approximation. It maintains a history of the past \(m\) updates (where \(m\) is typically small, e.g., 5–20) and uses the **two-loop recursion** to apply the inverse Hessian approximation to a vector.
- **Performance:** It enjoys similar convergence properties to full BFGS in many cases, though it can be slightly less robust on very ill-conditioned problems.
- **Memory cost:** \(O(mn)\), which makes it well suited for large-scale problems with thousands or millions of variables, as the memory footprint grows only linearly with the problem dimension.

---

### Organization of the code

The directory structure is as follows:

```text
./Project
  ├── build/
  ├── lib/
  ├── paper/
  └── src/
```

- `build` contains the output of CMAKE
- `lib` contains the external libraries
- `paper`containes the academic literature provided
- `src` contains all the C++ code to be compiled

### Compiling

To compile we use CMAKE, from the `Project` directory run:

```zsh
mkdir build
cd build
cmake ..
make
```
#### Run
To run the test from the `build` directory:
```zsh
./test.o
```

### Documentation

The full API documentation generated with Doxygen is available here:

👉 [**Open the documentation**](https://amsc-25-26.github.io/lfbgs-1-lbfgs/)