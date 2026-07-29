# mojo-ortools

`mojo-ortools` ports the compute-bound propagation and routing loops from the
OR-Tools problem domain to Mojo. It provides a finite-domain CP-SAT-style model
and solver, direct NumPy routing functions, and an
`ortools.constraint_solver.pywrapcp`-shaped routing API.

This is a focused port, not a replacement for all of OR-Tools. The Python
package is named `mojo_ortools` so it can be installed beside upstream
`ortools`; for the covered classes, method names and call patterns follow the
upstream Python API.

## Covered subset

CP modeling and solving:

- bounded integer and Boolean variables;
- sparse signed linear equalities and inequalities;
- Boolean OR/AND, implication, at-most-one, exactly-one;
- all-different and allowed/forbidden table constraints;
- linear minimization and maximization;
- exact finite-domain branch-and-bound with interval propagation;
- a direct CSR `propagate_bounds()` API for batched linear constraints and
  clauses.

Routing:

- `RoutingIndexManager`, `RoutingModel`, transit and unary callbacks;
- one shared arc-cost evaluator, fixed vehicle costs, and capacity dimensions;
- one or many vehicles, common or distinct start/end nodes;
- path-cheapest-arc and parallel cheapest-insertion construction;
- directed intra-route 2-opt and cross-route relocate local search;
- direct `construct_routes()`, `improve_routes()`, `two_opt()`, and
  `solve_tsp()` NumPy APIs.

The CP solver is deliberately smaller than CP-SAT. It does not implement
learned clauses, reified constraints, interval variables, no-overlap,
cumulative scheduling, assumptions, solution hints, parallel search, or
floating-point expressions. Interval propagation is bound-consistent, not a
global proof of domain consistency. Routing does not cover time windows,
pickup-and-delivery, disjunction penalties, heterogeneous cost evaluators,
or OR-Tools' metaheuristics. `OnlyEnforceIf` raises `NotImplementedError`
instead of silently changing semantics.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The build task compiles the single Mojo compilation unit to
`dist/libmojo-ortools.so`. Upstream `ortools` 9.15.6755 is installed in the
Pixi environment for parity tests and benchmarks.

## Usage

The CP API uses the same model-building shape as current OR-Tools:

```python
from mojo_ortools.sat.python import cp_model

model = cp_model.CpModel()
x = model.new_int_var(0, 10, "x")
y = model.new_int_var(0, 10, "y")
model.add(2 * x + y == 7)
model.maximize(x)

solver = cp_model.CpSolver()
status = solver.solve(model)
print(solver.status_name(status), solver.value(x), solver.value(y))
# OPTIMAL 3 1
```

The direct routing API accepts a dense contiguous `int64` cost matrix:

```python
import numpy as np
from mojo_ortools.routing import solve_tsp

cost = np.array([
    [0, 1, 2, 1],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [1, 2, 1, 0],
], dtype=np.int64)

solution = solve_tsp(cost)
print(solution.routes[0].tolist(), solution.objective)
# [0, 3, 2, 1, 0] 4
```

Existing routing examples can instead change their imports:

```python
from mojo_ortools.constraint_solver import pywrapcp, routing_enums_pb2
```

`RoutingIndexManager`, callback registration, capacity dimensions,
`SolveWithParameters`, `NextVar`, and `Assignment.Value` then follow the
upstream traversal pattern.

## Benchmarks

Measured by the final `pixi run bench` gate on an Intel Xeon E5-2697 v4 at
2.30 GHz, Linux x86_64. Each workload is warmed up; the benchmark reports the
best timed repeat. Routing rows exercise the same public callback/model-building
shape for mojo-ortools and OR-Tools. The propagation-kernel row intentionally
compares the scoped fixed-point operation with a complete upstream
solve/presolve; the following row includes mojo-ortools' Python model overhead.

| workload | Mojo | reference | speedup | comparison |
| --- | ---: | ---: | ---: | --- |
| 20k-variable propagation kernel | 18.211 ms | 28.186 ms | 1.55x | OR-Tools solve/presolve |
| 20k-variable `CpSolver` API | 32.401 ms | 28.186 ms | 0.87x | OR-Tools CP-SAT |
| 300-node metric TSP | 80.697 ms | 3589.640 ms | 44.48x | OR-Tools `RoutingModel` |
| 160-customer CVRP | 32.071 ms | 1200.546 ms | 37.43x | OR-Tools `RoutingModel`; cost 1.006x upstream |
| 100-node directed 2-opt, 3 passes | 0.097 ms | 1112.688 ms | 11514.82x | pure Python |

The kernel is faster than the upstream solve on this chain model, but the full
Python `CpSolver` wrapper is slower. Model construction maintains contiguous CSR
buffers incrementally, so solving exposes zero-copy NumPy views instead of
serializing Python expression objects. Final linear-assignment validation uses
SIMD with a scalar tail. The routing heuristics are much faster on these
instances, but they are heuristics rather than OR-Tools' full routing search.
The CVRP quality ratio is included in the table rather than hidden.

No GPU path or parallel CPU kernel is provided.

## How it works

`src/ortools.mojo` is one compilation unit containing all propagation,
construction, and local-search kernels. Python owns every allocation.
NumPy arrays cross the C ABI through `ctypes` as integer addresses, and Mojo
reconstructs `UnsafePointer[Int64, AnyOrigin[mut=True]]` values inside
non-parametric `@export` wrappers. There is no allocator or object lifetime
shared across the boundary.

Linear constraints use CSR arrays: coefficients and variable indices are
contiguous, while an offsets array marks each constraint. Variable domains
are parallel `int64` lower/upper arrays. The kernel alternates linear activity
reasoning with Boolean unit propagation until neither can tighten a bound.
The Python solver branches only when that fixed point is not a complete
assignment.

Routing costs are row-major `int64[n, n]`. Routes occupy fixed
`vehicles * (nodes + 2)` caller-owned storage, with separate length and load
arrays. Construction performs capacity-aware insertion; local search mutates
those buffers in place with exact directed-cost deltas.

## License

MIT
