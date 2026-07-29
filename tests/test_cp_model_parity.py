import numpy as np
import pytest

from ortools.sat.python import cp_model as upstream
from mojo_ortools.sat.python import cp_model as mojo


def _solve_knapsack(module):
    weights = [12, 7, 11, 8, 9, 6, 5, 14, 3, 10, 4, 13]
    profits = [24, 13, 23, 15, 16, 11, 9, 28, 6, 19, 8, 25]
    model = module.CpModel()
    selected = [model.new_bool_var(f"x{i}") for i in range(len(weights))]
    model.add(sum(w * x for w, x in zip(weights, selected)) <= 45)
    model.maximize(sum(p * x for p, x in zip(profits, selected)))
    solver = module.CpSolver()
    status = solver.solve(model)
    return solver.status_name(status), int(solver.objective_value), [
        solver.value(x) for x in selected
    ]


def test_knapsack_objective_and_solution_parity():
    ours = _solve_knapsack(mojo)
    theirs = _solve_knapsack(upstream)
    assert ours[0] == theirs[0] == "OPTIMAL"
    assert ours[1] == theirs[1]
    assert sum(
        w * value
        for w, value in zip([12, 7, 11, 8, 9, 6, 5, 14, 3, 10, 4, 13], ours[2])
    ) <= 45


def _solve_assignment(module):
    costs = np.array(
        [
            [90, 76, 75, 70],
            [35, 85, 55, 65],
            [125, 95, 90, 105],
            [45, 110, 95, 115],
        ]
    )
    model = module.CpModel()
    x = [[model.new_bool_var(f"x_{i}_{j}") for j in range(4)] for i in range(4)]
    for row in x:
        model.add_exactly_one(row)
    for j in range(4):
        model.add_exactly_one([x[i][j] for i in range(4)])
    model.minimize(sum(int(costs[i, j]) * x[i][j] for i in range(4) for j in range(4)))
    solver = module.CpSolver()
    status = solver.solve(model)
    return solver.status_name(status), int(solver.objective_value)


def test_assignment_parity():
    assert _solve_assignment(mojo) == _solve_assignment(upstream) == ("OPTIMAL", 265)


def test_all_different_linear_optimization_parity():
    def solve(module):
        model = module.CpModel()
        x = [model.new_int_var(1, 4, f"x{i}") for i in range(4)]
        model.add_all_different(x)
        model.add(x[0] + 2 * x[1] + x[2] == 9)
        model.minimize(4 * x[0] + x[1] + 3 * x[2] + 2 * x[3])
        solver = module.CpSolver()
        status = solver.solve(model)
        return solver.status_name(status), int(solver.objective_value), solver.values(x).tolist()

    ours = solve(mojo)
    theirs = solve(upstream)
    assert ours[:2] == theirs[:2]
    assert len(set(ours[2])) == 4


def test_boolean_implication_and_infeasibility_parity():
    def solve(module):
        model = module.CpModel()
        a = model.new_bool_var("a")
        b = model.new_bool_var("b")
        model.add_implication(a, b)
        model.add(a == 1)
        model.add(b == 0)
        solver = module.CpSolver()
        return solver.status_name(solver.solve(model))

    assert solve(mojo) == solve(upstream) == "INFEASIBLE"


def test_solution_callback_enumeration():
    model = mojo.CpModel()
    x = model.new_bool_var("x")
    y = model.new_bool_var("y")
    model.add_bool_or([x, y])

    class Counter(mojo.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_solution_callback(self):
            self.rows.append((self.value(x), self.value(y)))

    callback = Counter()
    solver = mojo.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    assert solver.solve(model, callback) == mojo.OPTIMAL
    assert set(callback.rows) == {(0, 1), (1, 0), (1, 1)}


def test_current_upstream_style_properties():
    model = mojo.CpModel()
    x = model.new_int_var(2, 2, "x")
    solver = mojo.CpSolver()
    assert solver.solve(model) == mojo.OPTIMAL
    assert solver.value(x) == 2
    assert solver.objective_value == 0.0
    assert solver.num_branches == 0
    assert "status: OPTIMAL" in solver.response_stats()


def test_model_can_extend_and_resolve_after_contiguous_views_are_released():
    model = mojo.CpModel()
    x = model.new_int_var(2, 2, "x")
    solver = mojo.CpSolver()
    assert solver.solve(model) == mojo.OPTIMAL

    y = model.new_int_var(3, 3, "y")
    model.add(x + y == 5)
    assert solver.solve(model) == mojo.OPTIMAL
    assert solver.values([x, y]).tolist() == [2, 3]


def test_documented_boolean_and_table_constraints():
    model = mojo.CpModel()
    a, b, c = [model.new_bool_var(name) for name in "abc"]
    model.add_bool_and([a, b])
    model.add_bool_or([c])
    model.add_at_most_one([b, c])
    model.add_allowed_assignments([a, b], [(1, 1)])
    model.add_forbidden_assignments([b, c], [(0, 0)])
    solver = mojo.CpSolver()
    assert solver.status_name(solver.solve(model)) == "INFEASIBLE"
