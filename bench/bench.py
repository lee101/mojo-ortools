"""mojo-ortools against upstream OR-Tools on the same models."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_ortools import routing  # noqa: E402
from mojo_ortools.constraint_solver import pywrapcp as mojo_pywrapcp  # noqa: E402
from mojo_ortools.constraint_solver import routing_enums_pb2 as mojo_enums  # noqa: E402
from mojo_ortools.propagation import propagate_bounds  # noqa: E402
from mojo_ortools.sat.python import cp_model as mojo_cp_model  # noqa: E402
from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402


def best_time(function, repeats=3):
    function()
    best = float("inf")
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - started)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def chain_benchmark(size=20_000):
    coefficients = np.ones(size * 2 - 1, dtype=np.int64)
    variables = np.empty(size * 2 - 1, dtype=np.int64)
    offsets = np.empty(size + 1, dtype=np.int64)
    lower = np.full(size, 100, dtype=np.int64)
    upper = lower.copy()
    variables[0] = 0
    offsets[0] = 0
    offsets[1] = 1
    cursor = 1
    for i in range(size - 1):
        variables[cursor : cursor + 2] = (i, i + 1)
        cursor += 2
        offsets[i + 2] = cursor
    lower[0] = upper[0] = 17
    variable_lower = np.zeros(size, dtype=np.int64)
    variable_upper = np.full(size, 100, dtype=np.int64)

    def mojo_run():
        return propagate_bounds(
            coefficients,
            variables,
            offsets,
            lower,
            upper,
            variable_lower_bounds=variable_lower,
            variable_upper_bounds=variable_upper,
        )

    model = cp_model.CpModel()
    x = [model.new_int_var(0, 100, f"x{i}") for i in range(size)]
    model.add(x[0] == 17)
    for i in range(size - 1):
        model.add(x[i] + x[i + 1] == 100)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1

    def upstream_run():
        status = solver.solve(model)
        return status, solver.value(x[-1])

    mojo_seconds, mojo_result = best_time(mojo_run)
    upstream_seconds, upstream_result = best_time(upstream_run)
    assert mojo_result.feasible
    assert mojo_result.lower_bounds[-1] == upstream_result[1]

    mojo_model = mojo_cp_model.CpModel()
    mojo_x = [mojo_model.new_int_var(0, 100, f"x{i}") for i in range(size)]
    mojo_model.add(mojo_x[0] == 17)
    for i in range(size - 1):
        mojo_model.add(mojo_x[i] + mojo_x[i + 1] == 100)
    mojo_solver = mojo_cp_model.CpSolver()

    def mojo_api_run():
        status = mojo_solver.solve(mojo_model)
        return status, mojo_solver.value(mojo_x[-1])

    mojo_api_seconds, mojo_api_result = best_time(mojo_api_run)
    assert mojo_api_result[1] == upstream_result[1]
    return mojo_seconds, mojo_api_seconds, upstream_seconds


def make_circle(nodes):
    angles = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return np.rint(
        np.linalg.norm(points[:, None] - points[None, :], axis=2) * 1_000_000
    ).astype(np.int64)


def routing_tsp(cost, wrapper, enums):
    manager = wrapper.RoutingIndexManager(len(cost), 1, 0)
    model = wrapper.RoutingModel(manager)
    callback = model.RegisterTransitCallback(
        lambda i, j: int(cost[manager.IndexToNode(i), manager.IndexToNode(j)])
    )
    model.SetArcCostEvaluatorOfAllVehicles(callback)
    parameters = wrapper.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        enums.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = model.SolveWithParameters(parameters)
    return solution.ObjectiveValue()


def tsp_benchmark(nodes=300):
    cost = make_circle(nodes)
    mojo_seconds, mojo_result = best_time(
        lambda: routing_tsp(cost, mojo_pywrapcp, mojo_enums)
    )
    upstream_seconds, upstream_result = best_time(
        lambda: routing_tsp(cost, pywrapcp, routing_enums_pb2)
    )
    assert mojo_result == upstream_result
    return mojo_seconds, upstream_seconds


def clustered_cvrp(clusters=8, per_cluster=20):
    rng = np.random.default_rng(7)
    centers = np.column_stack(
        (
            20 * np.cos(np.linspace(0, 2 * np.pi, clusters, endpoint=False)),
            20 * np.sin(np.linspace(0, 2 * np.pi, clusters, endpoint=False)),
        )
    )
    points = np.vstack(([0.0, 0.0], *(center + rng.normal(0, 1, (per_cluster, 2)) for center in centers)))
    cost = np.rint(
        np.linalg.norm(points[:, None] - points[None, :], axis=2) * 1000
    ).astype(np.int64)
    demands = np.r_[0, np.ones(clusters * per_cluster, dtype=np.int64)]
    capacities = [per_cluster] * clusters
    return cost, demands, capacities


def wrapper_cvrp(cost, demands, capacities, wrapper, enums):
    vehicles = len(capacities)
    manager = wrapper.RoutingIndexManager(len(cost), vehicles, 0)
    model = wrapper.RoutingModel(manager)
    transit = model.RegisterTransitCallback(
        lambda i, j: int(cost[manager.IndexToNode(i), manager.IndexToNode(j)])
    )
    model.SetArcCostEvaluatorOfAllVehicles(transit)
    demand = model.RegisterUnaryTransitCallback(
        lambda i: int(demands[manager.IndexToNode(i)])
    )
    model.AddDimensionWithVehicleCapacity(demand, 0, capacities, True, "Capacity")
    parameters = wrapper.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        enums.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    solution = model.SolveWithParameters(parameters)
    return solution.ObjectiveValue()


def cvrp_benchmark():
    cost, demands, capacities = clustered_cvrp()
    mojo_seconds, mojo_result = best_time(
        lambda: wrapper_cvrp(
            cost, demands, capacities, mojo_pywrapcp, mojo_enums
        )
    )
    upstream_seconds, upstream_result = best_time(
        lambda: wrapper_cvrp(
            cost, demands, capacities, pywrapcp, routing_enums_pb2
        )
    )
    quality = mojo_result / upstream_result
    return mojo_seconds, upstream_seconds, quality


def two_opt_benchmark(nodes=100):
    rng = np.random.default_rng(11)
    points = rng.normal(size=(nodes, 2))
    cost = np.rint(
        np.linalg.norm(points[:, None] - points[None, :], axis=2) * 1_000_000
    ).astype(np.int64)
    route = np.r_[0, rng.permutation(np.arange(1, nodes)), 0].astype(np.int64)
    mojo_seconds, mojo_result = best_time(
        lambda: routing.two_opt(cost, route, max_iterations=3), repeats=5
    )
    python_seconds, python_result = best_time(
        lambda: python_two_opt_limited(cost, route, 3), repeats=2
    )
    python_cost = sum(cost[a, b] for a, b in zip(python_result[:-1], python_result[1:]))
    assert mojo_result[1] == python_cost
    return mojo_seconds, python_seconds


def python_two_opt_limited(cost, route, passes):
    route = route.copy()
    for _ in range(passes):
        current = sum(cost[a, b] for a, b in zip(route[:-1], route[1:]))
        best_cost = current
        best = None
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                candidate = route.copy()
                candidate[i : j + 1] = candidate[i : j + 1][::-1]
                candidate_cost = sum(
                    cost[a, b] for a, b in zip(candidate[:-1], candidate[1:])
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best = candidate
        if best is None:
            break
        route = best
    return route


def row(name, mojo_seconds, reference_seconds, reference, note=""):
    speedup = reference_seconds / mojo_seconds
    print(
        f"| {name} | {mojo_seconds * 1000:.3f} ms | "
        f"{reference_seconds * 1000:.3f} ms | {speedup:.2f}x | {reference}{note} |"
    )


def main():
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print()
    print("| workload | Mojo | reference | speedup | comparison |")
    print("| --- | ---: | ---: | ---: | --- |")
    kernel_seconds, mojo_seconds, reference_seconds = chain_benchmark()
    row(
        "20k-variable propagation kernel",
        kernel_seconds,
        reference_seconds,
        "OR-Tools solve/presolve",
    )
    row("20k-variable CpSolver API", mojo_seconds, reference_seconds, "OR-Tools CP-SAT")
    mojo_seconds, reference_seconds = tsp_benchmark()
    row("300-node metric TSP", mojo_seconds, reference_seconds, "OR-Tools RoutingModel")
    mojo_seconds, reference_seconds, quality = cvrp_benchmark()
    row(
        "160-customer CVRP",
        mojo_seconds,
        reference_seconds,
        "OR-Tools RoutingModel",
        f"; cost {quality:.3f}x upstream",
    )
    mojo_seconds, reference_seconds = two_opt_benchmark()
    row("100-node directed 2-opt, 3 passes", mojo_seconds, reference_seconds, "pure Python")


if __name__ == "__main__":
    main()
