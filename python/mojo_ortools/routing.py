"""Direct NumPy APIs for Mojo routing construction and local search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import addr, i64, lib


@dataclass(frozen=True)
class RoutingSolution:
    routes: tuple[np.ndarray, ...]
    loads: np.ndarray
    objective: int


def _cost_matrix(cost_matrix) -> np.ndarray:
    cost = i64(cost_matrix)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost_matrix must be square")
    if np.any(cost < 0):
        raise ValueError("arc costs must be nonnegative")
    if cost.size and int(cost.max()) * max(1, len(cost) + 1) > np.iinfo(np.int64).max:
        raise OverflowError("route objective could overflow int64")
    return cost


def _path(route, nodes: int, *, name: str = "route") -> np.ndarray:
    path = i64(route)
    if path.ndim != 1 or len(path) < 2:
        raise ValueError(f"{name} must contain at least two nodes")
    if np.any(path < 0) or np.any(path >= nodes):
        raise ValueError(f"{name} contains a node outside the cost matrix")
    return path


def _iterations(value: int) -> int:
    value = int(value)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError("max_iterations must be a nonnegative int64")
    return value


def construct_routes(
    cost_matrix,
    *,
    demands=None,
    vehicle_capacities=None,
    starts=None,
    ends=None,
    strategy: str = "parallel_cheapest_insertion",
) -> RoutingSolution:
    """Build a TSP or capacitated multi-vehicle solution.

    ``path_cheapest_arc`` appends at route tails. The default evaluates every
    feasible insertion in every route and chooses the least added cost.
    """

    cost = _cost_matrix(cost_matrix)
    nodes = cost.shape[0]
    if starts is None:
        starts = [0]
    starts_array = i64(starts)
    vehicles = len(starts_array)
    if starts_array.ndim != 1 or vehicles == 0:
        raise ValueError("starts must be a nonempty one-dimensional sequence")
    ends_array = i64(starts if ends is None else ends)
    if ends_array.shape != starts_array.shape:
        raise ValueError("starts and ends must contain one node per vehicle")
    if cost.size and int(cost.max()) * (nodes + vehicles) > np.iinfo(np.int64).max:
        raise OverflowError("total routing objective could overflow int64")
    if np.any(starts_array < 0) or np.any(starts_array >= nodes):
        raise ValueError("start node outside cost matrix")
    if np.any(ends_array < 0) or np.any(ends_array >= nodes):
        raise ValueError("end node outside cost matrix")

    demand = i64(np.zeros(nodes, dtype=np.int64) if demands is None else demands)
    if demand.shape != (nodes,) or np.any(demand < 0):
        raise ValueError("demands must be a nonnegative vector with one entry per node")
    if vehicle_capacities is None:
        vehicle_capacities = [int(demand.sum()) + 1] * vehicles
    capacity = i64(vehicle_capacities)
    if capacity.shape != (vehicles,) or np.any(capacity < 0):
        raise ValueError("vehicle_capacities must contain one nonnegative capacity per vehicle")

    strategy_code = {
        "path_cheapest_arc": 0,
        "parallel_cheapest_insertion": 1,
        "savings": 1,
    }.get(strategy.lower())
    if strategy_code is None:
        raise ValueError(f"unsupported routing strategy: {strategy}")

    storage = np.full((vehicles, nodes + 2), -1, dtype=np.int64)
    lengths = np.zeros(vehicles, dtype=np.int64)
    loads = np.zeros(vehicles, dtype=np.int64)
    visited = np.zeros(nodes, dtype=np.int64)
    objective = int(
        lib().mot_construct_routes(
            addr(cost),
            addr(demand),
            addr(capacity),
            addr(starts_array),
            addr(ends_array),
            addr(storage),
            addr(lengths),
            addr(loads),
            addr(visited),
            nodes,
            vehicles,
            strategy_code,
        )
    )
    if objective < 0:
        raise ValueError("no feasible greedy insertion remained; check vehicle capacities")
    routes = tuple(storage[v, : int(lengths[v])].copy() for v in range(vehicles))
    return RoutingSolution(routes, loads.copy(), objective)


def route_cost(cost_matrix, route) -> int:
    cost = _cost_matrix(cost_matrix)
    path = _path(route, len(cost))
    return int(lib().mot_route_cost(addr(cost), addr(path), len(cost), len(path)))


def two_opt(cost_matrix, route, *, max_iterations: int = 100) -> tuple[np.ndarray, int]:
    """Best-improvement directed 2-opt; endpoints remain fixed."""

    cost = _cost_matrix(cost_matrix)
    path = _path(route, len(cost)).copy()
    max_iterations = _iterations(max_iterations)
    objective = int(
        lib().mot_two_opt(
            addr(cost), addr(path), len(cost), len(path), max_iterations
        )
    )
    return path, objective


def improve_routes(
    cost_matrix,
    solution: RoutingSolution,
    *,
    demands=None,
    vehicle_capacities=None,
    max_iterations: int = 100,
) -> RoutingSolution:
    """Run intra-route 2-opt and cross-route relocate to local optimality."""

    cost = _cost_matrix(cost_matrix)
    nodes = len(cost)
    vehicles = len(solution.routes)
    if vehicles == 0:
        raise ValueError("solution must contain at least one route")
    max_iterations = _iterations(max_iterations)
    if cost.size and int(cost.max()) * vehicles * (nodes + 1) > np.iinfo(np.int64).max:
        raise OverflowError("total routing objective could overflow int64")
    demand = i64(np.zeros(nodes, dtype=np.int64) if demands is None else demands)
    capacity = i64(
        [int(demand.sum()) + 1] * vehicles
        if vehicle_capacities is None
        else vehicle_capacities
    )
    if demand.shape != (nodes,) or np.any(demand < 0):
        raise ValueError("demands must be a nonnegative vector with one entry per node")
    if capacity.shape != (vehicles,) or np.any(capacity < 0):
        raise ValueError("vehicle_capacities must contain one nonnegative capacity per vehicle")
    if np.asarray(solution.loads).shape != (vehicles,):
        raise ValueError("solution loads must contain one entry per route")
    storage = np.full((vehicles, nodes + 2), -1, dtype=np.int64)
    lengths = np.zeros(vehicles, dtype=np.int64)
    for v, route in enumerate(solution.routes):
        path = _path(route, nodes, name=f"route {v}")
        if len(path) > nodes + 2:
            raise ValueError("route exceeds fixed FFI storage")
        storage[v, : len(path)] = path
        lengths[v] = len(path)
    loads = i64(solution.loads, copy=True)

    for v in range(vehicles):
        lib().mot_two_opt(
            addr(cost),
            int(storage[v].ctypes.data),
            nodes,
            int(lengths[v]),
            max_iterations,
        )
    objective = int(
        lib().mot_relocate(
            addr(cost),
            addr(demand),
            addr(capacity),
            addr(storage),
            addr(lengths),
            addr(loads),
            nodes,
            vehicles,
            max_iterations,
        )
    )
    routes = tuple(storage[v, : int(lengths[v])].copy() for v in range(vehicles))
    return RoutingSolution(routes, loads.copy(), objective)


def solve_tsp(
    cost_matrix,
    *,
    depot: int = 0,
    first_solution_strategy: str = "parallel_cheapest_insertion",
    local_search: bool = True,
) -> RoutingSolution:
    solution = construct_routes(
        cost_matrix, starts=[depot], strategy=first_solution_strategy
    )
    return improve_routes(cost_matrix, solution) if local_search else solution
