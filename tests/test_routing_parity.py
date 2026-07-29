import numpy as np
import pytest

from ortools.constraint_solver import pywrapcp as upstream_pywrapcp
from ortools.constraint_solver import routing_enums_pb2 as upstream_enums

from mojo_ortools import routing
from mojo_ortools.constraint_solver import pywrapcp, routing_enums_pb2


def _upstream_objective(cost, vehicles=1, demands=None, capacities=None):
    manager = upstream_pywrapcp.RoutingIndexManager(len(cost), vehicles, 0)
    model = upstream_pywrapcp.RoutingModel(manager)
    transit = model.RegisterTransitCallback(
        lambda i, j: int(cost[manager.IndexToNode(i), manager.IndexToNode(j)])
    )
    model.SetArcCostEvaluatorOfAllVehicles(transit)
    if demands is not None:
        demand = model.RegisterUnaryTransitCallback(
            lambda i: int(demands[manager.IndexToNode(i)])
        )
        model.AddDimensionWithVehicleCapacity(demand, 0, capacities, True, "Capacity")
    parameters = upstream_pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        upstream_enums.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    solution = model.SolveWithParameters(parameters)
    assert solution is not None
    return solution.ObjectiveValue()


def test_tsp_objective_parity_on_metric_instance():
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    cost = np.rint(
        np.linalg.norm(points[:, None] - points[None, :], axis=2) * 1000
    ).astype(np.int64)
    ours = routing.solve_tsp(cost)
    assert ours.objective == _upstream_objective(cost)
    assert ours.routes[0][0] == ours.routes[0][-1] == 0
    assert sorted(ours.routes[0][:-1]) == list(range(20))


def test_cvrp_objective_and_capacity_parity():
    points = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [0, 2], [0, 3]],
        dtype=float,
    )
    cost = np.rint(
        np.linalg.norm(points[:, None] - points[None, :], axis=2) * 100
    ).astype(np.int64)
    demands = np.array([0, 2, 2, 2, 2, 2, 2], dtype=np.int64)
    capacities = [6, 6]
    initial = routing.construct_routes(
        cost,
        demands=demands,
        vehicle_capacities=capacities,
        starts=[0, 0],
    )
    ours = routing.improve_routes(
        cost,
        initial,
        demands=demands,
        vehicle_capacities=capacities,
    )
    assert ours.objective == _upstream_objective(cost, 2, demands, capacities) == 1200
    assert ours.loads.tolist() == capacities
    assert sorted(np.concatenate([route[1:-1] for route in ours.routes])) == list(
        range(1, 7)
    )


def test_directed_two_opt_never_worsens_and_cost_is_exact():
    cost = np.array(
        [
            [0, 8, 4, 9, 7],
            [3, 0, 6, 2, 8],
            [5, 7, 0, 6, 3],
            [4, 5, 9, 0, 2],
            [6, 3, 5, 7, 0],
        ],
        dtype=np.int64,
    )
    route = np.array([0, 1, 2, 3, 4, 0])
    before = sum(cost[a, b] for a, b in zip(route[:-1], route[1:]))
    improved, objective = routing.two_opt(cost, route)
    expected = sum(cost[a, b] for a, b in zip(improved[:-1], improved[1:]))
    assert objective == expected == routing.route_cost(cost, improved)
    assert objective <= before


def test_pywrapcp_drop_in_route_traversal_and_dimension():
    cost = np.array(
        [[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]],
        dtype=np.int64,
    )
    demands = [0, 1, 1, 1]
    manager = pywrapcp.RoutingIndexManager(4, 1, 0)
    model = pywrapcp.RoutingModel(manager)
    transit = model.RegisterTransitCallback(
        lambda i, j: int(cost[manager.IndexToNode(i), manager.IndexToNode(j)])
    )
    model.SetArcCostEvaluatorOfAllVehicles(transit)
    demand = model.RegisterUnaryTransitCallback(
        lambda i: demands[manager.IndexToNode(i)]
    )
    model.AddDimensionWithVehicleCapacity(demand, 0, [3], True, "Capacity")
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    assignment = model.SolveWithParameters(parameters)
    assert assignment is not None

    index = model.Start(0)
    route = []
    loads = []
    dimension = model.GetDimensionOrDie("Capacity")
    while not model.IsEnd(index):
        route.append(manager.IndexToNode(index))
        loads.append(assignment.Value(dimension.CumulVar(index)))
        index = assignment.Value(model.NextVar(index))
    route.append(manager.IndexToNode(index))
    assert sorted(route[:-1]) == [0, 1, 2, 3]
    assert route[-1] == 0
    assert loads[0] == 0
    assert assignment.ObjectiveValue() == routing.route_cost(cost, route)


def test_direct_routing_rejects_unsafe_inputs_before_ffi():
    cost = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        routing.route_cost(cost, [0, 3])
    with pytest.raises(ValueError, match="nonempty"):
        routing.construct_routes(cost, starts=[])
    with pytest.raises(TypeError, match="integer"):
        routing.two_opt(cost, [0.0, 1.0, 0.0])


def test_distinct_terminals_and_fixed_vehicle_costs():
    positions = np.arange(6)
    cost = np.abs(positions[:, None] - positions[None, :]).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(6, 2, [0, 5], [1, 4])
    model = pywrapcp.RoutingModel(manager)
    transit = model.RegisterTransitCallback(
        lambda i, j: int(cost[manager.IndexToNode(i), manager.IndexToNode(j)])
    )
    model.SetArcCostEvaluatorOfAllVehicles(transit)
    demand = model.RegisterUnaryTransitCallback(
        lambda index: int(manager.IndexToNode(index) in (2, 3))
    )
    model.AddDimensionWithVehicleCapacity(demand, 0, [1, 1], True, "Capacity")
    model.SetFixedCostOfAllVehicles(7)
    assignment = model.SolveWithParameters(pywrapcp.DefaultRoutingSearchParameters())
    assert assignment is not None
    arc_objective = 0
    for vehicle in range(2):
        index = model.Start(vehicle)
        visited = [manager.IndexToNode(index)]
        while not model.IsEnd(index):
            index = assignment.Value(model.NextVar(index))
            visited.append(manager.IndexToNode(index))
        assert visited[0] == manager.starts[vehicle]
        assert visited[-1] == manager.ends[vehicle]
        arc_objective += routing.route_cost(cost, visited)
    assert assignment.ObjectiveValue() == arc_objective + 14
