"""OR-Tools-shaped routing classes backed by Mojo heuristics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..routing import construct_routes, improve_routes
from . import routing_enums_pb2


class _Duration:
    def __init__(self):
        self.seconds = 0
        self.nanos = 0

    def FromSeconds(self, seconds):
        self.seconds = int(seconds)
        self.nanos = int((float(seconds) - self.seconds) * 1_000_000_000)


class RoutingSearchParameters:
    def __init__(self):
        self.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        self.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        self.time_limit = _Duration()
        self.log_search = False


def DefaultRoutingSearchParameters() -> RoutingSearchParameters:
    return RoutingSearchParameters()


class RoutingIndexManager:
    def __init__(self, num_nodes: int, num_vehicles: int, depot_or_starts, ends=None):
        self.num_nodes = int(num_nodes)
        self.num_vehicles = int(num_vehicles)
        if ends is None:
            if not isinstance(depot_or_starts, (int, np.integer)):
                raise TypeError("the three-argument form requires a single depot")
            self.starts = [int(depot_or_starts)] * self.num_vehicles
            self.ends = self.starts.copy()
        else:
            self.starts = [int(v) for v in depot_or_starts]
            self.ends = [int(v) for v in ends]
        if len(self.starts) != self.num_vehicles or len(self.ends) != self.num_vehicles:
            raise ValueError("starts and ends must contain one node per vehicle")
        if any(node < 0 or node >= self.num_nodes for node in self.starts + self.ends):
            raise ValueError("start or end node outside manager range")
        self._start_indices = [self.num_nodes + 2 * v for v in range(self.num_vehicles)]
        self._end_indices = [self.num_nodes + 2 * v + 1 for v in range(self.num_vehicles)]

    def IndexToNode(self, index: int) -> int:
        index = int(index)
        if index < self.num_nodes:
            return index
        for vehicle, start in enumerate(self._start_indices):
            if index == start:
                return self.starts[vehicle]
            if index == self._end_indices[vehicle]:
                return self.ends[vehicle]
        raise IndexError(index)

    def NodeToIndex(self, node: int) -> int:
        node = int(node)
        if node < 0 or node >= self.num_nodes:
            return -1
        return node

    def GetNumberOfNodes(self) -> int:
        return self.num_nodes

    def GetNumberOfVehicles(self) -> int:
        return self.num_vehicles


@dataclass(frozen=True)
class IntVar:
    kind: str
    index: int
    dimension: str = ""


class RoutingDimension:
    def __init__(self, model: "RoutingModel", name: str):
        self.model = model
        self.name = name

    def CumulVar(self, index: int) -> IntVar:
        return IntVar("cumul", int(index), self.name)

    def SetGlobalSpanCostCoefficient(self, coefficient: int):
        self.model._span_cost_coefficient = int(coefficient)


class Assignment:
    def __init__(self, next_values, cumul_values, objective: int):
        self._next_values = next_values
        self._cumul_values = cumul_values
        self._objective = int(objective)

    def Value(self, variable: IntVar) -> int:
        if variable.kind == "next":
            return self._next_values[variable.index]
        if variable.kind == "cumul":
            return self._cumul_values[(variable.dimension, variable.index)]
        if variable.kind == "cost":
            return self._objective
        raise KeyError(variable)

    def ObjectiveValue(self) -> int:
        return self._objective

    def Min(self, variable: IntVar) -> int:
        return self.Value(variable)

    def Max(self, variable: IntVar) -> int:
        return self.Value(variable)


class RoutingModel:
    ROUTING_NOT_SOLVED = 0
    ROUTING_SUCCESS = 1
    ROUTING_FAIL = 3

    def __init__(self, manager: RoutingIndexManager):
        self.manager = manager
        self._callbacks = []
        self._cost_callback: int | None = None
        self._demand_callback: int | None = None
        self._capacities: list[int] | None = None
        self._dimensions: dict[str, RoutingDimension] = {}
        self._fixed_costs = [0] * manager.num_vehicles
        self._span_cost_coefficient = 0
        self._status = self.ROUTING_NOT_SOLVED

    def RegisterTransitCallback(self, callback) -> int:
        self._callbacks.append(callback)
        return len(self._callbacks) - 1

    def RegisterUnaryTransitCallback(self, callback) -> int:
        self._callbacks.append(callback)
        return len(self._callbacks) - 1

    def SetArcCostEvaluatorOfAllVehicles(self, callback_index: int):
        self._cost_callback = int(callback_index)

    def SetArcCostEvaluatorOfVehicle(self, callback_index: int, vehicle: int):
        if vehicle < 0 or vehicle >= self.manager.num_vehicles:
            raise IndexError(vehicle)
        if self._cost_callback is not None and self._cost_callback != callback_index:
            raise NotImplementedError("per-vehicle cost evaluators are outside the covered subset")
        self._cost_callback = int(callback_index)

    def AddDimensionWithVehicleCapacity(
        self,
        evaluator_index: int,
        slack_max: int,
        vehicle_capacities,
        fix_start_cumul_to_zero: bool,
        name: str,
    ) -> bool:
        if slack_max:
            raise NotImplementedError("positive dimension slack is outside the covered subset")
        if not fix_start_cumul_to_zero:
            raise NotImplementedError("nonzero start cumuls are outside the covered subset")
        capacities = [int(value) for value in vehicle_capacities]
        if len(capacities) != self.manager.num_vehicles:
            raise ValueError("one capacity is required per vehicle")
        self._demand_callback = int(evaluator_index)
        self._capacities = capacities
        self._dimensions[name] = RoutingDimension(self, name)
        return True

    def AddDimension(
        self,
        evaluator_index: int,
        slack_max: int,
        capacity: int,
        fix_start_cumul_to_zero: bool,
        name: str,
    ) -> bool:
        return self.AddDimensionWithVehicleCapacity(
            evaluator_index,
            slack_max,
            [capacity] * self.manager.num_vehicles,
            fix_start_cumul_to_zero,
            name,
        )

    def GetDimensionOrDie(self, name: str) -> RoutingDimension:
        return self._dimensions[name]

    def Start(self, vehicle: int) -> int:
        return self.manager._start_indices[vehicle]

    def End(self, vehicle: int) -> int:
        return self.manager._end_indices[vehicle]

    def IsStart(self, index: int) -> bool:
        return int(index) in self.manager._start_indices

    def IsEnd(self, index: int) -> bool:
        return int(index) in self.manager._end_indices

    def NextVar(self, index: int) -> IntVar:
        return IntVar("next", int(index))

    def CostVar(self) -> IntVar:
        return IntVar("cost", 0)

    def Size(self) -> int:
        return self.manager.num_nodes + self.manager.num_vehicles

    def vehicles(self) -> int:
        return self.manager.num_vehicles

    def nodes(self) -> int:
        return self.manager.num_nodes

    def SetFixedCostOfAllVehicles(self, cost: int):
        self._fixed_costs = [int(cost)] * self.manager.num_vehicles

    def SetFixedCostOfVehicle(self, cost: int, vehicle: int):
        self._fixed_costs[vehicle] = int(cost)

    def GetArcCostForVehicle(self, from_index: int, to_index: int, vehicle: int) -> int:
        if self._cost_callback is None:
            return 0
        return int(self._callbacks[self._cost_callback](int(from_index), int(to_index)))

    def status(self) -> int:
        return self._status

    def SolveWithParameters(self, search_parameters: RoutingSearchParameters):
        if self._cost_callback is None:
            raise RuntimeError("an arc cost evaluator must be registered before solving")
        nodes = self.manager.num_nodes
        callback = self._callbacks[self._cost_callback]
        cost = np.empty((nodes, nodes), dtype=np.int64)
        for i in range(nodes):
            for j in range(nodes):
                cost[i, j] = callback(i, j)

        demand = np.zeros(nodes, dtype=np.int64)
        if self._demand_callback is not None:
            demand_callback = self._callbacks[self._demand_callback]
            for node in range(nodes):
                demand[node] = demand_callback(node)
        capacities = self._capacities or [int(demand.sum()) + 1] * self.manager.num_vehicles

        first = search_parameters.first_solution_strategy
        strategy = (
            "path_cheapest_arc"
            if first == routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            else "parallel_cheapest_insertion"
        )
        try:
            solution = construct_routes(
                cost,
                demands=demand,
                vehicle_capacities=capacities,
                starts=self.manager.starts,
                ends=self.manager.ends,
                strategy=strategy,
            )
        except ValueError:
            self._status = self.ROUTING_FAIL
            return None

        if search_parameters.local_search_metaheuristic != routing_enums_pb2.LocalSearchMetaheuristic.UNSET:
            solution = improve_routes(
                cost,
                solution,
                demands=demand,
                vehicle_capacities=capacities,
            )

        next_values: dict[int, int] = {}
        cumul_values: dict[tuple[str, int], int] = {}
        for vehicle, route in enumerate(solution.routes):
            internal = [
                self.Start(vehicle),
                *[int(node) for node in route[1:-1]],
                self.End(vehicle),
            ]
            load = 0
            for position, index in enumerate(internal):
                for name in self._dimensions:
                    cumul_values[(name, index)] = load
                if position + 1 < len(internal):
                    next_values[index] = internal[position + 1]
                    load += int(demand[self.manager.IndexToNode(index)])
            next_values[internal[-1]] = internal[-1]

        used = sum(len(route) > 2 for route in solution.routes)
        objective = solution.objective + sum(
            fixed for fixed, route in zip(self._fixed_costs, solution.routes) if len(route) > 2
        )
        if self._span_cost_coefficient:
            objective += self._span_cost_coefficient * int(max(solution.loads))
        _ = used
        self._status = self.ROUTING_SUCCESS
        return Assignment(next_values, cumul_values, objective)

    def Solve(self):
        return self.SolveWithParameters(DefaultRoutingSearchParameters())

    def CloseModel(self):
        return None

    def CloseModelWithParameters(self, search_parameters):
        return None
