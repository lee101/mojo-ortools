"""A finite-domain CP-SAT-style subset driven by Mojo propagation kernels."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import time
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from ...propagation import (
    INF,
    PropagationResult,
    propagate_bounds,
    validate_assignment,
)

UNKNOWN = 0
MODEL_INVALID = 1
FEASIBLE = 2
INFEASIBLE = 3
OPTIMAL = 4


def _merge(left: dict[int, int], right: dict[int, int], scale: int = 1):
    result = left.copy()
    for variable, coefficient in right.items():
        result[variable] = result.get(variable, 0) + scale * coefficient
        if result[variable] == 0:
            del result[variable]
    return result


class LinearExpr:
    def __init__(self, terms: dict[int, int] | None = None, offset: int = 0):
        self.terms = terms or {}
        self.offset = int(offset)

    @staticmethod
    def sum(expressions: Iterable) -> "LinearExpr":
        result = LinearExpr()
        for expression in expressions:
            result += expression
        return result

    @staticmethod
    def weighted_sum(expressions: Iterable, coefficients: Iterable[int]) -> "LinearExpr":
        result = LinearExpr()
        for expression, coefficient in zip(expressions, coefficients, strict=True):
            result += expression * int(coefficient)
        return result

    Sum = sum
    WeightedSum = weighted_sum

    def __add__(self, other):
        other = _as_expr(other)
        return LinearExpr(_merge(self.terms, other.terms), self.offset + other.offset)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = _as_expr(other)
        return LinearExpr(
            _merge(self.terms, other.terms, -1), self.offset - other.offset
        )

    def __rsub__(self, other):
        return _as_expr(other) - self

    def __mul__(self, coefficient: int):
        coefficient = int(coefficient)
        return LinearExpr(
            {v: coefficient * c for v, c in self.terms.items()},
            coefficient * self.offset,
        )

    def __rmul__(self, coefficient: int):
        return self * coefficient

    def __neg__(self):
        return self * -1

    def __le__(self, other):
        return BoundedLinearExpression(self - other, -int(INF), 0)

    def __ge__(self, other):
        return BoundedLinearExpression(self - other, 0, int(INF))

    def __eq__(self, other):
        return BoundedLinearExpression(self - other, 0, 0)

    def __ne__(self, other):
        return NotEqualExpression(self - other)

    def __bool__(self):
        raise TypeError("a linear expression cannot be used as a Boolean")


def _as_expr(value) -> LinearExpr:
    if isinstance(value, LinearExpr):
        return value
    if isinstance(value, (int, np.integer)):
        return LinearExpr(offset=int(value))
    raise TypeError(f"expected a linear expression or integer, got {type(value).__name__}")


def _expand_variadic(values):
    if len(values) == 1 and not isinstance(
        values[0], (LinearExpr, NotBooleanVariable, int, np.integer, bool, np.bool_)
    ):
        return tuple(values[0])
    return tuple(values)


class IntVar(LinearExpr):
    def __init__(self, index: int, lb: int, ub: int, name: str, is_boolean: bool):
        super().__init__({index: 1})
        self.index = index
        self.lb = int(lb)
        self.ub = int(ub)
        self.name = name
        self.is_boolean = is_boolean

    __hash__ = object.__hash__

    def not_(self):
        if not self.is_boolean:
            raise TypeError("not_() is only valid for Boolean variables")
        return NotBooleanVariable(self)

    Not = not_

    def Name(self) -> str:
        return self.name

    def Index(self) -> int:
        return self.index

    def __repr__(self):
        return self.name or f"var_{self.index}"


class NotBooleanVariable:
    def __init__(self, variable: IntVar):
        self.variable = variable

    def not_(self):
        return self.variable

    Not = not_

    def Index(self) -> int:
        return -self.variable.index - 1

    def __repr__(self):
        return f"not({self.variable!r})"


@dataclass(frozen=True)
class BoundedLinearExpression:
    expression: LinearExpr
    lower_bound: int
    upper_bound: int


@dataclass(frozen=True)
class NotEqualExpression:
    expression: LinearExpr


class Constraint:
    def only_enforce_if(self, *literals):
        raise NotImplementedError("OnlyEnforceIf is outside the covered subset")

    OnlyEnforceIf = only_enforce_if


class CpModel:
    def __init__(self):
        self.variables: list[IntVar] = []
        self.linear_constraints: list[tuple[dict[int, int], int, int]] = []
        self.clauses: list[list[int]] = []
        self.not_equal: list[LinearExpr] = []
        self.all_different: list[tuple[int, ...]] = []
        self.allowed_tables: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
        self.forbidden_tables: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
        self.objective: LinearExpr | None = None
        self.objective_sense = 1
        self._coefficients = array("q")
        self._linear_variables = array("q")
        self._linear_offsets = array("q", [0])
        self._constraint_lower = array("q")
        self._constraint_upper = array("q")
        self._literals = array("q")
        self._clause_offsets = array("q", [0])
        self._variable_lower = array("q")
        self._variable_upper = array("q")
        self._minimum_linear_variable = 0
        self._maximum_linear_variable = -1
        self._linear_bounds_valid = True

    def new_int_var(self, lb: int, ub: int, name: str) -> IntVar:
        lb, ub = int(lb), int(ub)
        if lb > ub:
            raise ValueError("lower bound exceeds upper bound")
        variable = IntVar(len(self.variables), lb, ub, name, False)
        self.variables.append(variable)
        self._variable_lower.append(lb)
        self._variable_upper.append(ub)
        return variable

    def new_bool_var(self, name: str) -> IntVar:
        variable = IntVar(len(self.variables), 0, 1, name, True)
        self.variables.append(variable)
        self._variable_lower.append(0)
        self._variable_upper.append(1)
        return variable

    def new_constant(self, value: int) -> IntVar:
        return self.new_int_var(value, value, str(value))

    NewIntVar = new_int_var
    NewBoolVar = new_bool_var
    NewConstant = new_constant

    def add(self, expression) -> Constraint:
        if isinstance(expression, (bool, np.bool_)):
            if not expression:
                self.clauses.append([])
                self._clause_offsets.append(len(self._literals))
            return Constraint()
        if isinstance(expression, NotEqualExpression):
            self.not_equal.append(expression.expression)
            return Constraint()
        if not isinstance(expression, BoundedLinearExpression):
            raise TypeError("add() expects a bounded linear expression")
        linear = expression.expression
        terms = linear.terms.copy()
        lower = int(expression.lower_bound) - linear.offset
        upper = int(expression.upper_bound) - linear.offset
        self.linear_constraints.append((terms, lower, upper))
        self._linear_bounds_valid &= lower <= upper
        for variable, coefficient in sorted(terms.items()):
            self._linear_variables.append(variable)
            self._coefficients.append(coefficient)
            self._minimum_linear_variable = min(
                self._minimum_linear_variable, variable
            )
            self._maximum_linear_variable = max(
                self._maximum_linear_variable, variable
            )
        self._linear_offsets.append(len(self._coefficients))
        self._constraint_lower.append(lower)
        self._constraint_upper.append(upper)
        return Constraint()

    Add = add

    def _literal(self, literal) -> int:
        if isinstance(literal, NotBooleanVariable):
            return -(literal.variable.index + 1)
        if isinstance(literal, IntVar) and literal.is_boolean:
            return literal.index + 1
        if isinstance(literal, (bool, np.bool_)):
            return int(INF) if literal else -int(INF)
        raise TypeError("Boolean constraints require Boolean variables or their negations")

    def add_bool_or(self, *literals) -> Constraint:
        literals = _expand_variadic(literals)
        encoded = [self._literal(literal) for literal in literals]
        if int(INF) in encoded:
            return Constraint()
        clause = [literal for literal in encoded if literal != -int(INF)]
        self.clauses.append(clause)
        self._literals.extend(clause)
        self._clause_offsets.append(len(self._literals))
        return Constraint()

    AddBoolOr = add_bool_or

    def add_at_least_one(self, *literals) -> Constraint:
        if len(literals) == 1 and not isinstance(literals[0], (IntVar, NotBooleanVariable)):
            literals = tuple(literals[0])
        return self.add_bool_or(literals)

    AddAtLeastOne = add_at_least_one

    def add_bool_and(self, *literals) -> Constraint:
        literals = _expand_variadic(literals)
        for literal in literals:
            self.add_bool_or([literal])
        return Constraint()

    AddBoolAnd = add_bool_and

    def add_at_most_one(self, *literals) -> Constraint:
        literals = _expand_variadic(literals)
        expressions = [_literal_expr(literal) for literal in literals]
        return self.add(LinearExpr.sum(expressions) <= 1)

    AddAtMostOne = add_at_most_one

    def add_exactly_one(self, *literals) -> Constraint:
        literals = _expand_variadic(literals)
        expressions = [_literal_expr(literal) for literal in literals]
        return self.add(LinearExpr.sum(expressions) == 1)

    AddExactlyOne = add_exactly_one

    def add_all_different(self, *expressions) -> Constraint:
        variables = _expand_variadic(expressions)
        if not all(isinstance(variable, IntVar) for variable in variables):
            raise NotImplementedError("AddAllDifferent currently accepts variables, not expressions")
        self.all_different.append(tuple(variable.index for variable in variables))
        return Constraint()

    AddAllDifferent = add_all_different

    def add_implication(self, left, right) -> Constraint:
        negated = left.not_() if isinstance(left, (IntVar, NotBooleanVariable)) else not left
        return self.add_bool_or([negated, right])

    AddImplication = add_implication

    def add_linear_constraint(self, expression, lb: int, ub: int) -> Constraint:
        linear = _as_expr(expression)
        return self.add(BoundedLinearExpression(linear, int(lb), int(ub)))

    AddLinearConstraint = add_linear_constraint

    def add_allowed_assignments(self, variables: Iterable[IntVar], tuples_list) -> Constraint:
        selected = tuple(variables)
        if not all(isinstance(variable, IntVar) for variable in selected):
            raise NotImplementedError("table constraints currently accept variables")
        rows = tuple(tuple(int(value) for value in row) for row in tuples_list)
        if any(len(row) != len(selected) for row in rows):
            raise ValueError("every tuple must have one value per variable")
        self.allowed_tables.append((tuple(v.index for v in selected), rows))
        return Constraint()

    AddAllowedAssignments = add_allowed_assignments

    def add_forbidden_assignments(self, variables: Iterable[IntVar], tuples_list) -> Constraint:
        selected = tuple(variables)
        if not all(isinstance(variable, IntVar) for variable in selected):
            raise NotImplementedError("table constraints currently accept variables")
        rows = tuple(tuple(int(value) for value in row) for row in tuples_list)
        if any(len(row) != len(selected) for row in rows):
            raise ValueError("every tuple must have one value per variable")
        self.forbidden_tables.append((tuple(v.index for v in selected), rows))
        return Constraint()

    AddForbiddenAssignments = add_forbidden_assignments

    def minimize(self, expression):
        self.objective = _as_expr(expression)
        self.objective_sense = 1

    def maximize(self, expression):
        self.objective = _as_expr(expression)
        self.objective_sense = -1

    Minimize = minimize
    Maximize = maximize

    def clone(self):
        import copy

        return copy.deepcopy(self)

    Clone = clone

    def validate(self) -> str:
        if not self._linear_bounds_valid:
            return "linear constraint has an empty domain"
        if (
            self._minimum_linear_variable < 0
            or self._maximum_linear_variable >= len(self.variables)
        ):
            return "linear constraint references an unknown variable"
        return ""

    Validate = validate

    def _arrays(self):
        return (
            np.frombuffer(self._coefficients, dtype=np.int64),
            np.frombuffer(self._linear_variables, dtype=np.int64),
            np.frombuffer(self._linear_offsets, dtype=np.int64),
            np.frombuffer(self._constraint_lower, dtype=np.int64),
            np.frombuffer(self._constraint_upper, dtype=np.int64),
            np.frombuffer(self._literals, dtype=np.int64),
            np.frombuffer(self._clause_offsets, dtype=np.int64),
        )

    def _initial_bounds(self):
        return (
            np.frombuffer(self._variable_lower, dtype=np.int64).copy(),
            np.frombuffer(self._variable_upper, dtype=np.int64).copy(),
        )


def _literal_expr(literal) -> LinearExpr:
    if isinstance(literal, IntVar) and literal.is_boolean:
        return literal
    if isinstance(literal, NotBooleanVariable):
        return 1 - literal.variable
    if isinstance(literal, (bool, np.bool_)):
        return LinearExpr(offset=int(literal))
    raise TypeError("expected a Boolean literal")


def _expr_bounds(expression: LinearExpr, lower: np.ndarray, upper: np.ndarray):
    minimum = expression.offset
    maximum = expression.offset
    for variable, coefficient in expression.terms.items():
        if coefficient >= 0:
            minimum += coefficient * int(lower[variable])
            maximum += coefficient * int(upper[variable])
        else:
            minimum += coefficient * int(upper[variable])
            maximum += coefficient * int(lower[variable])
    return minimum, maximum


def propagate(
    model: CpModel,
    variable_lower_bounds=None,
    variable_upper_bounds=None,
    *,
    max_rounds: int = 100,
) -> PropagationResult:
    """Return the interval-domain fixed point for the current model."""

    arrays = model._arrays()
    lower = (
        np.frombuffer(model._variable_lower, dtype=np.int64).copy()
        if variable_lower_bounds is None
        else np.array(variable_lower_bounds, dtype=np.int64)
    )
    upper = (
        np.frombuffer(model._variable_upper, dtype=np.int64).copy()
        if variable_upper_bounds is None
        else np.array(variable_upper_bounds, dtype=np.int64)
    )
    total_rounds = 0
    for _ in range(max_rounds):
        result = propagate_bounds(
            *arrays,
            variable_lower_bounds=lower,
            variable_upper_bounds=upper,
            max_rounds=max_rounds,
        )
        total_rounds += result.rounds
        if not result.feasible:
            return PropagationResult(False, result.lower_bounds, result.upper_bounds, total_rounds)
        lower, upper = result.lower_bounds, result.upper_bounds
        changed = False

        singleton_values: dict[int, int] = {}
        for group in model.all_different:
            singleton_values.clear()
            for variable in group:
                if lower[variable] == upper[variable]:
                    value = int(lower[variable])
                    if value in singleton_values:
                        return PropagationResult(False, lower, upper, total_rounds)
                    singleton_values[value] = variable
            for value, owner in singleton_values.items():
                for variable in group:
                    if variable == owner:
                        continue
                    if lower[variable] == value < upper[variable]:
                        lower[variable] += 1
                        changed = True
                    elif lower[variable] < value == upper[variable]:
                        upper[variable] -= 1
                        changed = True

        for expression in model.not_equal:
            minimum, maximum = _expr_bounds(expression, lower, upper)
            if minimum == maximum == 0:
                return PropagationResult(False, lower, upper, total_rounds)

        for variables, rows in model.allowed_tables:
            compatible = [
                row
                for row in rows
                if all(lower[v] <= value <= upper[v] for v, value in zip(variables, row))
            ]
            if not compatible:
                return PropagationResult(False, lower, upper, total_rounds)
            for position, variable in enumerate(variables):
                table_lower = min(row[position] for row in compatible)
                table_upper = max(row[position] for row in compatible)
                if table_lower > lower[variable]:
                    lower[variable] = table_lower
                    changed = True
                if table_upper < upper[variable]:
                    upper[variable] = table_upper
                    changed = True

        for variables, rows in model.forbidden_tables:
            if all(lower[v] == upper[v] for v in variables):
                values = tuple(int(lower[v]) for v in variables)
                if values in rows:
                    return PropagationResult(False, lower, upper, total_rounds)

        if not changed:
            return PropagationResult(True, lower, upper, total_rounds)
    return PropagationResult(True, lower, upper, total_rounds)


class SatParameters:
    def __init__(self):
        self.max_time_in_seconds = math.inf
        self.max_number_of_conflicts = 0
        self.num_search_workers = 1
        self.enumerate_all_solutions = False
        self.log_search_progress = False


class CpSolverSolutionCallback:
    def __init__(self):
        self._solver: CpSolver | None = None

    def on_solution_callback(self):
        pass

    OnSolutionCallback = on_solution_callback

    def value(self, expression):
        if self._solver is None:
            raise RuntimeError("callback is not active")
        return self._solver.value(expression)

    Value = value

    def stop_search(self):
        if self._solver is not None:
            self._solver._stop = True

    StopSearch = stop_search


class CpSolver:
    def __init__(self):
        self.parameters = SatParameters()
        self._solution: np.ndarray | None = None
        self._objective = 0
        self._best_bound = 0
        self._status = UNKNOWN
        self._branches = 0
        self._conflicts = 0
        self._wall_time = 0.0
        self._stop = False

    def solve(self, model: CpModel, solution_callback: CpSolverSolutionCallback | None = None):
        started = time.perf_counter()
        validation = model.validate()
        if validation:
            self._status = MODEL_INVALID
            return self._status
        self._solution = None
        self._branches = 0
        self._conflicts = 0
        self._stop = False
        best_objective: int | None = None
        initial_lower, initial_upper = model._initial_bounds()
        stack = [(initial_lower, initial_upper)]
        timed_out = False

        while stack and not self._stop:
            if time.perf_counter() - started >= self.parameters.max_time_in_seconds:
                timed_out = True
                break
            if (
                self.parameters.max_number_of_conflicts
                and self._conflicts >= self.parameters.max_number_of_conflicts
            ):
                timed_out = True
                break

            lower, upper = stack.pop()
            result = propagate(model, lower, upper)
            if not result.feasible:
                self._conflicts += 1
                continue
            lower, upper = result.lower_bounds, result.upper_bounds

            if model.objective is not None and best_objective is not None:
                minimum, maximum = _expr_bounds(model.objective, lower, upper)
                bound = minimum if model.objective_sense == 1 else maximum
                if (
                    model.objective_sense == 1
                    and bound >= best_objective
                    or model.objective_sense == -1
                    and bound <= best_objective
                ):
                    continue

            widths = upper - lower
            candidates = np.flatnonzero(widths)
            if not len(candidates):
                values = lower
                if not _is_solution(model, values):
                    self._conflicts += 1
                    continue
                objective = (
                    _eval_expr(model.objective, values)
                    if model.objective is not None
                    else 0
                )
                improved = (
                    best_objective is None
                    or model.objective_sense == 1
                    and objective < best_objective
                    or model.objective_sense == -1
                    and objective > best_objective
                )
                if improved:
                    best_objective = objective
                    self._objective = objective
                    self._solution = values.copy()
                if solution_callback is not None and (
                    improved or model.objective is None
                ):
                    self._solution = values.copy()
                    solution_callback._solver = self
                    solution_callback.on_solution_callback()
                if model.objective is None and not self.parameters.enumerate_all_solutions:
                    break
                continue

            variable = int(candidates[np.argmin(widths[candidates])])
            midpoint = (int(lower[variable]) + int(upper[variable])) // 2
            low_lower, low_upper = lower.copy(), upper.copy()
            high_lower, high_upper = lower.copy(), upper.copy()
            low_upper[variable] = midpoint
            high_lower[variable] = midpoint + 1
            self._branches += 1

            coefficient = (
                model.objective.terms.get(variable, 0)
                if model.objective is not None
                else 0
            )
            prefer_low = model.objective_sense * coefficient >= 0
            if prefer_low:
                stack.append((high_lower, high_upper))
                stack.append((low_lower, low_upper))
            else:
                stack.append((low_lower, low_upper))
                stack.append((high_lower, high_upper))

        self._wall_time = time.perf_counter() - started
        if solution_callback is not None:
            solution_callback._solver = None
        if self._solution is None:
            self._status = UNKNOWN if timed_out else INFEASIBLE
        elif timed_out or self._stop:
            self._status = FEASIBLE
        else:
            self._status = OPTIMAL
        if model.objective is not None and self._solution is not None:
            self._best_bound = self._objective if not stack else (
                _expr_bounds(model.objective, initial_lower, initial_upper)[
                    0 if model.objective_sense == 1 else 1
                ]
            )
        return self._status

    Solve = solve

    def value(self, expression):
        if self._solution is None:
            raise RuntimeError("solve() has not produced a solution")
        return _eval_expr(_as_expr(expression), self._solution)

    Value = value

    def boolean_value(self, literal) -> bool:
        if isinstance(literal, NotBooleanVariable):
            return not bool(self.value(literal.variable))
        return bool(self.value(literal))

    BooleanValue = boolean_value

    @property
    def objective_value(self):
        return float(self._objective)

    def ObjectiveValue(self):
        return self.objective_value

    @property
    def best_objective_bound(self):
        return float(self._best_bound)

    def BestObjectiveBound(self):
        return self.best_objective_bound

    @property
    def num_branches(self):
        return self._branches

    def NumBranches(self):
        return self.num_branches

    @property
    def num_conflicts(self):
        return self._conflicts

    def NumConflicts(self):
        return self.num_conflicts

    @property
    def wall_time(self):
        return self._wall_time

    def WallTime(self):
        return self.wall_time

    def values(self, variables):
        return np.array([self.value(variable) for variable in variables], dtype=np.int64)

    Values = values

    def boolean_values(self, variables):
        return np.array([self.boolean_value(variable) for variable in variables], dtype=bool)

    BooleanValues = boolean_values

    def status_name(self, status=None):
        return {
            UNKNOWN: "UNKNOWN",
            MODEL_INVALID: "MODEL_INVALID",
            FEASIBLE: "FEASIBLE",
            INFEASIBLE: "INFEASIBLE",
            OPTIMAL: "OPTIMAL",
        }[self._status if status is None else status]

    StatusName = status_name

    def response_stats(self):
        return (
            f"status: {self.status_name()}\n"
            f"objective: {self._objective}\n"
            f"branches: {self._branches}\n"
            f"conflicts: {self._conflicts}\n"
            f"walltime: {self._wall_time:.6f}"
        )

    ResponseStats = response_stats


def _eval_expr(expression: LinearExpr | None, values: np.ndarray) -> int:
    if expression is None:
        return 0
    return expression.offset + sum(
        coefficient * int(values[variable])
        for variable, coefficient in expression.terms.items()
    )


def _is_solution(model: CpModel, values: np.ndarray) -> bool:
    if not validate_assignment(*model._arrays(), values):
        return False
    for expression in model.not_equal:
        if _eval_expr(expression, values) == 0:
            return False
    for group in model.all_different:
        selected = [int(values[v]) for v in group]
        if len(set(selected)) != len(selected):
            return False
    for variables, rows in model.allowed_tables:
        if tuple(int(values[v]) for v in variables) not in rows:
            return False
    for variables, rows in model.forbidden_tables:
        if tuple(int(values[v]) for v in variables) in rows:
            return False
    return True


def display_bounds(bounds) -> str:
    return f"[{bounds[0]},{bounds[1]}]"


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1
