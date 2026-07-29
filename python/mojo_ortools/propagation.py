"""Low-level fixed-point propagation over integer bounds and Boolean clauses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import addr, i64, lib, nonempty

INF = np.int64(1 << 60)


@dataclass(frozen=True)
class PropagationResult:
    feasible: bool
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    rounds: int


def _offsets(offsets: np.ndarray, item_count: int, name: str) -> int:
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if offsets[0] != 0 or offsets[-1] != item_count:
        raise ValueError(f"{name} must start at zero and cover all items")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be nondecreasing")
    return len(offsets) - 1


def _validate_problem(
    coeff, var, linear_offsets, constraint_lb, constraint_ub,
    literals, clause_offsets, value_count, lower=None, upper=None,
):
    arrays = (coeff, var, constraint_lb, constraint_ub, literals)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("coefficients, indices, bounds, and literals must be one-dimensional")
    if len(coeff) != len(var):
        raise ValueError("coefficients and variables must have equal lengths")
    n_linear = _offsets(linear_offsets, len(coeff), "linear_offsets")
    n_clauses = _offsets(clause_offsets, len(literals), "clause_offsets")
    if len(constraint_lb) != n_linear or len(constraint_ub) != n_linear:
        raise ValueError("one lower and upper bound is required per linear constraint")
    if np.any(constraint_lb > constraint_ub):
        raise ValueError("constraint lower bound exceeds upper bound")
    if np.any(var < 0) or np.any(var >= value_count):
        raise ValueError("variable index outside the bounds vector")
    if np.any(literals == 0) or np.any(np.abs(literals.astype(object)) > value_count):
        raise ValueError("literal references a variable outside the bounds vector")
    if lower is not None:
        max_terms = int(np.max(np.diff(linear_offsets), initial=0))
        max_coefficient = max((abs(int(value)) for value in coeff), default=0)
        max_bound = max((abs(int(value)) for value in lower), default=0)
        max_bound = max(max_bound, max((abs(int(value)) for value in upper), default=0))
        if max_terms * max_coefficient * max_bound > (1 << 63) - 1:
            raise OverflowError("linear activity bounds might not fit in int64")
    return n_linear, n_clauses


def propagate_bounds(
    coefficients,
    variables,
    linear_offsets,
    constraint_lower_bounds,
    constraint_upper_bounds,
    literals=(),
    clause_offsets=(0,),
    variable_lower_bounds=(),
    variable_upper_bounds=(),
    *,
    max_rounds: int = 100,
) -> PropagationResult:
    """Propagate sparse linear constraints and clauses to a fixed point.

    Linear constraints use CSR storage. Constraint ``c`` contains terms in
    ``linear_offsets[c]:linear_offsets[c + 1]`` and is bounded on both sides.
    A literal is encoded as ``var + 1`` for positive and ``-(var + 1)`` for
    negated. Bounds are copied; callers retain ownership of their inputs.
    """

    coeff_raw = i64(coefficients)
    var_raw = i64(variables)
    linear_offsets_raw = i64(linear_offsets)
    constraint_lb_raw = i64(constraint_lower_bounds)
    constraint_ub_raw = i64(constraint_upper_bounds)
    literal_raw = i64(literals)
    clause_offsets_raw = i64(clause_offsets)
    lower = i64(variable_lower_bounds, copy=True)
    upper = i64(variable_upper_bounds, copy=True)

    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("variable bounds must be one-dimensional and equally sized")
    n_linear, n_clauses = _validate_problem(
        coeff_raw, var_raw, linear_offsets_raw, constraint_lb_raw,
        constraint_ub_raw, literal_raw, clause_offsets_raw, len(lower), lower, upper,
    )
    coeff, var = nonempty(coeff_raw), nonempty(var_raw)
    linear_offsets_array = nonempty(linear_offsets_raw)
    constraint_lb, constraint_ub = nonempty(constraint_lb_raw), nonempty(constraint_ub_raw)
    literal_array, clause_offsets_array = nonempty(literal_raw), nonempty(clause_offsets_raw)
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")

    result = int(
        lib().mot_propagate(
            addr(coeff),
            addr(var),
            addr(linear_offsets_array),
            addr(constraint_lb),
            addr(constraint_ub),
            addr(literal_array),
            addr(clause_offsets_array),
            addr(lower),
            addr(upper),
            len(lower),
            n_linear,
            n_clauses,
            max_rounds,
        )
    )
    if result == -2:
        raise RuntimeError("propagation did not converge within max_rounds")
    return PropagationResult(result >= 0, lower, upper, max(result, 0))


def validate_assignment(
    coefficients,
    variables,
    linear_offsets,
    constraint_lower_bounds,
    constraint_upper_bounds,
    literals,
    clause_offsets,
    values,
) -> bool:
    coeff_raw, var_raw = i64(coefficients), i64(variables)
    linear_offsets_raw = i64(linear_offsets)
    constraint_lb_raw, constraint_ub_raw = i64(constraint_lower_bounds), i64(constraint_upper_bounds)
    literal_raw, clause_offsets_raw = i64(literals), i64(clause_offsets)
    values_raw = i64(values)
    if values_raw.ndim != 1:
        raise ValueError("values must be one-dimensional")
    n_linear, n_clauses = _validate_problem(
        coeff_raw, var_raw, linear_offsets_raw, constraint_lb_raw,
        constraint_ub_raw, literal_raw, clause_offsets_raw, len(values_raw),
    )
    max_terms = int(np.max(np.diff(linear_offsets_raw), initial=0))
    max_coefficient = max((abs(int(value)) for value in coeff_raw), default=0)
    max_value = max((abs(int(value)) for value in values_raw), default=0)
    if max_terms * max_coefficient * max_value > (1 << 63) - 1:
        raise OverflowError("linear activity might not fit in int64")
    coeff, var = nonempty(coeff_raw), nonempty(var_raw)
    linear_offsets_array = nonempty(linear_offsets_raw)
    constraint_lb, constraint_ub = nonempty(constraint_lb_raw), nonempty(constraint_ub_raw)
    literal_array, clause_offsets_array = nonempty(literal_raw), nonempty(clause_offsets_raw)
    value_array = nonempty(values_raw)
    return bool(
        lib().mot_validate_assignment(
            addr(coeff),
            addr(var),
            addr(linear_offsets_array),
            addr(constraint_lb),
            addr(constraint_ub),
            addr(literal_array),
            addr(clause_offsets_array),
            addr(value_array),
            n_linear,
            n_clauses,
        )
    )
