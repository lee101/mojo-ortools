import itertools

import numpy as np
import pytest

from mojo_ortools.propagation import propagate_bounds, validate_assignment
from mojo_ortools.sat.python import cp_model


def test_linear_fixed_point():
    result = propagate_bounds(
        coefficients=[2, 1, 1, -1],
        variables=[0, 1, 0, 1],
        linear_offsets=[0, 2, 4],
        constraint_lower_bounds=[7, -1],
        constraint_upper_bounds=[7, 1],
        variable_lower_bounds=[0, 0],
        variable_upper_bounds=[10, 10],
    )
    assert result.feasible
    assert result.lower_bounds.tolist() == [2, 1]
    assert result.upper_bounds.tolist() == [3, 3]


def test_signed_coefficients_and_rounding():
    result = propagate_bounds(
        coefficients=[-3, 2],
        variables=[0, 1],
        linear_offsets=[0, 2],
        constraint_lower_bounds=[1],
        constraint_upper_bounds=[1],
        variable_lower_bounds=[-5, -5],
        variable_upper_bounds=[5, 5],
    )
    feasible = [
        values
        for values in itertools.product(range(-5, 6), repeat=2)
        if -3 * values[0] + 2 * values[1] == 1
    ]
    assert result.feasible
    assert result.lower_bounds.tolist() == [min(v[i] for v in feasible) for i in range(2)]
    assert result.upper_bounds.tolist() == [max(v[i] for v in feasible) for i in range(2)]


def test_clause_unit_propagation():
    result = propagate_bounds(
        coefficients=[],
        variables=[],
        linear_offsets=[0],
        constraint_lower_bounds=[],
        constraint_upper_bounds=[],
        literals=[1, 2, -2, 3],
        clause_offsets=[0, 2, 4],
        variable_lower_bounds=[0, 0, 0],
        variable_upper_bounds=[0, 1, 1],
    )
    assert result.feasible
    assert result.lower_bounds.tolist() == [0, 1, 1]
    assert result.upper_bounds.tolist() == [0, 1, 1]


def test_clause_conflict():
    result = propagate_bounds(
        [], [], [0], [], [], [1], [0, 1], [0], [0]
    )
    assert not result.feasible


def test_simd_assignment_validation_with_scalar_tail():
    count = 17
    coefficients = np.array(
        [value if value % 2 else -value for value in range(1, count + 1)],
        dtype=np.int64,
    )
    variables = np.arange(count, dtype=np.int64)
    values = np.arange(3, count + 3, dtype=np.int64)
    activity = int(coefficients @ values)
    arguments = (
        coefficients,
        variables,
        np.array([0, count], dtype=np.int64),
        np.array([activity], dtype=np.int64),
        np.array([activity], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
    )

    assert validate_assignment(*arguments, values)
    invalid = values.copy()
    invalid[-1] += 1
    assert not validate_assignment(*arguments, invalid)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ((0, 10, 0, 10, 4, 4), ([0, 0], [4, 4])),
        ((-4, 7, -2, 8, -3, -3), ([-4, -2], [-1, 1])),
    ],
)
def test_propagation_matches_exhaustive_single_sum(bounds, expected):
    xlo, xhi, ylo, yhi, target_lo, target_hi = bounds
    result = propagate_bounds(
        [1, 1],
        [0, 1],
        [0, 2],
        [target_lo],
        [target_hi],
        variable_lower_bounds=[xlo, ylo],
        variable_upper_bounds=[xhi, yhi],
    )
    feasible = [
        (x, y)
        for x in range(xlo, xhi + 1)
        for y in range(ylo, yhi + 1)
        if target_lo <= x + y <= target_hi
    ]
    assert result.feasible == bool(feasible)
    assert result.lower_bounds.tolist() == [min(row[i] for row in feasible) for i in range(2)]
    assert result.upper_bounds.tolist() == [max(row[i] for row in feasible) for i in range(2)]
    assert (result.lower_bounds.tolist(), result.upper_bounds.tolist()) == expected


def test_model_propagate_all_different_and_table():
    model = cp_model.CpModel()
    x = model.new_int_var(1, 3, "x")
    y = model.new_int_var(1, 3, "y")
    z = model.new_int_var(1, 1, "z")
    model.add_all_different([x, y, z])
    model.add_allowed_assignments([x, y], [(2, 3), (3, 2)])
    result = cp_model.propagate(model)
    assert result.feasible
    assert result.lower_bounds.tolist() == [2, 2, 1]
    assert result.upper_bounds.tolist() == [3, 3, 1]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"variables": [1]}, "variable index"),
        ({"linear_offsets": [0, 2, 1]}, "nondecreasing"),
        ({"literals": [0], "clause_offsets": [0, 1]}, "literal"),
    ],
)
def test_invalid_ffi_indices_are_rejected(kwargs, message):
    arguments = dict(
        coefficients=[1],
        variables=[0],
        linear_offsets=[0, 1],
        constraint_lower_bounds=[0],
        constraint_upper_bounds=[1],
        literals=[],
        clause_offsets=[0],
        variable_lower_bounds=[0],
        variable_upper_bounds=[1],
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        propagate_bounds(**arguments)


def test_lossy_dtype_coercion_is_rejected():
    with pytest.raises(TypeError, match="integer"):
        propagate_bounds(
            [1.5], [0], [0, 1], [0], [2],
            variable_lower_bounds=[0], variable_upper_bounds=[1],
        )


def test_round_limit_does_not_silently_report_a_fixed_point():
    with pytest.raises(RuntimeError, match="did not converge"):
        propagate_bounds(
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 3],
            [1, 1],
            [1, 1],
            variable_lower_bounds=[0, 0],
            variable_upper_bounds=[1, 1],
            max_rounds=1,
        )
