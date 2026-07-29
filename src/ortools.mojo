"""CP propagation and routing kernels exposed through a small C ABI."""

from std.sys.info import simd_width_of

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def ip(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def floor_div(a: Int64, b: Int64) -> Int64:
    var q = a // b
    var r = a % b
    if r != 0 and ((r > 0) != (b > 0)):
        q -= 1
    return q


def ceil_div(a: Int64, b: Int64) -> Int64:
    var q = a // b
    var r = a % b
    if r != 0 and ((r > 0) == (b > 0)):
        q += 1
    return q


@export("mot_propagate")
def mot_propagate(
    coeff_addr: Int,
    var_addr: Int,
    linear_offsets_addr: Int,
    constraint_lb_addr: Int,
    constraint_ub_addr: Int,
    literal_addr: Int,
    clause_offsets_addr: Int,
    variable_lb_addr: Int,
    variable_ub_addr: Int,
    n_vars: Int,
    n_linear: Int,
    n_clauses: Int,
    max_rounds: Int,
) abi("C") -> Int:
    var coeff = ip(coeff_addr)
    var variables = ip(var_addr)
    var linear_offsets = ip(linear_offsets_addr)
    var constraint_lb = ip(constraint_lb_addr)
    var constraint_ub = ip(constraint_ub_addr)
    var literals = ip(literal_addr)
    var clause_offsets = ip(clause_offsets_addr)
    var variable_lb = ip(variable_lb_addr)
    var variable_ub = ip(variable_ub_addr)

    for v in range(n_vars):
        if variable_lb[v] > variable_ub[v]:
            return -1

    for round_index in range(max_rounds):
        var changed = False

        for c in range(n_linear):
            var start = Int(linear_offsets[c])
            var stop = Int(linear_offsets[c + 1])
            var min_activity = Int64(0)
            var max_activity = Int64(0)
            for t in range(start, stop):
                var v = Int(variables[t])
                var a = coeff[t]
                if a >= 0:
                    min_activity += a * variable_lb[v]
                    max_activity += a * variable_ub[v]
                else:
                    min_activity += a * variable_ub[v]
                    max_activity += a * variable_lb[v]

            if max_activity < constraint_lb[c] or min_activity > constraint_ub[c]:
                return -1

            for t in range(start, stop):
                var v = Int(variables[t])
                var a = coeff[t]
                if a == 0:
                    continue
                var own_min = (
                    a * variable_lb[v] if a > 0 else a * variable_ub[v]
                )
                var own_max = (
                    a * variable_ub[v] if a > 0 else a * variable_lb[v]
                )
                var others_min = min_activity - own_min
                var others_max = max_activity - own_max
                var lo = variable_lb[v]
                var hi = variable_ub[v]

                if a > 0:
                    var candidate_lo = ceil_div(
                        constraint_lb[c] - others_max, a
                    )
                    var candidate_hi = floor_div(
                        constraint_ub[c] - others_min, a
                    )
                    if candidate_lo > lo:
                        lo = candidate_lo
                    if candidate_hi < hi:
                        hi = candidate_hi
                else:
                    var candidate_hi = floor_div(
                        constraint_lb[c] - others_max, a
                    )
                    var candidate_lo = ceil_div(
                        constraint_ub[c] - others_min, a
                    )
                    if candidate_lo > lo:
                        lo = candidate_lo
                    if candidate_hi < hi:
                        hi = candidate_hi

                if lo > hi:
                    return -1
                if lo != variable_lb[v] or hi != variable_ub[v]:
                    variable_lb[v] = lo
                    variable_ub[v] = hi
                    changed = True

        for c in range(n_clauses):
            var start = Int(clause_offsets[c])
            var stop = Int(clause_offsets[c + 1])
            var satisfied = False
            var possible_count = 0
            var unit_var = -1
            var unit_value = Int64(0)
            for t in range(start, stop):
                var lit = literals[t]
                var v = Int(abs(lit) - 1)
                var desired = Int64(1) if lit > 0 else Int64(0)
                if variable_lb[v] == desired and variable_ub[v] == desired:
                    satisfied = True
                    break
                if variable_lb[v] <= desired and desired <= variable_ub[v]:
                    possible_count += 1
                    unit_var = v
                    unit_value = desired
            if satisfied:
                continue
            if possible_count == 0:
                return -1
            if possible_count == 1:
                if (
                    variable_lb[unit_var] != unit_value
                    or variable_ub[unit_var] != unit_value
                ):
                    variable_lb[unit_var] = unit_value
                    variable_ub[unit_var] = unit_value
                    changed = True

        if not changed:
            return round_index + 1
    return -2


@export("mot_validate_assignment")
def mot_validate_assignment(
    coeff_addr: Int,
    var_addr: Int,
    linear_offsets_addr: Int,
    constraint_lb_addr: Int,
    constraint_ub_addr: Int,
    literal_addr: Int,
    clause_offsets_addr: Int,
    value_addr: Int,
    n_linear: Int,
    n_clauses: Int,
) abi("C") -> Int:
    var coeff = ip(coeff_addr)
    var variables = ip(var_addr)
    var linear_offsets = ip(linear_offsets_addr)
    var constraint_lb = ip(constraint_lb_addr)
    var constraint_ub = ip(constraint_ub_addr)
    var literals = ip(literal_addr)
    var clause_offsets = ip(clause_offsets_addr)
    var values = ip(value_addr)
    comptime W = simd_width_of[DType.int64]()

    for c in range(n_linear):
        var start = Int(linear_offsets[c])
        var stop = Int(linear_offsets[c + 1])
        var activity_vector = SIMD[DType.int64, W](0)
        var t = start
        while t + W <= stop:
            var index = variables.load[width=W](t)
            activity_vector += (
                coeff.load[width=W](t) * values.gather[width=W](index)
            )
            t += W
        var activity = activity_vector.reduce_add()
        while t < stop:
            activity += coeff[t] * values[Int(variables[t])]
            t += 1
        if activity < constraint_lb[c] or activity > constraint_ub[c]:
            return 0

    for c in range(n_clauses):
        var satisfied = False
        for t in range(
            Int(clause_offsets[c]), Int(clause_offsets[c + 1])
        ):
            var literal = literals[t]
            var value = values[Int(abs(literal) - 1)]
            if (
                (literal > 0 and value == 1)
                or (literal < 0 and value == 0)
            ):
                satisfied = True
                break
        if not satisfied:
            return 0
    return 1


def is_terminal(node: Int, starts: I64Ptr, ends: I64Ptr, vehicles: Int) -> Bool:
    for v in range(vehicles):
        if node == Int(starts[v]) or node == Int(ends[v]):
            return True
    return False


def route_total(cost: I64Ptr, route: I64Ptr, length: Int, nodes: Int) -> Int64:
    var total = Int64(0)
    for i in range(length - 1):
        total += cost[Int(route[i]) * nodes + Int(route[i + 1])]
    return total


@export("mot_construct_routes")
def mot_construct_routes(
    cost_addr: Int,
    demand_addr: Int,
    capacity_addr: Int,
    start_addr: Int,
    end_addr: Int,
    route_addr: Int,
    length_addr: Int,
    load_addr: Int,
    visited_addr: Int,
    nodes: Int,
    vehicles: Int,
    strategy: Int,
) abi("C") -> Int64:
    var cost = ip(cost_addr)
    var demand = ip(demand_addr)
    var capacity = ip(capacity_addr)
    var starts = ip(start_addr)
    var ends = ip(end_addr)
    var routes = ip(route_addr)
    var lengths = ip(length_addr)
    var loads = ip(load_addr)
    var visited = ip(visited_addr)
    var stride = nodes + 2

    for i in range(nodes):
        visited[i] = 0
    for v in range(vehicles):
        visited[Int(starts[v])] = 1
        visited[Int(ends[v])] = 1
        var base = v * stride
        routes[base] = starts[v]
        routes[base + 1] = ends[v]
        lengths[v] = 2
        loads[v] = 0

    var remaining = 0
    for node in range(nodes):
        if not is_terminal(node, starts, ends, vehicles):
            remaining += 1

    if vehicles > 1:
        for v in range(vehicles):
            var seed = -1
            var best_separation = Int64(-1)
            for node in range(nodes):
                if (
                    visited[node] != 0
                    or is_terminal(node, starts, ends, vehicles)
                    or demand[node] > capacity[v]
                ):
                    continue
                var separation = (
                    cost[Int(starts[v]) * nodes + node]
                    + cost[node * nodes + Int(ends[v])]
                )
                for previous_vehicle in range(v):
                    var previous = Int(
                        routes[previous_vehicle * stride + 1]
                    )
                    var pair_distance = (
                        cost[node * nodes + previous]
                        + cost[previous * nodes + node]
                    )
                    if pair_distance < separation:
                        separation = pair_distance
                if separation > best_separation:
                    best_separation = separation
                    seed = node
            if seed >= 0:
                var base = v * stride
                routes[base + 2] = routes[base + 1]
                routes[base + 1] = Int64(seed)
                lengths[v] = 3
                loads[v] = demand[seed]
                visited[seed] = 1
                remaining -= 1

    while remaining > 0:
        var best_delta = Int64(9223372036854775807)
        var best_node = -1
        var best_vehicle = -1
        var best_position = -1

        for node in range(nodes):
            if visited[node] != 0 or is_terminal(node, starts, ends, vehicles):
                continue
            for v in range(vehicles):
                if loads[v] + demand[node] > capacity[v]:
                    continue
                var base = v * stride
                var length = Int(lengths[v])
                var first_position = 1
                if strategy == 0:
                    first_position = length - 1
                for pos in range(first_position, length):
                    var before = Int(routes[base + pos - 1])
                    var after = Int(routes[base + pos])
                    var delta = (
                        cost[before * nodes + node]
                        + cost[node * nodes + after]
                        - cost[before * nodes + after]
                    )
                    if (
                        delta < best_delta
                        or (
                            delta == best_delta
                            and (
                                best_node == -1
                                or node < best_node
                                or (
                                    node == best_node
                                    and (
                                        v < best_vehicle
                                        or (
                                            v == best_vehicle
                                            and pos < best_position
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_delta = delta
                        best_node = node
                        best_vehicle = v
                        best_position = pos

        if best_node < 0:
            return -1

        var base = best_vehicle * stride
        var length = Int(lengths[best_vehicle])
        var shift = length
        while shift > best_position:
            routes[base + shift] = routes[base + shift - 1]
            shift -= 1
        routes[base + best_position] = Int64(best_node)
        lengths[best_vehicle] += 1
        loads[best_vehicle] += demand[best_node]
        visited[best_node] = 1
        remaining -= 1

    var total = Int64(0)
    for v in range(vehicles):
        total += route_total(
            cost, routes + v * stride, Int(lengths[v]), nodes
        )
    return total


@export("mot_two_opt")
def mot_two_opt(
    cost_addr: Int,
    route_addr: Int,
    nodes: Int,
    length: Int,
    max_passes: Int,
) abi("C") -> Int64:
    var cost = ip(cost_addr)
    var route = ip(route_addr)
    for _ in range(max_passes):
        var best_delta = Int64(0)
        var best_i = -1
        var best_j = -1
        for i in range(1, length - 2):
            var old_internal = Int64(0)
            var new_internal = Int64(0)
            for j in range(i + 1, length - 1):
                old_internal += cost[
                    Int(route[j - 1]) * nodes + Int(route[j])
                ]
                new_internal += cost[
                    Int(route[j]) * nodes + Int(route[j - 1])
                ]
                var old_cost = (
                    cost[Int(route[i - 1]) * nodes + Int(route[i])]
                    + old_internal
                    + cost[Int(route[j]) * nodes + Int(route[j + 1])]
                )
                var new_cost = (
                    cost[Int(route[i - 1]) * nodes + Int(route[j])]
                    + new_internal
                    + cost[Int(route[i]) * nodes + Int(route[j + 1])]
                )
                var delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        if best_i < 0:
            break
        var left = best_i
        var right = best_j
        while left < right:
            var tmp = route[left]
            route[left] = route[right]
            route[right] = tmp
            left += 1
            right -= 1
    return route_total(cost, route, length, nodes)


@export("mot_relocate")
def mot_relocate(
    cost_addr: Int,
    demand_addr: Int,
    capacity_addr: Int,
    route_addr: Int,
    length_addr: Int,
    load_addr: Int,
    nodes: Int,
    vehicles: Int,
    max_passes: Int,
) abi("C") -> Int64:
    var cost = ip(cost_addr)
    var demand = ip(demand_addr)
    var capacity = ip(capacity_addr)
    var routes = ip(route_addr)
    var lengths = ip(length_addr)
    var loads = ip(load_addr)
    var stride = nodes + 2

    for _ in range(max_passes):
        var best_delta = Int64(0)
        var source_vehicle = -1
        var source_position = -1
        var target_vehicle = -1
        var target_position = -1

        for a in range(vehicles):
            var abase = a * stride
            var alength = Int(lengths[a])
            for i in range(1, alength - 1):
                var node = Int(routes[abase + i])
                var prev = Int(routes[abase + i - 1])
                var next = Int(routes[abase + i + 1])
                var removal = (
                    cost[prev * nodes + next]
                    - cost[prev * nodes + node]
                    - cost[node * nodes + next]
                )
                for b in range(vehicles):
                    if b == a or loads[b] + demand[node] > capacity[b]:
                        continue
                    var bbase = b * stride
                    var blength = Int(lengths[b])
                    for j in range(1, blength):
                        var before = Int(routes[bbase + j - 1])
                        var after = Int(routes[bbase + j])
                        var insertion = (
                            cost[before * nodes + node]
                            + cost[node * nodes + after]
                            - cost[before * nodes + after]
                        )
                        var delta = removal + insertion
                        if delta < best_delta:
                            best_delta = delta
                            source_vehicle = a
                            source_position = i
                            target_vehicle = b
                            target_position = j

        if source_vehicle < 0:
            break

        var abase = source_vehicle * stride
        var bbase = target_vehicle * stride
        var node = routes[abase + source_position]
        var alength = Int(lengths[source_vehicle])
        for i in range(source_position, alength - 1):
            routes[abase + i] = routes[abase + i + 1]
        lengths[source_vehicle] -= 1
        loads[source_vehicle] -= demand[Int(node)]

        var blength = Int(lengths[target_vehicle])
        var shift = blength
        while shift > target_position:
            routes[bbase + shift] = routes[bbase + shift - 1]
            shift -= 1
        routes[bbase + target_position] = node
        lengths[target_vehicle] += 1
        loads[target_vehicle] += demand[Int(node)]

    var total = Int64(0)
    for v in range(vehicles):
        total += route_total(
            cost, routes + v * stride, Int(lengths[v]), nodes
        )
    return total


@export("mot_route_cost")
def mot_route_cost(
    cost_addr: Int, route_addr: Int, nodes: Int, length: Int
) abi("C") -> Int64:
    return route_total(ip(cost_addr), ip(route_addr), length, nodes)
