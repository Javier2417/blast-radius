"""Exact and fast millisecond-accurate worker packer matching SCORING.md."""

from __future__ import annotations

import math

MAX_WORKERS = 8
WORKER_COST_MS = 2000


def duration_ms(seconds: float) -> int:
    """Convert test seconds to exact milliseconds matching score.py."""
    return math.floor(seconds * 1000 + 0.5)


def _lpt_pack(items: list[tuple[int, int]], num_workers: int) -> tuple[int, list[list[int]], list[int]]:
    """Longest-Processing-Time greedy schedule across num_workers."""
    bins: list[list[int]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for test_num, dur in items:
        idx = loads.index(min(loads))
        bins[idx].append(test_num)
        loads[idx] += dur
    return max(loads), bins, loads


def _bfd_limit(items: list[tuple[int, int]], num_workers: int, limit: int) -> tuple[list[list[int]], list[int]] | None:
    """Best-Fit Decreasing check against a makespan limit."""
    bins: list[list[int]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for test_num, dur in items:
        best_idx = -1
        best_load = -1
        for i in range(num_workers):
            new_load = loads[i] + dur
            if new_load <= limit and loads[i] > best_load:
                best_load = loads[i]
                best_idx = i
        if best_idx < 0:
            return None
        bins[best_idx].append(test_num)
        loads[best_idx] += dur
    return bins, loads


def _local_improve(
    bins: list[list[int]],
    loads: list[int],
    dur_map: dict[int, int],
) -> tuple[int, list[list[int]], list[int]]:
    """Tighten makespan via 1-moves and 2-swaps out of busiest workers."""
    w = len(bins)
    if w <= 1:
        return (loads[0] if loads else 0), bins, loads

    for _ in range(64):
        max_load = max(loads)
        src = loads.index(max_load)
        improved = False

        # 1-move from max-load worker to any lighter worker
        for pos, tnum in enumerate(bins[src]):
            d = dur_map[tnum]
            for dst in range(w):
                if dst == src:
                    continue
                if loads[dst] + d < max_load:
                    bins[src].pop(pos)
                    bins[dst].append(tnum)
                    loads[src] -= d
                    loads[dst] += d
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # 2-swap between max-load worker and another worker
        for p1, t1 in enumerate(bins[src]):
            d1 = dur_map[t1]
            for dst in range(w):
                if dst == src:
                    continue
                for p2, t2 in enumerate(bins[dst]):
                    d2 = dur_map[t2]
                    if d1 > d2 and loads[src] - d1 + d2 < max_load and loads[dst] - d2 + d1 < max_load:
                        bins[src][p1] = t2
                        bins[dst][p2] = t1
                        loads[src] += d2 - d1
                        loads[dst] += d1 - d2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return max(loads), bins, loads


def _exact_pack_small(
    items: list[tuple[int, int]],
    num_workers: int,
    lb: int,
    ub_makespan: int,
    node_cap: int = 40000,
) -> tuple[int, list[list[int]]] | None:
    """Binary-search exact makespan with symmetry-broken DFS for len(items) <= 32."""
    n = len(items)
    durs = [d for _, d in items]

    def can_fit(limit: int) -> list[int] | None:
        loads = [0] * num_workers
        assign = [0] * n
        nodes = [0]

        def dfs(idx: int) -> bool:
            if idx == n:
                return True
            nodes[0] += 1
            if nodes[0] > node_cap:
                return False
            d = durs[idx]
            seen_loads: set[int] = set()
            # Try heavier bins first (best-fit tendency) to pack tight early
            order = sorted(range(num_workers), key=loads.__getitem__, reverse=True)
            for b in order:
                cur = loads[b]
                if cur in seen_loads or cur + d > limit:
                    continue
                seen_loads.add(cur)
                loads[b] = cur + d
                assign[idx] = b
                if dfs(idx + 1):
                    return True
                loads[b] = cur
            return False

        # Pin first item to worker 0 by symmetry
        if durs[0] > limit:
            return None
        loads[0] = durs[0]
        assign[0] = 0
        if dfs(1):
            return assign
        return None

    best_assign = None
    best_limit = ub_makespan
    lo, hi = lb, ub_makespan - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        res = can_fit(mid)
        if res is not None:
            best_assign = res
            best_limit = max(sum(durs[i] for i in range(n) if res[i] == b) for b in range(num_workers))
            hi = best_limit - 1
        else:
            lo = mid + 1

    if best_assign is None:
        return None
    bins: list[list[int]] = [[] for _ in range(num_workers)]
    for i, b in enumerate(best_assign):
        bins[b].append(items[i][0])
    return best_limit, bins


def pack(items: list[tuple[int, int]]) -> tuple[int, list[list[int]]]:
    """Partition (test_number, duration_ms) across 1..8 workers minimizing simulated ms.

    Returns (total_cost_ms, workers_list). Empty input returns (0, []).
    """
    if not items:
        return 0, []

    zero_nums = [num for num, dur in items if dur <= 0]
    nonzero = sorted(((num, dur) for num, dur in items if dur > 0), key=lambda x: (-x[1], x[0]))

    if not nonzero:
        return WORKER_COST_MS, [zero_nums]

    if len(nonzero) == 1:
        worker = [nonzero[0][0], *zero_nums]
        return nonzero[0][1] + WORKER_COST_MS, [worker]

    dur_map = dict(nonzero)
    total_dur = sum(dur for _, dur in nonzero)
    max_dur = nonzero[0][1]

    best_cost = total_dur + WORKER_COST_MS
    best_workers: list[list[int]] = [[num for num, _ in nonzero]]

    max_w = min(MAX_WORKERS, len(nonzero))
    for w in range(2, max_w + 1):
        lb = max(max_dur, (total_dur + w - 1) // w)
        if lb + WORKER_COST_MS * w >= best_cost:
            break

        makespan, bins, loads = _lpt_pack(nonzero, w)
        if makespan > lb:
            # Try Multi-Fit (BFD binary search) first
            lo, hi = lb, makespan - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                res = _bfd_limit(nonzero, w, mid)
                if res is not None:
                    bins, loads = res
                    makespan = max(loads)
                    hi = makespan - 1
                else:
                    lo = mid + 1

            makespan, bins, loads = _local_improve(bins, loads, dur_map)

            if makespan > lb and len(nonzero) <= 32:
                exact = _exact_pack_small(nonzero, w, lb, makespan)
                if exact is not None:
                    makespan, bins = exact

        cost = makespan + WORKER_COST_MS * w
        if cost < best_cost:
            best_cost = cost
            best_workers = [b for b in bins if b]

    if zero_nums:
        best_workers[0].extend(zero_nums)
    return best_cost, best_workers


def marginal_cost_ms(
    base_items: list[tuple[int, int]],
    base_cost: int,
    base_workers: list[list[int]],
    dur_by_num: dict[int, int],
    extra_item: tuple[int, int],
) -> int:
    """Compute minimal additional simulated ms caused by adding extra_item to base_items."""
    if not base_items:
        return WORKER_COST_MS + extra_item[1]

    extra_dur = extra_item[1]
    if extra_dur <= 0:
        return 0

    # Fast exact free-slack check on existing packed workers
    if base_workers:
        max_load = base_cost - WORKER_COST_MS * len(base_workers)
        for worker in base_workers:
            w_load = sum(dur_by_num.get(t, 0) for t in worker)
            if w_load + extra_dur <= max_load:
                return 0

    new_cost, _ = pack([*base_items, extra_item])
    return max(0, new_cost - base_cost)
