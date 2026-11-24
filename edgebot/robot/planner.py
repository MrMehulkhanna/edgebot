"""Global path planning: A* search on an inflated occupancy grid (Russell & Norvig,
Ch. 3) followed by line-of-sight path shortening. The learned RL policy acts as the
local controller that tracks the resulting waypoints while reacting to obstacles."""

from __future__ import annotations

import heapq

import numpy as np

from edgebot.envs.jetbot_nav import Scenario


def occupancy_grid(scn: Scenario, res: float = 0.05, inflate: float = 0.25) -> np.ndarray:
    nx, ny = int(np.ceil(scn.width / res)), int(np.ceil(scn.height / res))
    xs, ys = (np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    occ = (X < inflate) | (Y < inflate) | (X > scn.width - inflate) | (Y > scn.height - inflate)
    for b in scn.boxes:
        dx = np.maximum(np.maximum(b[0] - X, 0), X - b[2])
        dy = np.maximum(np.maximum(b[1] - Y, 0), Y - b[3])
        occ |= np.hypot(dx, dy) < inflate
    for c in scn.circles:
        occ |= np.hypot(X - c[0], Y - c[1]) < c[2] + inflate
    return occ  # indexed [iy, ix]


def astar(occ: np.ndarray, start: tuple, goal: tuple):
    """8-connected A* with an octile heuristic. start/goal are (ix, iy) cells."""
    H, W = occ.shape
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return max(dx, dy) + 0.414 * min(dx, dy)

    g = {start: 0.0}
    parent = {start: None}
    pq = [(h(start), start)]
    closed = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], len(closed)
        closed.add(cur)
        for dx, dy, cost in moves:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < W and 0 <= ny < H) or occ[ny, nx]:
                continue
            if dx and dy and (occ[cur[1], nx] or occ[ny, cur[0]]):
                continue  # no corner cutting
            ng = g[cur] + cost
            if ng < g.get((nx, ny), np.inf):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = cur
                heapq.heappush(pq, (ng + h((nx, ny)), (nx, ny)))
    return None, len(closed)


def _line_free(occ, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    xs = np.linspace(a[0], b[0], n + 1).round().astype(int)
    ys = np.linspace(a[1], b[1], n + 1).round().astype(int)
    return not occ[ys, xs].any()


def shorten(occ, path):
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(occ, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def nearest_free(occ, cell):
    if not occ[cell[1], cell[0]]:
        return cell
    free = np.argwhere(~occ)
    d = (free[:, 1] - cell[0]) ** 2 + (free[:, 0] - cell[1]) ** 2
    iy, ix = free[np.argmin(d)]
    return int(ix), int(iy)


def plan_path(scn: Scenario, start_xy, goal_xy, res: float = 0.05, inflate: float = 0.25, max_seg: float = 0.6):
    """Returns a list of world-frame waypoints from start to goal (excluding start)."""
    occ = occupancy_grid(scn, res, inflate)
    to_cell = lambda p: (int(np.clip(p[0] / res, 0, occ.shape[1] - 1)), int(np.clip(p[1] / res, 0, occ.shape[0] - 1)))
    s = nearest_free(occ, to_cell(start_xy))
    g = nearest_free(occ, to_cell(goal_xy))
    path, expanded = astar(occ, s, g)
    if path is None:
        return None
    pts = [np.array([(c[0] + 0.5) * res, (c[1] + 0.5) * res]) for c in shorten(occ, path)]
    pts[-1] = np.asarray(goal_xy, float)
    # subdivide long segments so the local policy always sees a nearby goal
    wps = []
    prev = np.asarray(start_xy, float)
    for p in pts[1:]:
        n = int(np.ceil(np.linalg.norm(p - prev) / max_seg))
        for k in range(1, n + 1):
            wps.append(prev + (p - prev) * k / n)
        prev = p
    return wps
