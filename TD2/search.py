"""Cost-based search algorithms for the Pokémon environment."""

from __future__ import annotations

import heapq
import random
from itertools import count

from pokemon_game import (
    GameMap,
    State,
    movement_cost,
    successors,
)


def reconstruct_path(
    parents: dict[State, State | None],
    goal: State,
) -> list[State]:
    """Reconstruct a path from a parent dictionary."""
    path: list[State] = []
    current: State | None = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


def path_cost(
    game_map: GameMap,
    path: list[State],
) -> int:
    """Return the total cost of a path."""
    return sum(
        movement_cost(game_map, state)
        for state in path[1:]
    )


def manhattan_distance(
    state: State,
    goal: State,
) -> int:
    """Return the Manhattan distance between two states."""
    return (
        abs(state[0] - goal[0])
        + abs(state[1] - goal[1])
    )


def generate_random_path(
    game_map: GameMap,
    start: State,
    goal: State,
    seed: int | None = None,
) -> list[State]:
    """Generate a random valid path from start to goal.

    This function is provided only to demonstrate the visualization.
    It is not guaranteed to return a minimum-cost or shortest path.

    The function performs a randomized depth-first exploration: the valid
    successor states are shuffled before being added to the frontier.

    Parameters
    ----------
    game_map : GameMap
        Weighted Pokémon map.
    start : State
        Initial trainer position.
    goal : State
        Pokémon Center position.
    seed : int or None, default=None
        Random seed used to make the result reproducible.

    Returns
    -------
    list[State]
        A valid path beginning at ``start`` and ending at ``goal``.

    Raises
    ------
    RuntimeError
        If no path exists between the start and the goal.
    """
    rng = random.Random(seed)

    frontier = [start]

    parents: dict[State, State | None] = {
        start: None,
    }

    while frontier:
        current = frontier.pop()

        if current == goal:
            return reconstruct_path(parents, goal)

        next_states = successors(game_map, current)
        rng.shuffle(next_states)

        for next_state in next_states:
            if next_state in parents:
                continue

            parents[next_state] = current
            frontier.append(next_state)

    raise RuntimeError(
        "No path exists between the start and the goal."
    )

def dijkstra(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using Dijkstra's algorithm.

    Returns
    -------
    path : list[State]
        Minimum-cost path from start to goal.
    considered_states : int
        Number of states removed from the priority queue.
    """
    frontier: list[tuple[int, int, State]] = []
    tie_breaker = count()
    heapq.heappush(frontier, (0, next(tie_breaker), start))

    distances: dict[State, int] = {start: 0}
    parents: dict[State, State | None] = {start: None}
    considered_states = 0

    while frontier:
        current_cost, _, current = heapq.heappop(frontier)

        if current_cost != distances[current]:
            continue

        considered_states += 1
        if current == goal:
            return reconstruct_path(parents, goal), considered_states

        for next_state in successors(game_map, current):
            new_cost = current_cost + movement_cost(game_map, next_state)
            if new_cost >= distances.get(next_state, float("inf")):
                continue

            distances[next_state] = new_cost
            parents[next_state] = current
            heapq.heappush(
                frontier,
                (new_cost, next(tie_breaker), next_state),
            )

    raise RuntimeError(
        "No path exists between the start and the goal."
    )


def heuristic(state: State, goal: State) -> int:
    """Return the heuristic value for a state.

    The Manhattan distance is used as the heuristic.
    """
    return manhattan_distance(state, goal)

def a_star(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using A* search.

    The Manhattan distance is used as the heuristic.

    Returns
    -------
    path : list[State]
        Minimum-cost path from start to goal.
    considered_states : int
        Number of states removed from the priority queue.
    """
    frontier: list[tuple[int, int, int, State]] = []
    tie_breaker = count()
    heapq.heappush(
        frontier,
        (heuristic(start, goal), next(tie_breaker), 0, start),
    )

    distances: dict[State, int] = {start: 0}
    parents: dict[State, State | None] = {start: None}
    considered_states = 0

    while frontier:
        _, _, current_cost, current = heapq.heappop(frontier)

        if current_cost != distances[current]:
            continue

        considered_states += 1
        if current == goal:
            return reconstruct_path(parents, goal), considered_states

        for next_state in successors(game_map, current):
            new_cost = current_cost + movement_cost(game_map, next_state)
            if new_cost >= distances.get(next_state, float("inf")):
                continue

            distances[next_state] = new_cost
            parents[next_state] = current
            estimated_total_cost = new_cost + heuristic(next_state, goal)
            heapq.heappush(
                frontier,
                (
                    estimated_total_cost,
                    next(tie_breaker),
                    new_cost,
                    next_state,
                ),
            )

    raise RuntimeError(
        "No path exists between the start and the goal."
    )

    