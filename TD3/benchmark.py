import matplotlib.pyplot as plt
from statistics import mean

import algorithms
from game import create_board

SEARCH_ALGORITHMS = {
    "minimax": algorithms.minimax,
    "alphabeta": algorithms.alphabeta,
    "mcts": algorithms.mcts,
}

SEARCH_DEPTHS = range(1, 6)
MCTS_SIMULATIONS = (100, 500, 1000, 2000)
MCTS_SEED = 42
NUMBER_OF_RUNS = 100  # Number of runs for each configuration to average results


def benchmark(board, algorithm, **kwargs):
    """Run one search algorithm and return its performance metrics."""
    if not isinstance(algorithm, str) or algorithm not in SEARCH_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; choose from "
            f"{', '.join(SEARCH_ALGORITHMS)}"
        )

    result = SEARCH_ALGORITHMS[algorithm](board, **kwargs)
    if "depth" in kwargs:
        configuration = f"Depth {kwargs['depth']}"
    elif "num_simulations" in kwargs:
        configuration = f"{kwargs['num_simulations']} simulations"
    else:
        configuration = "Default"

    return {
        "algorithm": algorithm,
        "configuration": configuration,
        "move": result["move"],
        "score": result["score"],
        "nodes_explored": result["nodes_explored"],
        "time": result["time"],
    }


def _print_results(results):
    headers = ("Algorithm", "Configuration", "Move", "Score", "Nodes", "Time (s)")
    rows = [
        (
            result["algorithm"],
            result["configuration"],
            str(result["move"]),
            str(result["score"]),
            str(result["nodes_explored"]),
            f"{result['time']:.6f}",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def _plot_results(minimax_results, alphabeta_results, mcts_results):
    figure, (nodes_axis, time_axis) = plt.subplots(1, 2, figsize=(12, 5))

    nodes_axis.plot(
        SEARCH_DEPTHS,
        [
            mean(
                result["nodes_explored"]
                for result in minimax_results
                if result["configuration"] == f"Depth {depth}"
            )
            for depth in SEARCH_DEPTHS
        ],
        marker="o",
        label="Minimax",
    )
    
    nodes_axis.plot(
        SEARCH_DEPTHS,
        [
            mean(
                result["nodes_explored"]
                for result in alphabeta_results
                if result["configuration"] == f"Depth {depth}"
            )
            for depth in SEARCH_DEPTHS
        ],
        marker="s",
        label="Alpha-Beta",
    )

    time_axis.plot(
        MCTS_SIMULATIONS,
        [
            mean(
                result["time"]
                for result in mcts_results
                if result["configuration"] == f"{simulations} simulations"
            )
            for simulations in MCTS_SIMULATIONS
        ],
        marker="o",
        label="MCTS",
    )

    nodes_axis.set_title("Explored nodes by search depth")
    nodes_axis.set_xlabel("Search depth")
    nodes_axis.set_ylabel("Number of explored nodes")
    nodes_axis.legend()
    nodes_axis.grid(True, alpha=0.3)

    time_axis.set_title("MCTS execution time by simulation budget")
    time_axis.set_xlabel("Number of simulations")
    time_axis.set_ylabel("Execution time (s)")
    time_axis.legend()
    time_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    plt.show()


def run_benchmarks(show_plots=True):
    """Run the required search comparisons on the same initial board."""
    board = create_board()
    minimax_results = []
    alphabeta_results = []
    mcts_results = []
    print("Running benchmarks...")
    for i in range(NUMBER_OF_RUNS):
        print(f"\n##############    Running benchmark iteration {i + 1}/{NUMBER_OF_RUNS}...     ##############\n")
        print(f"Running Minimax benchmarks for depths {list(SEARCH_DEPTHS)}...")
        minimax_results.extend([
            benchmark(board, "minimax", depth=depth) for depth in SEARCH_DEPTHS
        ])
        print(f"{i + 1}/{NUMBER_OF_RUNS} Minimax benchmarks completed.\n")

        print(f"Running Alpha-Beta benchmarks for depths {list(SEARCH_DEPTHS)}...")
        alphabeta_results.extend([
            benchmark(board, "alphabeta", depth=depth) for depth in SEARCH_DEPTHS
        ])
        print(f"{i + 1}/{NUMBER_OF_RUNS} Alpha-Beta benchmarks completed.\n")
        
        for simulations in MCTS_SIMULATIONS:
            print(f"Running MCTS benchmark with {simulations} simulations...")
            mcts_results.append(
                benchmark(
                    board,
                    "mcts",
                    num_simulations=simulations,
                    seed=MCTS_SEED,
                )
            )
        print(f"{i + 1}/{NUMBER_OF_RUNS} MCTS benchmarks completed.\n")

    for minimax_result, alphabeta_result in zip(
        minimax_results, alphabeta_results
    ):
        if minimax_result["score"] != alphabeta_result["score"]:
            raise AssertionError(
                f"Minimax and Alpha-Beta scores differ at "
                f"{minimax_result['configuration']}"
            )
        if alphabeta_result["nodes_explored"] > minimax_result["nodes_explored"]:
            raise AssertionError(
                f"Alpha-Beta explored more nodes at "
                f"{minimax_result['configuration']}"
            )

    results = {
        "minimax": minimax_results,
        "alphabeta": alphabeta_results,
        "mcts": mcts_results,
    }
    _print_results(minimax_results + alphabeta_results + mcts_results)
    if show_plots:
        _plot_results(minimax_results, alphabeta_results, mcts_results)
    return results


if __name__ == "__main__":
    run_benchmarks()
