"""Run Gaussian N-armed bandit strategy comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for one action-selection strategy."""

    name: str
    epsilon: float
    optimistic_initial: float
    ucb_c: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the simulation experiment."""
    parser = argparse.ArgumentParser(
        description="Compare strategies on a Gaussian N-armed testbed.",
    )
    parser.add_argument(
        "--arms",
        type=int,
        default=10,
        help="Number of bandit arms in the testbed.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2000,
        help="Number of independent runs used to estimate learning curves.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of interaction steps per run.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for epsilon-greedy.",
    )
    parser.add_argument(
        "--optimistic-initial",
        type=float,
        default=5.0,
        help="Initial value estimate used by optimistic initialization.",
    )
    parser.add_argument(
        "--ucb-c",
        type=float,
        default=2.0,
        help="Exploration coefficient used by UCB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=Path("src") / "learning_curves.png",
        help="Output path for the learning-curve figure.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument ranges before running the simulation."""
    if args.arms < 2:
        raise ValueError("`--arms` must be at least 2.")
    if args.runs < 1:
        raise ValueError("`--runs` must be at least 1.")
    if args.steps < 1:
        raise ValueError("`--steps` must be at least 1.")
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("`--epsilon` must be between 0 and 1.")
    if args.ucb_c <= 0:
        raise ValueError("`--ucb-c` must be greater than 0.")


def select_action(
    estimates: np.ndarray,
    counts: np.ndarray,
    step: int,
    config: StrategyConfig,
    rng: np.random.Generator,
) -> int:
    """Select an action under epsilon-greedy or UCB behavior."""
    if config.ucb_c is not None:
        # Force one pull per arm before applying the UCB bonus term.
        untried = np.where(counts == 0)[0]
        if untried.size > 0:
            return int(untried[0])
        bonus = config.ucb_c * np.sqrt(np.log(step + 1) / counts)
        values = estimates + bonus
    else:
        values = estimates

    if rng.random() < config.epsilon:
        return int(rng.integers(len(estimates)))

    best_actions = np.flatnonzero(values == values.max())
    return int(rng.choice(best_actions))


def run_strategy(
    q_true: np.ndarray,
    config: StrategyConfig,
    steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one strategy on one testbed instance and return traces."""
    arms = q_true.size
    estimates = np.full(arms, config.optimistic_initial, dtype=float)
    counts = np.zeros(arms, dtype=float)
    rewards = np.zeros(steps, dtype=float)
    optimal_hits = np.zeros(steps, dtype=float)
    optimal_action = int(np.argmax(q_true))

    for step in range(steps):
        action = select_action(estimates, counts, step, config, rng)
        reward = rng.normal(loc=q_true[action], scale=1.0)
        counts[action] += 1.0
        estimates[action] += (reward - estimates[action]) / counts[action]
        rewards[step] = reward
        optimal_hits[step] = float(action == optimal_action)

    return rewards, optimal_hits


def run_experiment(
    arms: int,
    runs: int,
    steps: int,
    strategies: List[StrategyConfig],
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Simulate all strategies and aggregate average learning curves."""
    rng = np.random.default_rng(seed)
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for strategy in strategies:
        reward_sum = np.zeros(steps, dtype=float)
        optimal_sum = np.zeros(steps, dtype=float)
        for _ in range(runs):
            q_true = rng.normal(loc=0.0, scale=1.0, size=arms)
            rewards, optimal_hits = run_strategy(q_true, strategy, steps, rng)
            reward_sum += rewards
            optimal_sum += optimal_hits
        results[strategy.name] = (reward_sum / runs, optimal_sum / runs)

    return results


def plot_results(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot average reward and optimal-action curves for each strategy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for name, (avg_reward, optimal_rate) in results.items():
        axes[0].plot(avg_reward, label=name)
        axes[1].plot(optimal_rate * 100.0, label=name)

    axes[0].set_title("Average reward")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Reward")

    axes[1].set_title("% optimal action")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action (%)")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Print final-step metrics to complement the learning-curve plot."""
    print("Final-step performance:")
    for name, (avg_reward, optimal_rate) in results.items():
        reward = avg_reward[-1]
        optimal_pct = optimal_rate[-1] * 100.0
        print(
            f"- {name}: reward={reward:.3f}, "
            f"optimal_action={optimal_pct:.1f}%"
        )


def main() -> None:
    """Run the full strategy comparison workflow."""
    args = parse_args()
    validate_args(args)
    strategies = [
        StrategyConfig(
            name=f"epsilon-greedy (eps={args.epsilon})",
            epsilon=args.epsilon,
            optimistic_initial=0.0,
        ),
        StrategyConfig(
            name=f"optimistic init (Q1={args.optimistic_initial})",
            epsilon=0.0,
            optimistic_initial=args.optimistic_initial,
        ),
        StrategyConfig(
            name=f"UCB (c={args.ucb_c})",
            epsilon=0.0,
            optimistic_initial=0.0,
            ucb_c=args.ucb_c,
        ),
    ]
    results = run_experiment(
        arms=args.arms,
        runs=args.runs,
        steps=args.steps,
        strategies=strategies,
        seed=args.seed,
    )
    plot_results(results, args.save_plot)
    print_summary(results)
    print(f"Saved learning-curve plot to: {args.save_plot}")


if __name__ == "__main__":
    main()
