#!/usr/bin/env python3
"""
Grid-world MDP + Value Iteration
Incrementally implement functions as described in the steps.
"""

import argparse
from typing import List, Tuple, Dict, Set
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ------------------------------
# Function placeholders (implement one at a time)
# ------------------------------
def parse_cli_args(args=None):
    """
    Parse CLI args. `args` can be passed for testing.
    Obstacles are passed as a semicolon-separated list of "r,c" pairs:
      e.g. --obstacles "1,2;2,2"
    """
    parser = argparse.ArgumentParser(description="Grid MDP value-iteration")
    parser.add_argument("--rows", type=int, default=5, help="number of rows")
    parser.add_argument("--cols", type=int, default=5, help="number of cols")
    parser.add_argument("--start", type=int, nargs=2, default=[0,0], metavar=("R","C"),
                        help="start cell r c (0-indexed)")
    parser.add_argument("--goal", type=int, nargs=2, default=[4,4], metavar=("R","C"),
                        help="goal cell r c (0-indexed)")
    parser.add_argument("--obstacles", type=str, default="", 
                        help='semicolon-separated list of obstacles "r,c;r,c"')
    parser.add_argument("--slip", type=float, default=0.1, help="slip probability for unintended move")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--theta", type=float, default=1e-6, help="value iteration threshold")
    parser.add_argument("--visualize", action="store_true", help="save visualizations to ./visuals")
    return parser.parse_args(args)

def get_actions():
    """
    Return a dict mapping action-name -> (dr, dc).
    dr = change in row (negative = up), dc = change in column.
    """
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

def create_state_space(rows, cols, obstacles_str, start, goal):
    """
    Returns:
      - states: list of (r,c) for every cell (every cell is a state)
      - obstacles_set: set of (r,c)
      - start: (r,c)
      - goal: (r,c)
    Validates start/goal are inside grid.
    """
    def parse_obstacles(s):
        s = s.strip()
        if not s:
            return set()
        parts = [p.strip() for p in s.split(";") if p.strip()]
        obs = set()
        for p in parts:
            r_str, c_str = p.split(",")
            obs.add((int(r_str), int(c_str)))
        return obs

    obstacles_set = parse_obstacles(obstacles_str)
    # validate
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("start out of bounds")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError("goal out of bounds")
    states = [(r, c) for r in range(rows) for c in range(cols)]
    return states, obstacles_set, tuple(start), tuple(goal)

def build_transition_model(states, actions, rows, cols, slip_prob=0.1):
    """
    transitions[state][action] = list of (prob, next_state)
    states are (r,c) tuples.
    slip_prob: total probability mass for unintended perpendicular moves.
    """
    # Precompute perpendiculars
    perp = {
        "up": ["left", "right"],
        "down": ["left", "right"],
        "left": ["up", "down"],
        "right": ["up", "down"],
    }
    transitions = {}
    state_set = set(states)
    for s in states:
        transitions[s] = {}
        for a, delta in actions.items():
            prob_map = {}  # next_state -> prob (accumulate)
            # intended
            intended = (s[0] + delta[0], s[1] + delta[1])
            if not (0 <= intended[0] < rows and 0 <= intended[1] < cols):
                intended = s  # hit wall -> stay
            prob_map[intended] = prob_map.get(intended, 0.0) + (1.0 - slip_prob)
            # slips
            for slip_a in perp[a]:
                d = actions[slip_a]
                nxt = (s[0] + d[0], s[1] + d[1])
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    nxt = s
                prob_map[nxt] = prob_map.get(nxt, 0.0) + (slip_prob / 2.0)
            # convert to list
            transitions[s][a] = [(p, ns) for ns, p in prob_map.items()]
    # quick check: each action prob sums to 1
    for s in states:
        for a in actions:
            psum = sum([p for p, _ in transitions[s][a]])
            if not math.isclose(psum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(f"probs for state {s}, action {a} sum to {psum}")
    return transitions

def get_reward(s, a, s_next, goal, obstacles_set, step_reward=0.0, obstacle_penalty=-1.0, goal_reward=1.0):
    """
    Reward for transition (s, a, s_next).
    - If entering goal (s != s_next and s_next == goal): +goal_reward
    - If entering obstacle (s != s_next and s_next in obstacles_set): obstacle_penalty
    - Otherwise step_reward (default 0)
    """
    if s != s_next and s_next == goal:
        return goal_reward
    if s != s_next and s_next in obstacles_set:
        return obstacle_penalty
    return step_reward

def initialize_values(states):
    """Return dict mapping state -> initial value (float)."""
    return {s: 0.0 for s in states}

def bellman_backup(state, values, transitions, goal, obstacles_set, gamma):
    """
    Compute the value and best action for a single state using current `values`.
    Returns (best_value, best_action)
    """
    best_action = None
    best_value = -float("inf")
    # for terminal (goal) we model it as an absorbing state whose value is 0 (no further reward)
    if state == goal:
        return 0.0, None

    for a, outcomes in transitions[state].items():
        action_value = 0.0
        for prob, s_next in outcomes:
            r = get_reward(state, a, s_next, goal, obstacles_set)
            action_value += prob * (r + gamma * values[s_next])
        if action_value > best_value:
            best_value = action_value
            best_action = a
    return best_value, best_action

def value_iteration(states, actions, transitions, goal, obstacles_set, gamma=0.9, theta=1e-6, max_iter=10000):
    """
    Run value iteration. Returns (values dict, num_iterations, converged Bool).
    """
    values = initialize_values(states)
    for i in range(1, max_iter+1):
        delta = 0.0
        new_values = values.copy()
        for s in states:
            v_old = values[s]
            v_new, _ = bellman_backup(s, values, transitions, goal, obstacles_set, gamma)
            new_values[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        values = new_values
        if delta < theta:
            return values, i, True
    return values, max_iter, False

def extract_policy(states, actions, transitions, values, goal, obstacles_set, gamma=0.9):
    """
    Return a dict policy[state] = best_action (string). For terminal (goal) -> None.
    """
    policy = {}
    for s in states:
        if s == goal:
            policy[s] = None
            continue
        best_action = None
        best_value = -float("inf")
        for a, outcomes in transitions[s].items():
            action_value = 0.0
            for prob, s_next in outcomes:
                r = get_reward(s, a, s_next, goal, obstacles_set)
                action_value += prob * (r + gamma * values[s_next])
            if action_value > best_value:
                best_value = action_value
                best_action = a
        policy[s] = best_action
    return policy

def visualize_grid(rows, cols, obstacles_set, start, goal, values, policy, out_dir="visuals"):
    """
    Save two plots:
     - heatmap of values (value_heatmap.png)
     - heatmap + arrows of policy (policy.png)
    """
    os.makedirs(out_dir, exist_ok=True)
    # create value matrix with shape (rows, cols)
    mat = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            mat[r, c] = values.get((r, c), 0.0)

    fig, ax = plt.subplots(figsize=(cols * 0.6 + 1, rows * 0.6 + 1))
    # Add value text in each cell
    for r in range(rows):
        for c in range(cols):
            val = values.get((r, c), 0.0)
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=10)
    im = ax.imshow(mat, origin="lower", interpolation="nearest")
    ax.set_title("Value function heatmap")
    plt.colorbar(im, ax=ax)
    # mark obstacles
    for (r, c) in obstacles_set:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black")
        ax.add_patch(rect)
    # mark start and goal
    ax.scatter([start[1]], [start[0]], marker="*", s=150, label="start", facecolors="none", edgecolors="green", linewidths=2)
    ax.scatter([goal[1]], [goal[0]], marker="o", s=120, label="goal", facecolors="none", edgecolors="red", linewidths=2)    
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(out_dir, "value_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    # Policy plot: arrows
    U = np.zeros((rows, cols))
    V = np.zeros((rows, cols))
    action_map = { "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1) }
    for r in range(rows):
        for c in range(cols):
            a = policy.get((r, c))
            if a is None:
                U[r, c] = 0.0
                V[r, c] = 0.0
            else:
                dr, dc = action_map[a]
                # quiver uses X=cols, Y=rows, so we put V for x component (col), U for y component (row)
                U[r, c] = dr
                V[r, c] = dc

    fig, ax = plt.subplots(figsize=(cols * 0.6 + 1, rows * 0.6 + 1))
    # Show value in each cell as text (like in the value heatmap)
    for r in range(rows):
        for c in range(cols):
            val = values.get((r, c), 0.0)
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=10)
    im = ax.imshow(mat, origin="lower", interpolation="nearest")
    # overlay obstacles again
    for (r, c) in obstacles_set:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black")
        ax.add_patch(rect)
    # arrow grid coordinates: X are columns, Y are rows
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    ax.quiver(X, Y, V, U, scale=1, scale_units='xy')
    # mark start/goal
    ax.scatter([start[1]], [start[0]], marker="*", s=150, label="start", facecolors="none", edgecolors="green", linewidths=2)
    ax.scatter([goal[1]], [goal[0]], marker="o", s=120, label="goal", facecolors="none", edgecolors="red", linewidths=2)
    ax.set_title("Policy (arrows) + value heatmap")
    plt.colorbar(im, ax=ax)
    plt.savefig(os.path.join(out_dir, "policy.png"), bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_cli_args()
    rows, cols = args.rows, args.cols
    states, obstacles_set, start, goal = create_state_space(rows, cols, args.obstacles, args.start, args.goal)
    actions = get_actions()
    transitions = build_transition_model(states, actions, rows, cols, slip_prob=args.slip)
    values, iters, converged = value_iteration(states, actions, transitions, goal, obstacles_set, gamma=args.gamma, theta=args.theta)
    policy = extract_policy(states, actions, transitions, values, goal, obstacles_set, gamma=args.gamma)

    print(f"Value iteration finished: iterations={iters}, converged={converged}")
    # Show value grid numerically
    for r in range(rows):
        rowvals = [f"{values[(r,c)]:6.3f}" for c in range(cols)]
        print(" ".join(rowvals))

    if args.visualize:
        visualize_grid(rows, cols, obstacles_set, start, goal, values, policy, out_dir="visuals")
        print("Saved visuals to ./visuals (value_heatmap.png, policy.png)")

if __name__ == "__main__":
    main()
