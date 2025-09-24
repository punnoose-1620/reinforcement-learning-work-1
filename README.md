# Reinforcement Learning Project

This is a Python project focused on reinforcement learning algorithms and implementations. Here is the problem statement we wish to address : 

Consider a simple grid world environment with a starting state, a goal state, and obstacles. The agent can move up, down, left, or right. The agent receives a reward of +1 upon reaching the goal state and a reward of -1 for hitting an obstacle. The agent’s goal is to find a policy that maximizes the expected cumulative reward.

- Define the MDP: Clearly define the states, actions, transition probabilities, and reward function for this grid world environment. In more detail: Clearly specify the states: Each grid cell represents a state. Define the actions: The agent can move up, down, left, or right. Determine the transition probabilities: For each state and action, specify the probability of landing in each possible next state. Consider factors like obstacles and edge cases (e.g., moving off the grid). Define the reward function: Assign rewards for reaching the goal, hitting obstacles, and taking any other action.

- Value Iteration: Implement the value iteration algorithm to compute the optimal value function for this grid world. In more detail: Write code to implement the value iteration algorithm. Initialize the value function for all states. Iteratively update the value function using the Bellman optimality equation until convergence. Use the right discount factor - for instance (γ = 0.9) as an initial guess.

- Policy Extraction: Extract the optimal policy from the computed value function. In more detail: Once the value function converges, determine the optimal action for each state based on the maximum expected future reward.

- Visualization: Visualize the grid world, the value function, and the optimal policy. In more detail: Create a visual representation of the grid world. Color-code the grid cells based on their value function. Indicate the optimal policy for each state using arrows or other symbols. 

To be exact, students should submit a PDF report, Python/JS code, and visualization. For the PDF report, the report should include:
- A detailed description of the grid world MDP, including states, actions, transition probabilities, and the reward function.
- A step-by-step explanation of the value iteration algorithm implementation.
- A clear description of how the optimal policy was extracted from the value function.

In addition, the code used to implement the value iteration algorithm. This should be included as a ZIP file with
the code, readme file, and requirements.txt with all the required library installs.
For the visualizations: 
1. a clear and informative visualization of the grid world, including obstacles, starting state, and goal state; 
2. A visualization of the computed value function for each state; and 
3. A visualization of the optimal policy, indicating the best action for each state.


## Project Structure

- `app.py` - Main application file implementing Grid-world MDP with Value Iteration
- `requirements.txt` - Python dependencies (numpy, matplotlib)
- `README.md` - Project documentation
- `Report.md` - Project report and findings
- `visuals/` - Folder containing visualization outputs
- `venv/` - Virtual environment with installed dependencies

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

The application implements a complete Grid-world MDP with Value Iteration algorithm. All functions are implemented and ready to use.

### Command Line Options

```bash
python app.py [options]
```

**Available Options:**
- `--rows ROWS` - Number of grid rows (default: 5)
- `--cols COLS` - Number of grid columns (default: 5)
- `--start R C` - Start cell coordinates (default: 0 0)
- `--goal R C` - Goal cell coordinates (default: 4 4)
- `--obstacles "r,c;r,c"` - Semicolon-separated obstacle coordinates
- `--slip SLIP` - Slip probability for unintended moves (default: 0.1)
- `--gamma GAMMA` - Discount factor (default: 0.9)
- `--theta THETA` - Value iteration convergence threshold (default: 1e-6)
- `--visualize` - Save visualizations to ./visuals folder

### Example Usage

```bash
# Basic 5x5 grid with default settings
python app.py

# Custom grid with obstacles and visualization
python app.py --rows 6 --cols 6 --start 0 0 --goal 5 5 --obstacles "1,1;2,2;3,3" --slip 0.1 --gamma 0.9 --theta 1e-4 --visualize

# Different grid configurations
python app.py --rows 4 --cols 3 --start 0 0 --goal 3 2 --obstacles "1,1;2,0"
```

### Expected Output

When you run the application, you'll see output like:

```
Value iteration finished: iterations=19, converged=True
 0.299  0.334  0.435  0.488  0.546  0.605
 0.334  0.385  0.433  0.544  0.613  0.683
 0.435  0.433  0.501  0.563  0.688  0.771
 0.488  0.544  0.563  0.689  0.776  0.871
 0.546  0.613  0.688  0.776  0.876  0.984
 0.605  0.683  0.771  0.871  0.984  0.000
Saved visuals to ./visuals (value_heatmap.png, policy.png)
```

### Testing Individual Functions

You can also test individual functions in Python:

```python
# Test CLI argument parsing
from app import parse_cli_args
args = parse_cli_args(['--rows','4','--cols','3','--start','0','0','--goal','3','2','--obstacles','1,1;2,0'])
print(args)

# Test action definitions
from app import get_actions
actions = get_actions()
print(actions)  # {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Test state space creation
from app import create_state_space
states, obstacles, start, goal = create_state_space(3,3,'1,1;2,0',(0,0),(2,2))
print(f"States: {len(states)}, Obstacles: {obstacles}")

# Test value iteration
from app import create_state_space, get_actions, build_transition_model, value_iteration
states, obs, start, goal = create_state_space(4,4,"1,1;2,2",(0,0),(3,3))
actions = get_actions()
trans = build_transition_model(states, actions, 4, 4, slip_prob=0.1)
values, iters, conv = value_iteration(states, actions, trans, goal, obs, gamma=0.9, theta=1e-4)
print(f"Iterations: {iters}, Converged: {conv}")
```

### Visualization Output

When using the `--visualize` flag, the application generates:
- `value_heatmap.png` - Heatmap showing value function for each state
- `policy.png` - Grid showing optimal policy with arrows indicating best actions

### Algorithm Features

- **Actions**: Four directional movements (up, down, left, right)
- **Slip Probability**: Configurable probability of unintended moves
- **Reward Structure**: 
  - +1.0 for reaching goal state
  - -1.0 for hitting obstacles
  - 0.0 for normal moves
- **Convergence**: Value iteration continues until value changes are below threshold
- **Policy Extraction**: Optimal policy derived from converged value function

## Dependencies

- numpy: For numerical computations
- matplotlib: For data visualization
