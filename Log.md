# Development Log - Grid-world MDP with Value Iteration

This document tracks the step-by-step development process for implementing the Grid-world MDP with Value Iteration algorithm.

## Step 0: Project Setup and Virtual Environment

**Objective**: Set up the development environment and project structure.

- Create a project folder and the initial files
- Virtual environment activated and dependencies installed
- Ready for implementation

---

## Step 1: CLI Argument Parsing

**Objective**: Implement flow to capture grid world parameters from command-line interface for initial configuration.

- [ ] Parse grid dimensions (rows, cols)
- [ ] Parse start and goal positions
- [ ] Parse obstacle locations
- [ ] Parse algorithm parameters (slip, gamma, theta)
- [ ] Add visualization option
- Command-line interface allows flexible configuration
- All parameters have sensible defaults
- Obstacles specified as semicolon-separated coordinates

---

## Step 2: Action Definition

**Objective**: Define the available actions in the grid world. Centralizes action definitions and action deltas.

- [ ] Define four directional actions (up, down, left, right)
- [ ] Create action-to-movement mapping
- [ ] Handle action validation
- Actions represent discrete movements in the grid
- Each action has a corresponding (row, col) delta
- Actions are deterministic but affected by slip probability

---

## Step 3: State Space Creation

**Objective**: Create the state space from grid dimensions and handle obstacles. This helps in state management, parsing obstacle string, and validating inputs.

- [ ] Generate all valid grid positions
- [ ] Parse obstacle coordinates
- [ ] Validate start and goal positions
- [ ] Return state space components
- States are represented as (row, col) tuples
- Obstacles are excluded from valid states
- Start and goal positions must be valid states

---

## Step 4: Transition Model

**Objective**: Build the transition probability model for the MDP. The transition mapping and modelling all stochastic actions is handled here. Each action `1-slip_prob` slips equally to two actions each with `slip_prob/2`. If any intended move goes off-grid, the agent stays in place. This model allows entering obstacle cells.

- [ ] Calculate transition probabilities for each state-action pair
- [ ] Handle slip probability (unintended movements)
- [ ] Handle boundary conditions (grid edges)
- [ ] Handle obstacle collisions
- Transition model accounts for slip probability
- Boundary conditions prevent moving off the grid
- Obstacle collisions result in staying in current state

---

## Step 5: Reward Function

**Objective**: Define the reward structure for the MDP. The 'reward' is credited when entering a goal box or an obstacle box - positive for goal box and negative for obstacle box. We avoid redundant rewards by making sure `s != s_next` and `s_next == goal` for states.

- [ ] Implement reward for reaching goal state
- [ ] Implement penalty for hitting obstacles
- [ ] Implement step cost for normal moves
- [ ] Handle edge cases
- Goal state provides positive reward (+1.0)
- Obstacles provide negative reward (-1.0)
- Normal moves have zero or small negative reward

---

## Step 6: Value Initialization

**Objective**: Initialize the value function for all states. Set all states to `0.0` by default.

- [ ] Initialize value function for all states
- [ ] Set appropriate initial values
- [ ] Handle special cases (goal state)
- Value function typically initialized to zero
- Goal state may have special initialization
- Initial values affect convergence speed

---

## Step 7: Bellman Backup

**Objective**: Implement the Bellman equation for value updates. This is meant to compute the Bellman Optimal backup for single states using the current values. This expectedly returns the new values and the best action string.

- [ ] Calculate expected value for each action
- [ ] Find maximum expected value
- [ ] Update value function
- [ ] Return optimal action
- Bellman equation: V(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]
- Finds optimal action for given state
- Returns both value and action

---

## Step 8: Value Iteration Algorithm

**Objective**: Implement the main value iteration algorithm. This performs full value iteration using loops, iteratively applying Bellman Backups untill convergence.

- [ ] Iterate until convergence
- [ ] Update all state values
- [ ] Check convergence criteria
- [ ] Handle maximum iterations
- Algorithm continues until value changes are below threshold
- Maximum iterations prevent infinite loops
- Returns convergence information

---

## Step 9: Policy Extraction

**Objective**: Extract optimal greedy policy (mapping each `state -> best action`) from converged value function.

- [ ] Calculate optimal action for each state
- [ ] Build policy dictionary
- [ ] Handle ties in action selection
- [ ] Validate policy completeness
- Policy maps each state to its optimal action
- Ties in action values handled consistently
- Policy should be deterministic

---

## Step 10: Visualization

**Objective**: Create visualizations of the grid world, value function, and policy. Performed using a heatmap to visualize values and overlay optimal policy using arrows and use markings for goal, start and obstacles.

- [ ] Create grid world visualization
- [ ] Create value function heatmap
- [ ] Create policy visualization with arrows
- [ ] Save visualizations to files
- Visualizations saved as PNG files
- Heatmap shows value function values
- Policy visualization shows optimal actions
- Files saved in specified output directory

---

## Step 11: Main Function Integration

**Objective**: Basically integrate all functions previously described from previous steps into a complete application. OFC, this has been done iteratively and refined over several attempts and what you see is the current version only.

- [ ] Parse command-line arguments
- [ ] Set up grid world environment
- [ ] Run value iteration algorithm
- [ ] Extract and display policy
- [ ] Generate visualizations if requested
- [ ] Display results
- Main function orchestrates entire process
- Handles all command-line options
- Provides comprehensive output
- Integrates all previous components

---

## Development Summary

**Total Steps**: 11
**Implementation Status**: Complete
**Testing Status**: [To be filled]
**Documentation Status**: [To be filled]

**Key Achievements**:
- [ ] Complete MDP implementation
- [ ] Value iteration algorithm
- [ ] Policy extraction
- [ ] Visualization capabilities
- [ ] Command-line interface
- [ ] Comprehensive testing

**Next Steps**:
- [ ] Performance optimization
- [ ] Additional algorithms (Policy Iteration, Q-Learning)
- [ ] Extended visualization options
- [ ] Documentation improvements
