import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import pandas as pd
import shap  # Import SHAP library

# 1. Environment Setup
class CoinCollectingGame:
    def __init__(self, grid_size=5, num_coins=2):  # Corrected __init__ method
        self.grid_size = grid_size
        self.num_coins = num_coins
        self.reset()

    def reset(self):
        self.agent_position = (0, 0)  # Start position
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)  # Goal position
        self.coins = self.place_coins()  # Randomly place coins
        self.obstacles = self.place_obstacles()  # Randomly place obstacles
        self.steps_taken = 0
        self.total_reward = 0
        return self.get_state()  # Return initial state

    def place_coins(self):
        coins = set()
        while len(coins) < self.num_coins:
            coin = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if coin != self.agent_position and coin != self.goal_position:
                coins.add(coin)
        return coins

    def place_obstacles(self):
        obstacles = set()
        while len(obstacles) < 2:  # Place two obstacles
            obstacle = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if obstacle != self.agent_position and obstacle != self.goal_position and obstacle not in self.coins:
                obstacles.add(obstacle)
        return obstacles

    def step(self, action):
        # Update agent position based on action
        if action == 0:  # Up
            new_position = (max(0, self.agent_position[0] - 1), self.agent_position[1])
        elif action == 1:  # Down
            new_position = (min(self.grid_size - 1, self.agent_position[0] + 1), self.agent_position[1])
        elif action == 2:  # Left
            new_position = (self.agent_position[0], max(0, self.agent_position[1] - 1))
        elif action == 3:  # Right
            new_position = (self.agent_position[0], min(self.grid_size - 1, self.agent_position[1] + 1))

        # Check for collision with obstacles
        if new_position in self.obstacles:
            return self.get_state(), -10, True  # Lose a life (penalty)

        self.agent_position = new_position
        reward = 0

        # Check for coin collection
        if self.agent_position in self.coins:
            self.coins.remove(self.agent_position)
            reward = 10  # Reward for collecting a coin

        # Check if the agent has reached the goal
        done = self.agent_position == self.goal_position and len(self.coins) == 0

        self.steps_taken += 1
        self.total_reward += reward

        return self.get_state(), reward, done

    def get_state(self):
        return np.array(self.agent_position)  # Return agent position as state

    def render(self, episode, step):
        plt.figure(figsize=(5, 5))
        plt.xlim(-0.5, self.grid_size - 0.5)
        plt.ylim(-0.5, self.grid_size - 0.5)
        plt.grid()

        # Draw agent, goal, coins, and obstacles
        plt.text(self.agent_position[1], self.agent_position[0], 'A', fontsize=30, ha='center', va='center', color='green')
        plt.text(self.goal_position[1], self.goal_position[0], 'G', fontsize=30, ha='center', va='center', color='red')
        for coin in self.coins:
            plt.text(coin[1], coin[0], 'C', fontsize=30, ha='center', va='center', color='gold')
        for obs in self.obstacles:
            plt.text(obs[1], obs[0], 'X', fontsize=30, ha='center', va='center', color='blue')

        plt.gca().invert_yaxis()
        plt.title(f'Episode: {episode}, Step: {step}, Position: {self.agent_position}, Coins Left: {len(self.coins)}')
        plt.show(block=False)
        plt.pause(0.1)  # Short pause to visualize the update
        plt.clf()  # Clear the figure for the next rendering


# 2. DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 3. Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# 4. DQN Agent
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.update_target_network()
        self.memory = ReplayBuffer(max_size=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.q_values_history = []  # Track Q-values for plotting

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore: select random action
        else:
            with torch.no_grad():
                return self.model(state).argmax().item()  # Exploit: select best action

    def train(self, batch_size):
        if self.memory.size() < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in transitions:
            target = reward + self.gamma * self.target_model(next_state).max().item() * (1 - done)
            prediction = self.model(state)[0, action]
            loss = self.criterion(prediction, torch.tensor(target, dtype=torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Store the Q-value for visualization
        self.q_values_history.append(self.model(state).detach().numpy())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# 5. Training Function
def train_dqn(num_episodes, batch_size):
    env = CoinCollectingGame(grid_size=5, num_coins=2)
    agent = DQNAgent(input_dim=2, output_dim=4)
    rewards_per_episode = []
    log_data = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0

        for steps in range(100):  # Limit the number of steps per episode
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.choose_action(state_tensor)
            next_state, reward, done = env.step(action)
            agent.memory.add((state_tensor, action, reward, torch.FloatTensor(next_state).unsqueeze(0), done))
            agent.train(batch_size)

            state = next_state
            total_reward += reward

            log_data.append({
                'Episode': episode,
                'Step': steps,
                'Action': action,
                'Reward': reward,
                'Total Reward': total_reward
            })

            env.render(episode, steps)

        rewards_per_episode.append(total_reward)
        agent.decay_epsilon()

    # Plot rewards per episode
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid()
    plt.show()

    # Plot Q-values
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(agent.q_values_history).mean(axis=1))
    plt.xlabel('Training Steps')
    plt.ylabel('Average Q-Value')
    plt.title('Q-Value Progression')
    plt.grid()
    plt.show()

    # SHAP Values Calculation
    q_values_array = np.array(agent.q_values_history)
    if q_values_array.shape[1] == 4:  # Ensure there are four Q-values (for actions)
        background_data = q_values_array.mean(axis=0).reshape(1, -1)  # Using average Q-values as background
        explainer = shap.Explainer(agent.model)
        shap_values = explainer(q_values_array)
        shap.summary_plot(shap_values, feature_names=["Agent Position", "Coins Remaining"])


# Run training
if __name__ == "__main__":  # Corrected __name__ condition
    train_dqn(num_episodes=100, batch_size=32)
