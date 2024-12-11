import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
np.bool = bool  # Temporary fix for deprecated `np.bool`


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action_T, max_action_S):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action_T = max_action_T
        self.max_action_S = max_action_S

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        # Apply tanh activation for continuous actions and scale them by their respective max values
        throttle_action = torch.tanh(actions[:, 0]) * self.max_action_T
        steering_action = torch.tanh(actions[:, 1]) * self.max_action_S

        return throttle_action, steering_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class TD3:
    def __init__(self, state_dim, action_dim, max_action_T, max_action_S):
        self.actor = Actor(state_dim, action_dim, max_action_T, max_action_S)
        self.actor_target = Actor(state_dim, action_dim, max_action_T, max_action_S)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=1e-3)

        self.max_action_T = max_action_T
        self.max_action_S = max_action_S
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.1
        self.noise_clip = 0.2
        self.target_update_interval = 2

        # To store loss values for plotting later
        self.actor_losses = []
        self.critic_1_losses = []
        self.critic_2_losses = []
        self.episode_rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        throttle_action, steering_action = self.actor(state)
        return throttle_action.item(), steering_action.item()

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # Ensure action is 2D: batch_size x action_dim
        action = action.squeeze()

        # Critic loss calculation
        with torch.no_grad():
            # Generate noise with the same shape as actions
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            # Get next actions from the target actor
            throttle_action, steering_action = self.actor_target(next_state)

            # Combine actions - ensure it's 2D
            next_action = torch.stack([throttle_action, steering_action], dim=1)

            # Add noise to actions
            noisy_next_action = next_action + noise

            # Clamp the actions to their max values
            noisy_next_action = torch.clamp(noisy_next_action, 
                                            min=-max(self.max_action_T, self.max_action_S), 
                                            max=max(self.max_action_T, self.max_action_S))

            target_q1 = self.critic_1_target(next_state, noisy_next_action)
            target_q2 = self.critic_2_target(next_state, noisy_next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Ensure target_q is a 1D tensor
            target_q = reward + (1 - done) * self.discount * target_q.squeeze()

        current_q1 = self.critic_1(state, action).squeeze()
        current_q2 = self.critic_2(state, action).squeeze()
        
        # Ensure target_q and current_q have the same shape
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Store critic losses for plotting
        self.critic_1_losses.append(critic_1_loss.item())
        self.critic_2_losses.append(critic_2_loss.item())

        # Actor loss calculation
        throttle, steering = self.actor(state)
        combined_action = torch.stack([throttle, steering], dim=1)
        actor_loss = -self.critic_1(state, combined_action).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Store actor loss for plotting
        self.actor_losses.append(actor_loss.item())

        print(f"Actor Loss: {actor_loss.item()}, Critic 1 Loss: {critic_1_loss.item()}, Critic 2 Loss: {critic_2_loss.item()}")

        # Update target networks
        self.update_target(self.actor, self.actor_target)
        self.update_target(self.critic_1, self.critic_1_target)
        self.update_target(self.critic_2, self.critic_2_target)

    def update_target(self, model, target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)


class ActorWrapper(nn.Module):
    def __init__(self, actor, version_number=1.0):
        super(ActorWrapper, self).__init__()
        self.actor = actor
        self.version_number = version_number

    def forward(self, state):
        action = self.actor(state)
        # Add the "version_number" as a constant output
        return action, torch.tensor([self.version_number])

def save_onnx(agent, filename, state_dim):
    dummy_input = torch.randn(1, state_dim)
    wrapped_actor = ActorWrapper(agent.actor)
    
    torch.onnx.export(
        wrapped_actor,
        dummy_input,
        f"{filename}_actor.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action', 'version_number'],  # Add version_number to outputs
    )
    print(f"Actor model saved in ONNX format as {filename}_actor.onnx")


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return (
            torch.FloatTensor(state), 
            torch.FloatTensor(action).squeeze(), 
            torch.FloatTensor(next_state), 
            torch.FloatTensor(reward), 
            torch.FloatTensor(done)
        )

    def size(self):
        return len(self.buffer)


def train_td3(env, agent, num_episodes=10, batch_size=64, replay_buffer_size=1000, max_steps=200, state_normalization_params=None):
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    # List to store per-step rewards
    per_step_rewards = []

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")

        # Reset the environment and get initial decision steps
        env.reset()
        decision_steps, terminal_steps = env.get_steps(list(env.behavior_specs.keys())[0])
        behavior_name = list(env.behavior_specs.keys())[0]

        # Assume single agent for simplicity
        state = decision_steps[0].obs[0]

        # Normalize state if normalization parameters are provided
        if state_normalization_params:
            mean, std = state_normalization_params
            state = (state - mean) / std

        episode_reward = 0
        for step in range(max_steps):

            # Assuming 'action' is a numpy array or PyTorch tensor, you can ensure it is one by converting it:
            action = np.array(agent.select_action(state))  # Convert action to numpy array if it's not already

            # Now reshape it if necessary (for ensuring it's in the correct shape):
            action = action.reshape(1, -1)

            # Send action to the environment as a continuous action
            action_tuple = ActionTuple(continuous=action)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # Get updated decision and terminal steps
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:  # If the agent is still active
                next_state = decision_steps[0].obs[0]
                reward = decision_steps[0].reward
                done = False
            else:  # If the agent is done
                next_state = terminal_steps[0].obs[0]
                reward = terminal_steps[0].reward
                done = True

            # Store per-step reward
            per_step_rewards.append(reward)

            # Normalize next state if normalization parameters are provided
            if state_normalization_params:
                next_state = (next_state - mean) / std

            # Store the experience in the replay buffer
            replay_buffer.push(state, action, next_state, reward, done)

            # Update the current state
            state = next_state
            episode_reward += reward

            # Train the agent if enough samples are in the replay buffer
            if replay_buffer.size() >= batch_size:
                print(f"Replay Buffer Size: {replay_buffer.size()}")  # Debug replay buffer size
                agent.train(replay_buffer, batch_size)

            # Break the loop if the episode is done
            if done:
                break

        print(f"Episode {episode + 1}: Total Reward: {episode_reward:.2f}")
        agent.episode_rewards.append(episode_reward)

        print(f"Actor Losses Recorded: {len(agent.actor_losses)}, Critic 1 Losses Recorded: {len(agent.critic_1_losses)}, Critic 2 Losses Recorded: {len(agent.critic_2_losses)}")

    # Plot and save losses and rewards
    plot_losses_and_rewards(agent, per_step_rewards)


def plot_losses_and_rewards(agent, per_step_rewards):
    # Plot 1: Critic Losses
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(agent.critic_1_losses)), agent.critic_1_losses, label="Critic 1 Loss", color='blue')
    plt.title("Critic 1 Loss", fontsize=12)
    plt.xlabel("Training Steps", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(agent.critic_2_losses)), agent.critic_2_losses, label="Critic 2 Loss", color='green')
    plt.title("Critic 2 Loss", fontsize=12)
    plt.xlabel("Training Steps", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.savefig("critics_losses.png", dpi=300)
    plt.close()

    # Plot 2: Actor Loss, Per-Step Rewards, and Moving Average
    plt.figure(figsize=(18, 6))
    
    # Actor Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(range(len(agent.actor_losses)), agent.actor_losses, label="Actor Loss", color='red')
    plt.title("Actor Loss", fontsize=12)
    plt.xlabel("Training Steps", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Per-Step Rewards subplot
    plt.subplot(1, 3, 2)
    plt.plot(range(len(per_step_rewards)), per_step_rewards, label="Per-Step Rewards", color='purple')
    plt.title("Per-Step Rewards", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Reward", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Moving Average of Rewards
    plt.subplot(1, 3, 3)
    window_size = 100  # Adjust this value as needed
    rewards_moving_avg = np.convolve(per_step_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(len(rewards_moving_avg)), rewards_moving_avg, label=f"Rewards Moving Avg", color='orange')
    plt.title(f"Rewards Moving Average (Window={window_size})", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Average Reward", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.savefig("losses_and_rewards.png", dpi=300)
    plt.close()


def main():
    # # Initialize Unity environment
    # env = UnityEnvironment(file_name=None, base_port=5004, worker_id=0)
    unity_executable_path = "D:\RL Project\Build\AutonomousParking.exe"
    env = UnityEnvironment(file_name=unity_executable_path, base_port=5004, worker_id=0, no_graphics=True)
    env.reset()

    # Get behavior name
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"Behavior name: {behavior_name}")

    # Get behavior specs
    spec = env.behavior_specs[behavior_name]
    print(f"Observation space: {spec.observation_specs}")
    print(f"Action space: {spec.action_spec}")

    # Get state and action dimensions
    observation_shape = spec.observation_specs[0].shape
    state_dim = observation_shape[0]
    print(f"State dimension: {state_dim}")

    action_dim = spec.action_spec.continuous_size
    max_action_t = 50.0
    max_action_s = 20.0
    print(f"Action dimension: {action_dim}")

    # Initialize TD3 agent
    agent = TD3(state_dim, action_dim, max_action_t, max_action_s)

    # Train the agent using train_td3 function
    train_td3(
        env=env,
        agent=agent,
        num_episodes=1000,  # Number of episodes
        batch_size=64,  # Batch size for training
        replay_buffer_size=50000,  # Replay buffer capacity
        max_steps=750,  # Max steps per episode
        state_normalization_params=None  # Add normalization if needed
    )

    # Close the environment after training
    env.close()


if __name__ == "__main__":
    main()

