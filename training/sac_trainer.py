"""SAC training and environment"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

from models.policy import ActorSAC, Critic


class LatentEnv:
    """Latent Space Environment"""
    def __init__(self, autoencoder, teacher_latent, teacher_windows,
                 reward_weights, device='cuda'):
        self.autoencoder = autoencoder
        self.teacher_latent = teacher_latent
        self.teacher_windows = teacher_windows
        self.device = device
        self.num_steps = len(teacher_latent) - 1
        
        self.w_latent, self.w_recon, self.w_dir, self.w_smooth, self.w_final = reward_weights
        
        self.current_step = 0
        self.z_current = None
        self.z_trajectory = []
    
    def reset(self):
        self.z_current = self.teacher_latent[0].clone()
        noise = torch.randn_like(self.z_current) * 0.005
        self.z_current = self.z_current + noise
        self.z_current = torch.clamp(self.z_current, -3.0, 3.0)
        self.current_step = 0
        self.z_trajectory = [self.z_current.clone()]
        return self.z_current.cpu().numpy()
    
    def step(self, action):
        delta_z = torch.FloatTensor(action).to(self.device)
        self.z_current = self.z_current + delta_z
        self.z_current = torch.clamp(self.z_current, -3.0, 3.0)
        
        z_norm = torch.norm(self.z_current)
        if z_norm > 3.0:
            self.z_current = self.z_current / z_norm * 3.0
        
        self.z_trajectory.append(self.z_current.clone())
        self.current_step += 1
        
        reward = self._compute_reward()
        done = self.current_step >= self.num_steps
        
        teacher_z = self.teacher_latent[self.current_step]
        info = {
            'step': self.current_step,
            'teacher_distance': torch.norm(self.z_current - teacher_z).item()
        }
        
        return self.z_current.cpu().numpy(), reward, done, info
    
    def _compute_reward(self):
        with torch.no_grad():
            teacher_z = self.teacher_latent[self.current_step]
            
            # Latent distance
            latent_dist = torch.norm(self.z_current - teacher_z).item()
            r_latent = -latent_dist * self.w_latent
            
            # Reconstruction
            x_recon = self.autoencoder.decoder(self.z_current.unsqueeze(0)).squeeze(0)
            target_window = self.teacher_windows[self.current_step]
            recon_error = F.mse_loss(x_recon, target_window).item()
            r_recon = -recon_error * self.w_recon
            
            # Direction
            if self.current_step > 0:
                teacher_dir = teacher_z - self.teacher_latent[self.current_step - 1]
                agent_dir = self.z_current - self.z_trajectory[-2]
                
                teacher_dir_norm = teacher_dir / (torch.norm(teacher_dir) + 1e-8)
                agent_dir_norm = agent_dir / (torch.norm(agent_dir) + 1e-8)
                
                cos_sim = torch.dot(teacher_dir_norm, agent_dir_norm).item()
                r_direction = cos_sim * self.w_dir
            else:
                r_direction = 0.0
            
            # Smoothness
            if self.current_step >= 2:
                delta1 = self.z_trajectory[-1] - self.z_trajectory[-2]
                delta2 = self.z_trajectory[-2] - self.z_trajectory[-3]
                accel = torch.norm(delta1 - delta2).item()
                r_smoothness = -accel * self.w_smooth if accel > 0.3 else 1.0
            else:
                r_smoothness = 0.0
            
            # Final bonus
            if self.current_step == self.num_steps:
                final_dist = latent_dist
                if final_dist < 0.3:
                    r_final = self.w_final
                elif final_dist < 0.5:
                    r_final = self.w_final * 0.5
                else:
                    r_final = 0.0
            else:
                r_final = 0.0
        
        reward = r_latent + r_recon + r_direction + r_smoothness + r_final
        return reward


class ReplayBuffer:
    """Replay Buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


def train_sac(env, state_dim, action_dim, action_scale, episodes=200,
              device='cuda', verbose=True):
    """Train SAC"""
    actor = ActorSAC(state_dim, action_dim, action_scale=action_scale).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    
    replay_buffer = ReplayBuffer(capacity=10000)
    
    gamma = 0.99
    tau = 0.005
    alpha = 0.05
    batch_size = 64
    
    episode_rewards = []
    episode_distances = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_distance = 0
        
        for step in range(env.num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = actor.sample(state_tensor)
            action = action.cpu().numpy()[0]
            
            next_state, reward, done, info = env.step(action)
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_distance += info['teacher_distance']
            state = next_state
            
            if len(replay_buffer) > batch_size:
                s_batch, a_batch, r_batch, ns_batch, d_batch = replay_buffer.sample(batch_size)
                
                s_batch = torch.FloatTensor(s_batch).to(device)
                a_batch = torch.FloatTensor(a_batch).to(device)
                r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                ns_batch = torch.FloatTensor(ns_batch).to(device)
                d_batch = torch.FloatTensor(d_batch).unsqueeze(1).to(device)
                
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(ns_batch)
                    q1_next, q2_next = critic_target(ns_batch, next_action)
                    q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                    q_target = r_batch + (1 - d_batch) * gamma * q_next
                
                q1, q2 = critic(s_batch, a_batch)
                critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                new_action, log_prob = actor.sample(s_batch)
                q1_new, q2_new = critic(s_batch, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_prob - q_new).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_distances.append(episode_distance / env.num_steps)
        
        if verbose and (episode + 1) % 50 == 0:
            print(f"  Episode {episode+1}/{episodes}, "
                  f"Reward: {np.mean(episode_rewards[-50:]):.2f}, "
                  f"Distance: {np.mean(episode_distances[-50:]):.4f}")
    
    return actor, episode_rewards, episode_distances
