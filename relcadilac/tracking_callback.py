import torch
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class TrackingCallback(BaseCallback):
    def __init__(self, total_timesteps, num_samples, verbose = 0): # , do_entropy_annealing=True, initial_entropy=0.4, min_entropy=0.005, cycle_length=10_000, damping_factor=0.5): danish add later
        super(TrackingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        # all rewards are negative of the bic - scaled due to reward normalization
        self.best_reward = -np.inf
        self.best_action = None  # the z vector
        self.pbar = None  # this is the progress bar
        self.num_samples = num_samples
        self.average_rewards = []  # in order to track the rewards
        # self.action_lengths = np.empty((total_timesteps,), dtype=np.float32)
        # self.action_cursor = 0  # to track the actions that have been inserted
        # entropy calculation variables

        # danish add later
        # self.do_entropy_annealing = do_entropy_annealing
        # # if do_entropy_annealing:  # danish -- add this later
        # self.initial_entropy = initial_entropy
        # self.min_entropy = min_entropy
        # self.cycle_length = cycle_length
        # self.damping_factor = damping_factor
        # self.curr_ent_coef = initial_entropy

    def _on_training_start(self) -> None:
        """ Initialize the progress bar. """
        self.pbar = tqdm(total=self.total_timesteps, desc="RL Training Progress", unit="step")
        # if self.verbose > 0 and self.do_entropy_annealing: danish add later
        #     self.pbar.write(f"Starting Entropy Coefficient Annealing: \n\t{self.initial_entropy = }\n\t{self.min_entropy = }\n\t{self.cycle_length = }\n\t{self.damping_factor = }")

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`.
        rewards = self.locals['rewards']
        self.average_rewards.append(np.mean(rewards))
        batch_best_idx = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_idx]
        curr_actions = self.locals['actions']
        if batch_best_reward > self.best_reward:
            self.best_reward = batch_best_reward
            self.best_action = curr_actions[batch_best_idx, :]
            if self.verbose > 0:
                self.pbar.write(f"Step: {self.num_timesteps}; New least BIC found: {- self.best_reward * self.num_samples}")
        if self.n_calls % 50 == 0:
            self.pbar.update(self.training_env.num_envs * 50)
        # if curr_actions is not None and self.action_cursor < self.total_timesteps:
        #     self.action_lengths[self.action_cursor] = np.mean(np.linalg.norm(curr_actions, axis=1), axis=0)
        #     self.action_cursor += 1
        # if self.do_entropy_annealing:  # danish add this later
        # e = min + 0.5 * (max - min) * (1 + cos(2 pi (t mod T)/ T)) * exp(-lambda * t / kT)

        # danish add later
        # current_step = self.num_timesteps
        # cycle_progress = (current_step % self.cycle_length) / self.cycle_length
        # cosine_val = 0.5 * (1 + np.cos(2 * np.pi * cycle_progress))
        # decay_val = np.exp(-self.damping_factor * (current_step / (self.cycle_length * 10)))
        # entropy_range = self.initial_entropy - self.min_entropy
        # new_ent_coef = self.min_entropy + (entropy_range * cosine_val * decay_val)
        # self.current_ent_coef = new_ent_coef
        # if hasattr(self.model, 'ent_coef'):
        #     self.model.ent_coef = torch.tensor(new_ent_coef, device=self.model.device)
        return True  # continue training

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
