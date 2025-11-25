import numpy as np
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class TrackingCallback(BaseCallback):
    def __init__(self, total_timesteps, num_samples, verbose = 0):
        super(TrackingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        # all rewards are negative of the bic - scaled due to reward normalization
        self.best_reward = -np.inf
        self.best_action = None  # the z vector
        self.pbar = None  # this is the progress bar
        self.num_samples = num_samples
        self.average_rewards = []  # in order to track the rewards
        # self.action_values = np.empty((total_timesteps, self.training_env.num_envs, self.training_env.action_space.shape[0]), dtype=np.float32)
        # self.action_cursor = 0  # to track the actions that have been inserted

    def _on_training_start(self) -> None:
        """ Initialize the progress bar. """
        self.pbar = tqdm(total=self.total_timesteps, desc="RL Training Progress", unit="step")

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`.
        rewards = self.locals['rewards']
        infos = self.locals['infos']
        self.average_rewards.append(np.mean(rewards))
        batch_best_idx = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_idx]
        # curr_actions = self.locals['action_vector']
        if batch_best_reward > self.best_reward:
            self.best_reward = batch_best_reward
            self.best_action = infos[batch_best_idx]['action_vector']
            if self.verbose > 0:
                self.pbar.write(f"Num Calls * Num Envs: {self.n_calls * self.training_env.num_envs}; New least BIC found: {- self.best_reward * self.num_samples}")
        if self.n_calls % 50 == 0:
            self.pbar.update(self.training_env.num_envs * 50)
        # if curr_actions is not None and self.action_cursor < self.total_timesteps:
        #     self.action_buffer[self.cursor] = curr_actions
        #     self.cursor += 1
        return True  # continue training

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
