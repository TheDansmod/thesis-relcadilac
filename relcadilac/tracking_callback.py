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

    def _on_training_start(self) -> None:
        """ Initialize the progress bar. """
        self.pbar = tqdm(total=self.total_timesteps, desc="RL Training Progress", unit="step")

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`.
        rewards = self.locals['rewards']
        infos = self.locals['infos']
        batch_best_idx = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_idx]
        if batch_best_reward > self.best_reward:
            self.best_reward = batch_best_reward
            self.best_action = infos[batch_best_idx]['action_vector']
            if self.verbose > 0:
                self.pbar.write(f"New least BIC found: {- self.best_reward * self.num_samples}")
        if self.n_calls % 50 == 0:
            self.pbar.update(self.training_env.num_envs * 50)
        return True  # continue training

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
