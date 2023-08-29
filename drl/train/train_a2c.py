
import os
import pandas as pd
import time
from stable_baselines3 import A2C
import numpy as np
from drl.env.env_pricing import DynamicPricingEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
import pickle

# load the model from disk
filename = 'classification_model/rf_model_simple.sav'
rf_model = pickle.load(open(filename, 'rb'))

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
              # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


test_raw = pd.read_csv('x_test.csv')
test = test_raw[ (test_raw['rateplan_type']=='Package') & (test_raw['base_price']<= test_raw['ref_price'])
               ].sample(8000, random_state=24).reset_index(drop=True)


test[test['is_booked']==1].groupby(['ab_test_bucket', 'is_booked']).size().plot.bar()
print(test[test['is_booked']==1].groupby(['ab_test_bucket','is_booked']).size())


logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

INDICATORS = ['base_price', 'base_cost', 'ref_price', 'margin_perc', 'ab_test_bucket', 'dow','checkin_month']
state_space = len(INDICATORS)

env_kwargs = {"state_space": state_space}
env = DynamicPricingEnv(df = test, booking_model = rf_model, a=0.1, **env_kwargs)


rl = 0.0007
n_steps = 10
policy_kwargs=dict(optimizer_class = RMSpropTFLike
                       , optimizer_kwargs = dict(weight_decay=0.001))


model_a2c = A2C('MlpPolicy'
                , env
                , verbose=1
                , n_steps = n_steps
                , learning_rate=rl
                , policy_kwargs=policy_kwargs
                , tensorboard_log=logdir
               )

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)


model_name = 'A2C'
models_dir = f"pricing_models/{model_name}/{int(time.time())}"

episodes = 150
TIMESTEPS = int(16000)
for ep in range(1, episodes):
    model_a2c.learn(total_timesteps=TIMESTEPS
                , callback=callback
                    , reset_num_timesteps=False
               ) # The total number of samples (env steps) to train on
    
    model_a2c.save(f"{models_dir}/{TIMESTEPS * ep}") 