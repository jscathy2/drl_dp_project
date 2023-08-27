#!/usr/bin/env python
# coding: utf-8

# Hyperparam
# 
# A2C 7: Default 
# A2C 8: learning rate 0.0003
# <!-- A2C 9: learning rate 0.0007, decay 0.9
# A2C 10: learning rate 0.002, decay 0.98 -->
# A2C 11: learning rate 0.0007, decay 0.001
# A2C 12 = ppo steps=10: learning rate 0.0007, decay 0.001

# In[1]:


import os
import pandas as pd
import time
from stable_baselines3 import PPO, A2C
import numpy as np

# from finrl.meta.env_pricing.env_pkg_pricing import MarketEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from finrl.meta.env_pricing.env_pkg_pricing_v2 import DynamicPricingEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


# In[2]:


import gym
from gym import spaces
import numpy as np

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
# from stable_baselines3.ddpg.policies import LnMlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


# In[3]:


import pickle


# In[4]:


# load the model from disk
filename = 'classification_model/rf_model_simple.sav'
rf_model = pickle.load(open(filename, 'rb'))


# #todo
# 1. filter package rate only
# 2. a = 0.1
# 3. action norm -1, 1
# 4. learning rate

# In[5]:


# class PrintRewardCallback(BaseCallback):
#     def __init__(self, check_freq: int, verbose=1):
#         super(PrintRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.episode_rewards = []

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             mean_reward = np.mean(self.episode_rewards[-self.check_freq:])
#             print(f"Step: {self.num_timesteps}, Mean Reward: {mean_reward}")
#         return True

#     def _on_rollout_end(self) -> None:
#         self.episode_rewards.append(self.locals["ep_info_buf"].mean())

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

# class SaveCheckpoint(BaseCallback):
#     def __init__(self, save_freq, verbose = 0):
#         super(SaveCheckpoint, self).__init__(verbose)
#         self.save_freq = save_freq

#     def _on_step(self):
#         if self.num_timesteps % self.save_freq == 0:
#             self.model.save("model.zip")
#             self.training_env.save("stats.pkl")

#         return True


# In[6]:


# train = pd.read_csv('train_df.csv')[['TRANS_DATE_KEY','base_price','max_price','cost']]
# train['TRANS_DATE_KEY2']= pd.to_datetime(train['TRANS_DATE_KEY'])
# train['dow'] = train['TRANS_DATE_KEY2'].dt.dayofweek
# test = pd.read_csv('test_df.csv')[['TRANS_DATE_KEY','base_price','max_price','cost']]
# test['TRANS_DATE_KEY2']= pd.to_datetime(test['TRANS_DATE_KEY'])
# test['dow'] = test['TRANS_DATE_KEY2'].dt.dayofweek
# test = test[['base_price','max_price','cost','dow']]

test_raw = pd.read_csv('x_test.csv')
test = test_raw[ (test_raw['rateplan_type']=='Package') & (test_raw['base_price']<= test_raw['ref_price'])
               ].sample(8000, random_state=24).reset_index(drop=True)
test.tail()


# In[7]:


test[test['is_booked']==1].groupby(['ab_test_bucket', 'is_booked']).size().plot.bar()
print(test[test['is_booked']==1].groupby(['ab_test_bucket','is_booked']).size())


# In[8]:


len(test)


# In[9]:


test.isnull().sum()


# In[10]:


# model_name = 'PPO'
# models_dir = f"pricing_models/{model_name}/{int(time.time())}"
logdir = "logs"

# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)

# if not os.path.exists(logdir):
#     os.makedirs(logdir)
    
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


# In[11]:


INDICATORS = ['base_price', 'base_cost', 'ref_price', 'margin_perc', 'ab_test_bucket', 'dow','checkin_month']
state_space = len(INDICATORS)
print(state_space)


# In[12]:



# policy_kwargs["optimizer_class"] = RMSpropTFLike
# policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)


# In[13]:


env_kwargs = {"state_space": state_space}
env = DynamicPricingEnv(df = test, booking_model = rf_model, a=0.1, **env_kwargs)
# env = Monitor(env, log_dir)


# In[17]:


# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[128, 128], vf=[128, 128]))

rl = 0.0007
# 0.0007
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


# In[12]:


# timesteps = int(1e6)
# model_ppo.learn(total_timesteps=int(timesteps), callback=callback)


# In[18]:


model_name = 'A2C'
models_dir = f"pricing_models/{model_name}/{int(time.time())}"

episodes = 150
TIMESTEPS = int(16000)
for ep in range(1, episodes):
    model_a2c.learn(total_timesteps=TIMESTEPS
                , callback=callback
                    , reset_num_timesteps=False
               ) # The total number of samples (env steps) to train on
# learn(total_ti
# mesteps, callback=None, 
# log_interval=1, 
# tb_log_name='PPO', 
# reset_num_timesteps=True, 
# progress_bar=False)
    model_a2c.save(f"{models_dir}/{TIMESTEPS * ep}") 


# In[24]:


env.close()


# In[29]:


df = pd.read_csv('X_test.csv')
test = df.sample(frac=0.2, random_state=24)
test_new = df[~df.index.isin(test)][['base_price', 'base_cost', 'ref_price', 'margin%', 'ab_test_bucket', 'dow','checkin_month']]


# In[123]:


df[df['base_cost'] == 1095.7113648]


# In[30]:


test_new.tail()


# In[19]:


test.head()


# In[17]:


test.head()


# In[73]:


model_path = 'pricing_models/PPO/1692489821/100000.zip'

ppo_best_model = PPO.load(model_path, env = env)


# In[55]:


model_path = 'pricing_models/A2C/1692489929/100000.zip'

a2c_best_model = A2C.load(model_path, env = env)


# In[149]:


action, _states = ppo_best_model.predict(test_new[10:11])


# In[150]:


obs, rewards, done, _,info = env.step(action)


# In[151]:


print(obs, rewards)


# In[152]:


test_new[test_new['base_cost'] == 179.28]


# In[158]:


margin_base/224.10


# In[171]:


magin = 235.30501 - 179.28
margin_base = 224.10- 179.28

prob_m = 0.32480558
prob_base_m = 0.29494006

prob_p = 0.0225 *10
prob_base_p = 0.03 *10

a = 0.45

re = a*(magin * prob_m) + (1-a)*(magin * prob_p) - (a * margin_base * prob_base_m + (1-a)*(magin * prob_base_p))


# In[172]:


(magin * prob_m)


# In[173]:


(magin * prob_p)


# In[174]:


margin_base * prob_base_m


# In[175]:


magin * prob_base_p


# In[176]:


re


# In[107]:


obs


# In[98]:


rewards[0]


# In[93]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(action)
plt.show()


# In[74]:


len(test)


# In[53]:


test.head()


# In[ ]:


agent = DRLAgent(env = env_train)
# Set the corresponding values to 'True' for the algorithms that you want to use
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True


# In[ ]:


agent = DRLAgent(env = env_train)
model_a2c = agent.get_model("a2c")

if if_using_a2c:
  # set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)

