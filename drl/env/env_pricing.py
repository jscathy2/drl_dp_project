from __future__ import annotations

from typing import List
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
import math
import pickle

class DynamicPricingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, 
                 df: pd.DataFrame, 
                 state_space: int,
                 booking_model,
                 a = 0.5,
                 *arg,
                 **kwargs
                 ):
        super(DynamicPricingEnv, self).__init__()
        self.df = df
        self.booking_model = booking_model
        self.done = False
        self.a = a

        self.state_space = state_space
        self.num_timesteps = len(self.df)
        self.current_step = 0
        self.reward = 0

        # Action space: price adjustment between -10% and 10% of base price
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
            )
        
        # Observation space: base price, max price, cost, booking day of week, 
        # checkin date, checkout date
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(self.state_space,), dtype=np.float32
            )
         
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.state = [self.df['base_price'][self.current_step], 
                      self.df['base_cost'][self.current_step],
                      self.df['ref_price'][self.current_step], 
                      self.df['margin%'][self.current_step],
                      self.df['ab_test_bucket'][self.current_step], 
                    #   self.df['rateplan_type'][self.current_step],
                      self.df['dow'][self.current_step],
                    #   self.df['chain'][self.current_step],
                      self.df['checkin_month'][self.current_step]
                      ]

        return self.state, {}

    def step(self, action):
        # Adjust the price based on action, unornalise from -1,1 to -0.05, 0.05
        action_unnorm = action[0] / 20
        base_price = self.df.loc[self.current_step, 'base_price']
        
        # profit cannot be negative, min price = cost
        min_price = self.df.loc[self.current_step, 'base_cost']
        max_price =  self.df.loc[self.current_step, 'ref_price']
        adjusted_price = base_price + base_price * action_unnorm
        adjusted_price = np.clip(adjusted_price, min_price, max_price)
       
        ab_test_bucket = self.get_bucket(action_unnorm)
        # rateplan_type = self.df.loc[self.current_step, 'rateplan_type']
        dow = self.df.loc[self.current_step, 'dow']
        # chain = self.df.loc[self.current_step, 'chain']
        checkin_month = self.df.loc[self.current_step, 'checkin_month']

        # Compute margin
        margin = adjusted_price - min_price
        margin_base = base_price - min_price

        margin_perc = margin / adjusted_price
        margin_perc_base = margin_base / base_price

        
        X_test = pd.DataFrame([[margin_perc, ab_test_bucket, dow, checkin_month]]
                              , columns=['margin%', 'ab_test_bucket', 'dow', 'checkin_month'])
        
        X_test_base = pd.DataFrame([[margin_perc_base, 0, dow, checkin_month]]
                              , columns=['margin%', 'ab_test_bucket', 'dow', 'checkin_month'])
        
        booking_prob_new_margin =  self.booking_model.predict_proba(X_test)[::,1]
        booking_prob_base_margin = self.booking_model.predict_proba(X_test_base)[::,1]

        booking_prob_new_price =  self.booking_probability(adjusted_price, noise_std=0.05)
        booking_prob_base_price = self.booking_probability(base_price, noise_std=0.05)

        reward = (self.a * margin * booking_prob_new_margin + (1-self.a) * (margin * booking_prob_new_price) 
                  - self.a * margin_base * booking_prob_base_margin - (1-self.a) * (margin_base * booking_prob_base_price)
                )

        self.state = [adjusted_price, min_price, max_price, margin_perc
                      , ab_test_bucket, dow, checkin_month
                      ]
        
        self.current_step += 1
        done = self.current_step >= self.num_timesteps

        return self.state, reward, done, False, {}

    def get_bucket(self, action):

        if action>= 0:
            return 0

        elif  (action < 0) &  (action >= -0.02):
            return 1

        elif  (action < -0.02) &  (action >= -0.03):
            return 2
            
        return 3

    
    def booking_probability(self, price, noise_std=0.05):
        """
        Returns the booking probability given a price.
        This uses a sigmoid function and is a simplistic model.
        You might want to replace this with a more sophisticated function or use actual data to derive it.
        """
        base_price = self.df.loc[self.current_step, 'base_price']
        price_sensitivity = 0.45  # You can adjust this parameter 0.05 if we can adjust +-100%
        sigmoid_value = 0.205 + 0.205 * (1 / (1 + math.exp(price_sensitivity * (price - base_price))))
    
        # Add Gaussian noise
        noisy_sigmoid = sigmoid_value + np.random.normal(0, noise_std)
    
        # Since it's a probability, clip to ensure it's between 0 and 1
        return np.clip(noisy_sigmoid, 0, 1)
    

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs