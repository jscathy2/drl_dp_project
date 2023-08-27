from stable_baselines3.common.env_checker import check_env
from drl.env.env_pricing import DynamicPricingEnv

env = DynamicPricingEnv()

check_env(env)