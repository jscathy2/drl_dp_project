from drl.env.env_pricing import DynamicPricingEnv

env_kwargs = {
    "min_p": 1,
    "max_p": 50,
    "h": 0.5,
    "w": 0.9,
    "mode": "continuous",
    "step_size": 1.0,
    "competitor": "twobound",
    "comp_params": {
    "min_p": 1,
    "max_p": 50,
    "diff": 1},
    "sim_params": {
    "params": [-3.89, -0.56, -0.01, 0.07, -0.03, 0],
    "factor": 800
    }    

}

env = DynamicPricingEnv(**env_kwargs)
episodes = 1
for episode in range(episodes):
    done = False
    obs = env.reset()
    while True: # not done
        random_action = env.action_space.sample()
        print("action", random_action)
        # np.asarray([self.ref_p, self.price_b]), profits_a, False, {}
        obs, reward, done, info = env.step(random_action)
        print('oberservation', obs)
        print('reward', reward)