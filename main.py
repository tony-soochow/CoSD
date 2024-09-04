import gym
from Brain import SACAgent,Pro_memory
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py
import re
from Brain.replay_memory import Memory

from Brain.obs_projection import (
    OBS_TRANSFORMS,
    RECORD_TRANSFORMS,
    obs_transform_default,
    record_transform_default,
)

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def transform_state(state):
    if obs_transform:
        return obs_transform(env, state)
    return state


if __name__ == "__main__":

    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"])

    env_prefix = re.split("-|_", params["env_name"])[0]
    obs_transform = OBS_TRANSFORMS.get(env_prefix, obs_transform_default)

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)
    
    z_pro_momery = {z: Pro_memory(params["mem_size"], params["seed"]) for z in range(params["n_skills"])}

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):
                t_state = transform_state(state) #转换当前state

                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                t_next_state = transform_state(next_state) #转换下一步state
                agent.store(state, z, done, action, next_state, t_state, t_next_state) 
                        
                projection = t_next_state - t_state
                #将z的投影添加到其对应的circular buffer
                z_pro_momery[z].add(projection)

                logq_zs = agent.train(z_pro_momery)
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break

            logger.log(episode,
                        episode_reward,
                        z,
                        sum(logq_zses) / len(logq_zses),
                        step,
                        np.random.get_state(),
                        env.np_random.get_state(),
                        env.observation_space.np_random.get_state(),
                        env.action_space.np_random.get_state(),
                        *agent.get_rng_states(),
                        )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
   

 