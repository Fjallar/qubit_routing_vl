import gym
import sys
sys.path.append("./rl-baselines3-zoo/")
import pycirc
from dqn2.dqn import DQN2
from stable_baselines3 import DQN
from copy import deepcopy, copy
from stable_baselines3.common.vec_env import (
	DummyVecEnv,
	SubprocVecEnv
)
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import torch as th
import yaml
import pickle
import cProfile

def get_predict_info_dqn2(model):
	env = model.env
	next_observations = env.env_method("get_next_observations")
	actions = np.zeros(env.num_envs, dtype = int)
	values =[]
	for i, (obs, action_idxs, rewards,_,_) in enumerate(next_observations):
		with th.no_grad():
			obs_th = th.Tensor(obs)
			obs_th = obs_th.to(model.device)
			# value = self._predict(tensor_obs, deterministic=deterministic)
			# q_values = model.q_net.forward(tensor_obs)
			# value = th.max(q_values,1).values
			value_q = model.q_net.forward(obs_th)
			value_q = value_q.cpu()
			value_q = value_q.reshape((th.numel(value_q)))
			value = value_q
			value+=rewards
		values.append((value, value_q, rewards))
		actions[i] = action_idxs[np.argmax(value)]
	
	return actions, values

def get_predict_info_dqn(model, obs=None):
	env = model.env
	if obs is None:
		obs = env.env_method("get_obs")
	with th.no_grad():
		obs_th= th.Tensor(obs)
		obs_th = obs_th.to(model.device)
		q_values = model.q_net.forward(obs_th)
		# Greedy action
		actions = q_values.argmax(dim=1).reshape(-1)

	return actions, q_values

def predict_(model):
	actions,_ = get_predict_info_dqn2(model)
	return actions




import os.path as path
#Have to fix this later:
#not sure how I got it to work before.
def step_model_dqn2(model):
	obs = model.env.reset()
	done=False
	while not done:
		print(obs)
		input("continue")
		actions, values = get_predict_info_dqn2(model)
		obs, reward, done,_ = model.env.step(actions)
		print(reward)
		print(values)
		print(actions)

def step_model_dqn(model):
	

	obs = model.env.reset()
	done=False
	while not done:
		print(obs)
		input("continue")
		actions, values = get_predict_info_dqn(model, obs)
		obs, reward, done,_ = model.env.step(actions)
		q_values, _, rewards = values[0]
		print(rewards)
		print(q_values)
		print(reward)
		print(actions)
def run_model(model):
	done=False
	episodes=10
	for _ in range(episodes):
		# print("nr_episodes: {}\nnr steps: {}".format(episodes, steps))
		done=False
		obs = model.env.reset()
		while not done:
			input("continue")
			actions, values = get_predict_info_dqn(model, obs)
			obs, reward, done,_ = model.env.step(actions)

def main(model_dir,env_kwargs=None):
	
	if env_kwargs is None:
		with open(f"{model_dir}qcirc-routing-v0/qcirc-routing-v0.yml") as f:
			env_kwargs = yaml.safe_load(f)
	 
	# if not path.exists('env.pkl'):
	env_id="qcirc-routing-v0"
	vec_env_class= {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}["subproc"]
	venv = make_vec_env(
			env_id=env_id,
			n_envs=1,
			seed=4080625631,
			env_kwargs=env_kwargs,
			monitor_dir="./logs/Data1/2022_09_28/run_11445__18252/checkpoints/DQN_7",
			wrapper_class=None,
			vec_env_cls=vec_env_class,
			vec_env_kwargs={},#'start_method': 'fork'}
			monitor_kwargs={}
		)
	# venv = DummyVecEnv([lambda: gym.make(env_id) for i in range(1)])

	model = DQN.load(model_dir+"best_model.zip",env=venv)
	# run_model(model)
	step_model_dqn2(model)


if __name__ == '__main__':

	if len(sys.argv) > 1:
		model_dir = sys.argv[1]
	else:
		print("Please provide a path to the directory of the model.")
		sys.exit(1)
	# cProfile.run('main()')
	# env_kwargs={'obs_params': {'obs_depth':6, 'nr_swap_layers': 0, 'layers_left_display_on': False, 'swap_layers_on': False}}
	# model_dir="rl-baselines3-zoo/logs/DataDir1/2023_03_03/run_30784__30758/checkpoints/DQN_1/"
	# main(model_dir,env_kwargs=env_kwargs)
	
	# model_dir="rl-baselines3-zoo/logs/TMP/2023_03_19/run_23051__29775/checkpoints/DQN_1/"
	# model_dir="rl-baselines3-zoo/logs/DataLog/2023_03_29/run_9944_VL_d4s2_44_a0b3_N8C4L_26277/checkpoints/DQN2_1/"
	main(model_dir)
