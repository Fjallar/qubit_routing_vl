import numpy as np
from copy import deepcopy
from gym import Env
from gym.spaces import Discrete, Box
import gym


#Todos:
#   - Fix the assignent of the size of the generated circuit
#   - Find better ways of using tracker vatiables (e.g. self.n_layers_left, self.n_steps_taken and self.gates_solved)
#   - Add a SWAP circuit to store placement of the qubits
#   - Add the ability to create a quskit circuit from the current circuit
#   - Add a negative reward for performing an action that does nothing
#   - Fix so that you don't get the same permutation twice in a row when generating the circuit
#   - Make get next obs more efficient -> 

# class that describes a quantum circuit
class Circuit(Env):

	#initialize the circuit by the architecture
	def __init__(self, arch=[3,2], max_steps=100,
				 obs_params={'obs_depth':4, 'nr_swap_layers': 0, 'layers_left_display_on': False, 'swap_layers_on': False,  'extra_channel': False},
				 reward_params={'alpha':1, 'beta':0.2, 'gamma':0.03},
				 circuit_params={'density':1, 'depth':2, 'nr_qubits':6, 'init_swaps': False}
				 
				 ):
		self.reward_params = reward_params
		self.circuit_params = circuit_params

		
		self.max_steps = max_steps
		self.n_cols, self.n_rows= arch
		self.nr_qubits = self.n_cols*self.n_rows
		self.obs_params = obs_params
		self.actions, self.n_actions = self.get_actions()
		self.action_space = Discrete(self.n_actions+1)
		extra_channel = (1,) if self.obs_params['extra_channel'] else ()
		self.observation_space = Box(low=-self.nr_qubits, high=self.nr_qubits, shape=extra_channel+(self.obs_params['obs_depth'], self.n_rows, self.n_cols))

		#Specifies the state of the circuit        
		self.circ = []
		self.gate_is_swap = []
		self.current_layer=0
		self.n_steps_taken = 0
	
	def get_actions(self):
		actions = []
		for i in range(self.nr_qubits):
			if i//self.n_cols < self.n_rows-1:
				actions.append([i, i+self.n_cols])
			if i%self.n_cols < self.n_cols-1:
				actions.append([i, i+1])
		nr_actions = len(actions)
		return actions, nr_actions

	def reset(self):
		
		self.generate_circuit()

		self.n_layers = len(self.circ)
		self.n_layers_left = self.n_layers
		self.current_layer = 0
		self.n_steps_taken = 0
		self.gates_pr_layer = np.mean([len(layer) for layer in self.circ])

		return self.get_obs()
	
	#ugly code should change:
	def generate_circuit(self):
		depth = self.circuit_params['depth']
		nr_qubits = self.circuit_params['nr_qubits']
		density = self.circuit_params['density']
		self.circ = []
		self.gate_is_swap = []

		idx = np.round(2*(np.sqrt(nr_qubits)))

		axis1=np.floor(idx/2)
		axis2=np.ceil(idx/2)
		if (axis1+axis2)<(self.n_cols+self.n_rows):
			arch = [axis1, axis2]
			np.random.shuffle(arch)
		else:
			arch = [self.n_cols, self.n_rows]
		nr_qubit_gates = density*arch[0]*arch[1]//2


		pos_col, pos_row =  (self.n_cols-arch[0], self.n_rows-arch[1])
		top_corner = np.random.randint(0, (pos_col+1)*(pos_row+1))
		top_corner = np.unravel_index(top_corner, (pos_col+1,pos_row+1))
		top_corner = np.ravel_multi_index(top_corner, (self.n_cols, self.n_rows))

		for _ in range(depth):
			qubits = np.arange(arch[0]*arch[1])
			qubits = np.unravel_index(qubits, (arch[0], arch[1]))
			qubits = np.ravel_multi_index(qubits, (self.n_cols, self.n_rows))+top_corner
			np.random.shuffle(qubits)
			qubits = qubits.tolist()
			qubit_pairs = [[qubits.pop(), qubits.pop()] for i in range(nr_qubit_gates)]
			self.circ.append(qubit_pairs)
			self.gate_is_swap.append([False for _ in range(len(qubit_pairs))])
	 

	def get_full_overlap(self, qubits, layer):
		return [all(elem in qubits for elem in front_qubits) for front_qubits in layer]
	
	def get_overlap(self, qubits, layer):
		return [any(elem in qubits for elem in front_qubits) for front_qubits in layer]

	def place_swap(self, qubits, layer_i):

		#find the first layer that has no overlap with the qubits -> insert swap there/remove previous swap
		#otherwise create new layer (after loop)
		idx = layer_i
		while idx>0:
			is_swap, layer = self.gate_is_swap[idx-1], self.circ[idx-1]
			

			#remove swap if it overlaps completely with previous swap:
			full_overlap = self.get_full_overlap(qubits, layer)
			if any(full_overlap):
				index = full_overlap.index(True)
				if is_swap[index]:
					self.circ[idx-1].pop(index)
					self.gate_is_swap[idx-1].pop(index)
					if len(layer)==0:
						self.circ.pop(idx-1)
						self.gate_is_swap.pop(idx-1)
						self.current_layer-=1
					return
				else:
					break
			elif any(self.get_overlap(qubits, layer)):
				break
			idx-=1
		
		if idx ==layer_i:
			self.circ.insert(layer_i, [qubits])
			self.gate_is_swap.insert(layer_i,[True])
			self.current_layer+=1
		else:
			self.circ[idx].append(qubits)
			self.gate_is_swap[idx].append(True)



	def swap(self, qubits):

		#place the swap if not first layer, :
		if self.current_layer!=0:
			self.place_swap(qubits, self.current_layer)
		elif self.circuit_params['init_swaps']:
			self.place_swap(qubits, self.current_layer)
  
		#swap the qubits in the circuit
		for i in range(self.current_layer, len(self.circ)):
			for j in range(len(self.circ[i])):
				if self.circ[i][j][0] in qubits:
					self.circ[i][j][0] = qubits[1-qubits.index(self.circ[i][j][0])]
				if self.circ[i][j][1] in qubits:
					self.circ[i][j][1] = qubits[1-qubits.index(self.circ[i][j][1])]

	#function that calculates the one-norm of two qubits:
	def one_norm(self, qubits):
		qubit_vec = np.array([qubits[0]//self.n_cols-qubits[1]//self.n_cols, qubits[0]%self.n_cols-qubits[1]%self.n_cols])
		return np.sum(np.abs(qubit_vec))

	
	#function to check if the circuit is excecutable
	def is_layer_executable(self, layer):
		for qubits in layer:
			if self.one_norm(qubits) != 1:
				return False
		return True
	
	def executable_layers(self):
		return [self.is_layer_executable(layer) for layer in self.circ]
	
	def is_executable(self):
		return all(self.executable_layers())
	

	

	#compress circuit by pushing back connected qubit pairs to previous layers that do not have these qubits
	def compress_circuit(self,from_layer=0,to_layer=None):

		bool_circ = np.zeros((len(self.circ), self.nr_qubits), dtype=bool)
		for layer, bool_layer in zip(self.circ,bool_circ):
			for qubits in layer:
				bool_layer[qubits[0]] = True
				bool_layer[qubits[1]] = True                
		
		if to_layer is None:
			to_layer = len(self.circ)


		#compress circuit by pushing back connected qubit pairs to previous layers that do not have these qubits
		#the layers behind and in front of the current layer are compressed separately
		for layer_i, layer in zip(range(from_layer,to_layer),self.circ[from_layer:to_layer]):
			start_layer = self.current_layer if layer_i>=self.current_layer else 0
			if start_layer==layer_i:
				continue

			bool_layer = bool_circ[start_layer:layer_i,:]
			new_layers=[]
			for i, qubits in enumerate(layer):
				q1, q2 = qubits
				qbit_overlap = bool_layer[:,q1]|bool_layer[:,q2]
				overlap_idx = np.where(qbit_overlap)[0]
				new_layer = overlap_idx[-1]+1 if len(overlap_idx)>0 else 0
				new_layer+=start_layer
				if new_layer!=layer_i:
					new_layers.append((i,new_layer))
			
			for i, new_layer in reversed(new_layers):
				qubits = layer.pop(i)
				is_swap = self.gate_is_swap[layer_i].pop(i)
				self.circ[new_layer].append(qubits)
				self.gate_is_swap[new_layer].append(is_swap)


		#calculate current layer
		nr_empty_layers_before = len([layer for layer in self.circ[:self.current_layer] if len(layer)==0])
		self.current_layer-=nr_empty_layers_before

		#remove empty layers
		self.circ = [layer for layer in self.circ if len(layer)>0]
		self.gate_is_swap = [layer for layer in self.gate_is_swap if len(layer)>0]     
		
	def get_array_rep(self,start_layer=None, depth=None):
		depth = self.obs_params['obs_depth'] if depth is None else depth
		start_layer = self.current_layer-self.obs_params['nr_swap_layers'] if start_layer is None else start_layer

		if self.obs_params['nr_swap_layers']==0 and self.obs_params['layers_left_display_on']:
			start_layer -=1

		
		start_rep=0
		if start_layer < 0:
			start_rep=-start_layer
			start_layer=0

		array_rep = np.zeros((depth, self.n_rows, self.n_cols))

		for layer, is_swaps, array_layer in zip(self.circ[start_layer:], self.gate_is_swap[start_layer:], array_rep[start_rep:]):
			ravel_array = array_layer.ravel()
			for qubits, is_swap in zip(layer, is_swaps):
				sign = -1 if is_swap else 1
				ravel_array[qubits[0]] = sign*(qubits[1]+1)
				ravel_array[qubits[1]] = sign*(qubits[0]+1)
		if start_rep!=0:
			start_layer = min(-start_rep,start_layer)
		if self.obs_params['layers_left_display_on']:
			array_rep[-1,:,:] = max(0,len(self.circ)-(start_layer+depth))
		if self.obs_params['swap_layers_on']:
			array_rep[0,:,:] = max(0,start_layer)

		if self.obs_params['extra_channel']:
			array_rep = array_rep.reshape(1,*array_rep.shape)
		return array_rep
	
	#Pulls back the executable gates of the current to behind the current layer
	def pullback(self):

		is_executable = [self.one_norm(gate) == 1 for gate in self.circ[self.current_layer]]
		new_layer = [gates for gates, executable in zip(self.circ[self.current_layer], is_executable) if executable]
		self.circ[self.current_layer] = [gates for gates, executable in zip(self.circ[self.current_layer], is_executable) if not executable]
		new_gate_is_swap = [gate for gate, executable in zip(self.gate_is_swap[self.current_layer], is_executable) if executable]
		self.gate_is_swap[self.current_layer] = [gate for gate, executable in zip(self.gate_is_swap[self.current_layer], is_executable) if not executable]
		
		if len(new_layer)!=0:
			self.circ.insert(self.current_layer, new_layer)
			self.gate_is_swap.insert(self.current_layer, new_gate_is_swap)
			self.current_layer+=1
		
		self.gates_solved=len(new_layer)
	
	def get_obs(self):
		start_layer=self.current_layer-self.obs_params['nr_swap_layers']
		depth = self.obs_params['obs_depth']
		return self.get_array_rep(start_layer,depth)

	def get_reward(self,update=True):
		alpha = self.reward_params['alpha'] #proportion of reward for completing layers to qubits
		beta = self.reward_params['beta'] #beta * max_dist, is expected worst extra length per layer
		gamma = self.reward_params['gamma'] #punishement per step if circuit is not completed

		
		n_layers_left = len(self.circ)-self.current_layer

		#calculate increase in length and decrease in layers left
		new_layers_completed = self.n_layers_left-n_layers_left
		new_layers_add = len(self.circ)-self.n_layers

		if update:
			self.n_layers = len(self.circ)
			self.n_layers_left = n_layers_left

		reward = alpha*new_layers_completed-beta*new_layers_add
		reward += self.gates_solved*(1-alpha)/self.gates_pr_layer
		
		has_failed = self.n_steps_taken>=self.max_steps
		reward -= gamma if has_failed else 0
		return reward

	def is_done(self):
		has_failed = self.n_steps_taken>=self.max_steps
		has_completed = self.current_layer==len(self.circ)

		return has_failed or has_completed
	
	def is_state_changed(self,action):
		old_obs = self.get_obs()

		if action < len(self.actions):
			swap = self.actions[action]
			self.swap(swap)
			new_obs=self.get_obs()
			self.swap(swap)
			if not np.array_equal(old_obs,new_obs):
				return True
			else:
				return False
		elif action == len(self.actions):
			curr_circ = deepcopy(self.circ)
			curr_gate_is_swap = deepcopy(self.gate_is_swap)
			curr_current_layer = self.current_layer
			self.pullback()
			self.compress_circuit()
			new_obs=self.get_obs()
			self.circ = curr_circ
			self.gate_is_swap = curr_gate_is_swap
			self.current_layer = curr_current_layer
			if not np.array_equal(old_obs,new_obs):
				return True
			else:
				return False

	def get_action_mask(self):
		self.action_mask = np.array([self.is_state_changed(action) for action in range(self.n_actions+1)],dtype=bool)
		return self.action_mask

	def step(self,action):

		self.gates_solved=0
		#first layer where the action is not executable
		if action < len(self.actions):
			self.swap(self.actions[action])
		elif action == len(self.actions):
			self.pullback()
			self.compress_circuit()
		else:
			raise Exception("ValueError: Action not valid, action given: {}, nr_actions: {}".format(action, len(self.actions)))
		
		self.n_steps_taken+=1
		reward = self.get_reward()
		done = self.is_done()
		info = {}
		obs = self.get_obs()
		return obs, reward, done, info
	
	def get_next_observations(self):
		obs=[]
		rewards=[]
		infos=[{} for _ in range(len(self.actions)+1)]
		dones=[]
		act_idxs=[]
		last_obs = self.get_obs()
		for i, swap in enumerate(self.actions):
			self.swap(swap)
   
			new_obs=self.get_obs()
			if not np.array_equal(new_obs,last_obs):
				obs.append(new_obs)
				rewards.append(self.get_reward(update=False))
				dones.append(self.is_done())
				act_idxs.append(i)
    
			self.swap(swap)

		curr_circ = deepcopy(self.circ)
		curr_gate_is_swap = deepcopy(self.gate_is_swap)
		curr_current_layer = self.current_layer
		self.pullback()
		self.compress_circuit()
  
		new_obs=self.get_obs()
		if not np.array_equal(new_obs,last_obs):
			obs.append(new_obs)
			rewards.append(self.get_reward(update=False))
			dones.append(self.is_done())
			act_idxs.append(self.n_actions)
		self.circ = curr_circ
		self.gate_is_swap = curr_gate_is_swap
		self.current_layer = curr_current_layer

		return np.stack(obs,axis=0), act_idxs, np.array(rewards), dones, infos

gym.envs.register(
	id='qcirc-routing-v0',
	entry_point='pycirc:Circuit',
)



if __name__ == "__main__":
	circ = gym.make('qcirc-routing-v0', arch=[4,3])
	circ.circuit_params['depth'] = 10
	obs = circ.reset()
	done = False
	while not done:
		print(obs)
		action =int(input("take action (0-{}): ".format(len(circ.actions)-1)))
		
		obs, reward,_,_ =  circ.step(action)
		print("reward: {}".format(reward))

	circ.compress_circuit()
	print(circ.get_array_rep(start_layer=0, depth=len(circ.circ)))