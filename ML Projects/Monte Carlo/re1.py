from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

import gym
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

from collections import defaultdict
matplotlib.style.use('ggplot')


def plot_value_function(V, title = 'Value Function'):
	min_x = min(k[0] for k in V.keys())
	max_x = max(k[0] for k in V.keys())
	min_y = min(k[1] for k in V.keys())
	max_y = max(k[1] for k in V.keys())    

	x_range = np.arange(min_x, max_x + 1)
	y_range = np.arange(min_y, max_y + 1)
	X, Y = np.meshgrid(x_range, y_range)

	Z_noace = np.apply_along_axis(lambda _:V[(_[0],_[1],False)], 2, np.dstack([X, Y]))
	Z_ace = np.apply_along_axis(lambda _:V[(_[0],_[1],True)], 2, np.dstack([X, Y]))


	def plot_surface(X, Y, Z, title):
		fig = plt.figure(figsize=(20,10))
		ax = fig.add_subplot(111, projection = '3d')
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
		ax.set_xlabel('Player Sum')
		ax.set_ylabel('Dealer Showing')
		ax.set_zlabel('Value')
		ax.set_title(title)
		ax.view_init(ax.elev, -120)
		fig.colorbar(surf)
		plt.show()


	plot_surface(X, Y, Z_noace, '(No Usable Ace) '+ title)
	plot_surface(X, Y, Z_ace, '(Usable Ace) '+ title)


def sample_policy(observation):
	score, dealer_score, usable_ace = observation
	return 0 if score >= 20 else 1



def print_observation(observation):
	score, dealer_score, usable_ace = observation
	print('Player score: %d | Usable Ace: %s | Dealer score: %d' %(score,usable_ace,dealer_score))


def test_enviorment(env, n_episodes=10):
	for i_episodes in range(n_episodes):
		observation = env.reset()
		while True:
			print_observation(observation)
			action = sample_policy(observation)
			print('Taking action: %s' %(["Stick", "Hit"][action]))
			observation, reward, done, _ = env.step(action)
			if done:
				print(observation)
				print('Game end.Reward: %.1f\n' % reward)
				break


env = gym.make('Blackjack-v0')
test_enviorment(env)


# Monte Carlo Prediction algorithm
def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)
	display_freq = num_episodes // 10


	V = defaultdict(float)

	for i_episode in range(1, num_episodes + 1):
		if i_episode % display_freq == 0:
			print('Episode (%d/%d)' %(i_episode, num_episodes))
			sys.stdout.flush()

			# Generate episode
			episode = []
			state = env.reset()
			while True:
				action = policy(state)
				next_state, reward, done, _ = env.step(action)
				episode.append((state,action,reward))
				if done:
					break
				state = next_state
				
			# find all states we ahve visited in this episode
			# we convert each state to a tuple so that we can use it as dict key
			states_in_episode = set([tuple(x[0]) for x in episode])
			for state in states_in_episode:
				# find first occurences of the state in the episode
				first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
				# sum up all rewards since the first occurences
				G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
				returns_sum[state] += G
				returns_count[state] += 1.0
				V[state] = returns_sum[state] / returns_count[state]

	return V


value_function = mc_prediction(sample_policy, env, num_episodes=2000000)				

plot_value_function(value_function)



