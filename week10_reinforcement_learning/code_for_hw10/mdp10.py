import pdb
import random
import numpy as np
from dist import uniform_dist, delta_dist, mixture_dist, DDist
from util import argmax_with_val, argmax
from keras.api.layers import Dense
from keras.api.models import Sequential
from keras.api.optimizers import Adam

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn,
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

    def state2vec(self, s):
        '''
        Return one-hot encoding of state s; used in neural network agent implementations
        '''
        v = np.zeros((1, len(self.states)))
        v[0,self.states.index(s)] = 1.
        return v

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, max_iters = 1000):
    # Your code here (COPY FROM HW9)
    q_old = q
    for iter in range(max_iters):
        q_new = q_old.copy()
        max_diff = 0
        for s in mdp.states:
            for a in mdp.actions:
                q_new.set(s,a,mdp.reward_fn(s,a) + mdp.discount_factor * mdp.transition_model(s,a).expectation(lambda s: value(q_old,s)))
                max_diff = max(max_diff, np.abs(q_new.get(s,a) - q_old.get(s,a)))
        if max_diff < eps:
            return q_new
        q_old = q_new
    return q_new

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    # Your code here (COPY FROM HW9)
    return max([q.get(s,a) for a in q.actions])

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    # Your code here (COPY FROM HW9)
    return argmax(q.actions, lambda a: q.get(s,a))

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        # Your code here (COPY FROM HW9)
        return uniform_dist(q.actions).draw()
    else:
        # Your code here (COPY FROM HW9)
        return greedy(q,s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
    def update(self, data, lr):
        # Your code here
        for s, a, t in data:
            self.set(s, a, (1-lr)*self.get(s,a) + lr*t)

def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
    # Your code here
    s = mdp.init_state()
    for i in range(iters):
        a = epsilon_greedy(q,s,eps)
        r,s_prime = mdp.sim_transition(s,a)
        gamma = 0 if mdp.terminal(s) else mdp.discount_factor
        q.update([(s,a,r+gamma*value(q,s_prime))], lr)
        s=s_prime
        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
    return q

# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we find
# a terminal state, end the episode.  Return accumulated reward a list
# of (s, a, r, s') where s' is None for transition from terminal state.
# Also return an animation if draw=True.
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(int(episode_length)):
        a = policy(s)
        (r, s_prime) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            break
        episode.append((s, a, r, s_prime))
        if draw: 
            mdp.draw_state(s)
        s = s_prime
        all_states.append(s)
    animation = animate(all_states, mdp.n, episode_length) if draw else None
    return reward, episode, animation

# Create a matplotlib animation from all states of the MDP that
# can be played both in colab and in local versions.
def animate(states, n, ep_length):
    try:
        from matplotlib import animation, rc
        import matplotlib.pyplot as plt
        from google.colab import widgets

        plt.ion()
        plt.figure(facecolor="white")
        fig, ax = plt.subplots()
        plt.close()

        def animate(i):
            if states[i % len(states)] == None or states[i % len(states)] == 'over':
                return
            ((br, bc), (brv, bcv), pp, pv) = states[i % len(states)]
            im = np.zeros((n, n+1))
            im[br, bc] = -1
            im[pp, n] = 1
            ax.cla()
            ims = ax.imshow(im, interpolation = 'none',
                        cmap = 'viridis', 
                        extent = [-0.5, n+0.5,
                                    -0.5, n-0.5],
                        animated = True)
            ims.set_clim(-1, 1)
        rc('animation', html='jshtml')
        anim = animation.FuncAnimation(fig, animate, frames=ep_length, interval=100)
        return anim
    except:
        # we are not in colab, so the typical animation should work
        return None

# Return average reward for n_episodes of length episode_length
# while following policy (a function of state) to choose actions.
def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes

# def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
#                   episode_length=10, n_episodes=2,
#                   interactive_fn=None):
#     # Your code here
#     all_experiences = []
#     for i in range(iters):
#         for e in range(n_episodes):
#             _, episode, _ = sim_episode(mdp,episode_length,lambda s:epsilon_greedy(q,s,eps), draw=False)
#             all_experiences += episode
#         all_q_targets = []
#         for s,a,r,s_prime in all_experiences:
#             if s_prime is not None:
#                 all_q_targets.append((s,a,r+mdp.discount_factor*value(q,s_prime)))
#             else:
#                 all_q_targets.append((s,a,r))
#         q.update(all_q_targets, lr)
#         # include this line in the iteration, where i is the iteration number
#         if interactive_fn: interactive_fn(q, i)
#     return q
def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                               episode_length=10, n_episodes=2,
                  interactive_fn=None):
    all_experiences = []
    explore = lambda s: epsilon_greedy(q,s,eps)
    for i in range(iters):
        for e in range(n_episodes):
            _, episode, _ = sim_episode(mdp, episode_length, explore)
            all_experiences += episode
        all_q_targets = []
        for (s, a, r, s_prime) in all_experiences:
            future_val = 0 if s_prime is None else value(q, s_prime)
            all_q_targets.append((s, a, (r + mdp.discount_factor * future_val)))
        q.update(all_q_targets, lr)
        if interactive_fn: interactive_fn(q, i)
    return q


def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

class NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.state2vec = state2vec
        self.epochs = epochs
        # Your code here
        state_dim = state2vec(states[0]).shape[1]
        self.models = {a:make_nn(state_dim, num_layers, num_units) for a in actions}
    def get(self, s, a):
        # Your code here
        return self.models[a].predict(self.state2vec(s)).item()
    def update(self, data, lr):
        # Your code here
        actions = set([a for s,a,t in data])
        for action in actions:
            model = self.models[action]
            x = np.row_stack([self.state2vec(s) for s,a,t in data if a == action])
            y = np.row_stack([t for s,a,t in data if a == action])
            model.fit(x, y, epochs=self.epochs, verbose=False)
