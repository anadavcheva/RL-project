import os

import gym
import numpy as np

import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import hand_env
from gym.envs.robotics.utils import robot_get_obs

FINGERTIP_SITE_NAMES = [
    "robot0:S_fftip",
    "robot0:S_mftip",
    "robot0:S_rftip",
    "robot0:S_lftip",
    "robot0:S_thtip",
]

DEFAULT_INITIAL_QPOS = {
    "robot0:WRJ1": -0.16514339750464327,
    "robot0:WRJ0": -0.31973286565062153,
    "robot0:FFJ3": 0.14340512546557435,
    "robot0:FFJ2": 0.32028208333591573,
    "robot0:FFJ1": 0.7126053607727917,
    "robot0:FFJ0": 0.6705281001412586,
    "robot0:MFJ3": 0.000246444303701037,
    "robot0:MFJ2": 0.3152655251085491,
    "robot0:MFJ1": 0.7659800313729842,
    "robot0:MFJ0": 0.7323156897425923,
    "robot0:RFJ3": 0.00038520700007378114,
    "robot0:RFJ2": 0.36743546201985233,
    "robot0:RFJ1": 0.7119514095008576,
    "robot0:RFJ0": 0.6699446327514138,
    "robot0:LFJ4": 0.0525442258033891,
    "robot0:LFJ3": -0.13615534724474673,
    "robot0:LFJ2": 0.39872030433433003,
    "robot0:LFJ1": 0.7415570009679252,
    "robot0:LFJ0": 0.704096378652974,
    "robot0:THJ4": 0.003673823825070126,
    "robot0:THJ3": 0.5506291436028695,
    "robot0:THJ2": -0.014515151997119306,
    "robot0:THJ1": -0.0015229223564485414,
    "robot0:THJ0": -0.7894883021600622,
}

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("hand", "reach.xml")


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class HandReachEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(
            self,
            distance_threshold=0.01,
            n_substeps=20,
            relative_control=False,
            initial_qpos=DEFAULT_INITIAL_QPOS,
            reward_type="sparse",
    ):
        utils.EzPickle.__init__(**locals())
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        hand_env.HandEnv.__init__(
            self,
            MODEL_XML_PATH,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
            relative_control=relative_control,
        )

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[
            self.sim.model.body_name2id("robot0:palm")
        ].copy()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _sample_goal(self):
        thumb_name = "robot0:S_thtip"
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = self.np_random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = meeting_pos - goal[idx]
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if self.np_random.uniform() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        return goal.flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        goal = self.goal.reshape(5, 3)
        for finger_idx in range(5):
            site_name = "target{}".format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = "finger{}".format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = (
                    achieved_goal[finger_idx] - sites_offset[site_id]
            )
        self.sim.forward()


if __name__ == '__main__':
    env = HandReachEnv()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()

# import argparse
# import math
# import os
# import random
# import gym
# import numpy as np
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions import Normal
# from tensorboardX import SummaryWriter
# from lib.common import mkdir
# from lib.model import ActorCritic
# from lib.multiprocessing_env import SubprocVecEnv
#
#
# NUM_ENVS            = 8 #num of parallel envs
# ENV_ID              = "Humanoid-v2"
# HIDDEN_SIZE         = 64
# LEARNING_RATE       = 1e-3
# GAMMA               = 0.99 #discount factor
# GAE_LAMBDA          = 0.95 #smoothing factor
# PPO_EPSILON         = 0.2 #clip of the ratio
# CRITIC_DISCOUNT     = 0.5 # loss tends be bigger than actor, so we scale it down
# ENTROPY_BETA        = 0.001 # the amount of imporatence to give to the entropy bonus which helps exploration
# '''
# # number of transitions we sample for each training iteration, each step
# collects a transitoins from each parallel env, hence total amount of data
# collected = N_envs * PPOsteps --> buffer of 2048 data samples to train on
# '''
# PPO_STEPS           = 256
# MINI_BATCH_SIZE     = 64 # num of samples that are randomly  selected from the total amount of stored data
# '''one epoch means one PPO-epochs -- one epoch means one pass over the entire buffer of training data.
# So if one buffer has 2048 transitions and mini-batch-size is 64, then one epoch would be 32 selected mini batches.
# '''
# PPO_EPOCHS          = 10 # how many times we propagate the network over the entire buffer of training data
# TEST_EPOCHS         = 10 # how often we run tests to eval our network, one epoch is one entire ppo update cycle
# NUM_TESTS           = 10 # num of tests we run to average the total rewards, each time we want eval the performance of the network
# TARGET_REWARD       = 500
#
#
# def make_env():
#     ''' returns a function which creates a single environment '''
#     def _thunk():
#         env = gym.make(ENV_ID)
#         return env
#     return _thunk
#
#
# def test_env(env, model, device, deterministic=True):
#     '''
#     Training: sampling actions semi-randomly from the prob dist output by the network, so we get exploration
#     Testing: deterministic not random
#     functions runs for one episode and returns total reward
#     '''
#     state = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         dist, _ = model(state)
#         #continous action space instead of sampling based on the mean and stdf, we use means
#         action = dist.mean.detach().cpu().numpy()[0] if deterministic \
#             else dist.sample().cpu().numpy()[0]
#         next_state, reward, done, _ = env.step(action)
#         env.render()
#         state = next_state
#         total_reward += reward
#     return total_reward
#
#
# def normalize(x):
#     x -= x.mean()
#     x /= (x.std() + 1e-8)
#     return x
#
#
# def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
#     '''
#     mask is 0 if state is terminal, otherwise 1
#     '''
#     values = values + [next_value]
#     gae = 0
#     returns = []
#     for step in reversed(range(len(rewards))): #looping backwards from last step from the most recent experience to earlier
#         # bellman equaiton minus the value of the state and is essentially the same as the advantage
#         delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step] #if mask = 0, we use just reward as terminal state has no next_state
#         gae = delta + gamma * lam * masks[step] * gae # moving average of advantages discounted by gamma * gae lambda
#         # prepend to get correct order back
#         returns.insert(0, gae + values[step]) # add the value of the state we subtracted back in
#     return returns #ppo steps long list and env num wide
#
#
# def ppo_iter(states, actions, log_probs, returns, advantage):
#     '''generates random mini-batches until we have covered the full batch'''
#     #if update batch contains 2048 trajectories, and MINI_BATCH_SIZE=64, then 32 mini batches per epoch
#     batch_size = states.size(0)
#
#     for _ in range(batch_size // MINI_BATCH_SIZE):
#         rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
#         yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
#
#
# def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):
#     count_steps = 0
#     sum_returns = 0.0
#     sum_advantage = 0.0
#     sum_loss_actor = 0.0
#     sum_loss_critic = 0.0
#     sum_entropy = 0.0
#     sum_loss_total = 0.0
#
#     # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
#     for _ in range(PPO_EPOCHS):
#         # grabs random mini-batches several times until we have covered all data
#         for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
#
#             dist, value = model(state) #state into network to get latest prob dist and value of the state
#             entropy = dist.entropy().mean()
#             new_log_probs = dist.log_prob(action) # with each succesive update
#
#             # SURROGAGE POLICY LOSS in log space
#             # A long trajectory of experiences is collected at each update cycle
#             ratio = (new_log_probs - old_log_probs).exp()
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
#
#             actor_loss  = - torch.min(surr1, surr2).mean()
#             critic_loss = (return_ - value).pow(2).mean()
#             #Mean squared error between the actual GAE returns
#             #and network estimated value of the state
#
#             loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy
#             #from paper
#
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # track statistics
#             sum_returns += return_.mean()
#             sum_advantage += advantage.mean()
#             sum_loss_actor += actor_loss
#             sum_loss_critic += critic_loss
#             sum_loss_total += loss
#             sum_entropy += entropy
#
#             count_steps += 1
#
#     writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
#     writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
#     writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
#     writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
#     writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
#     writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)
#
#
# if __name__ == "__main__":
#     mkdir('.', 'checkpoints')
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
#     args = parser.parse_args()
#     writer = SummaryWriter(comment="ppo_" + args.name)
#
#     # Autodetect CUDA
#     use_cuda = torch.cuda.is_available()
#     device   = torch.device("cuda" if use_cuda else "cpu")
#     print('Device:', device)
#
#     # Prepare parallel nvironments
#     envs = [make_env() for i in range(NUM_ENVS)]
#     envs = SubprocVecEnv(envs)
#     env = gym.make(ENV_ID)
#     num_inputs  = envs.observation_space.shape[0]
#     num_outputs = envs.action_space.shape[0]
#
#     model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
#     print(model)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     print("parameters:",model.parameters())
#
#     frame_idx  = 0
#     train_epoch = 0 #one complte update cycle
#     best_reward = None
#
#     state = envs.reset() # 8 actions, 8 next states, 8 rewards, and 8 dones
#     early_stop = False
#
#     while not early_stop:
#         #storing training data
#         log_probs = []
#         values    = []
#         states    = []
#         actions   = []
#         rewards   = []
#         masks     = []
#
#         for _ in range(PPO_STEPS): #each ppo steps generates actions, states, rewards
#
#             state = torch.FloatTensor(state).to(device)
#
#             dist, value = model(state)
#
#             action = dist.sample()
#             # each state, reward, done is a list of results from each parallel environment
#             next_state, reward, done, _ = envs.step(action.cpu().numpy()) # really a lists of state foe each env
#             log_prob = dist.log_prob(action) #pass through the network
#
#             log_probs.append(log_prob)
#             values.append(value)
#             rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
#
#             masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
#
#             states.append(state)
#             actions.append(action)
#             # storing, each list will be len(PPO_steps) and contains a 8 wide list
#             state = next_state
#             frame_idx += 1
#
#         #run the final next state through the nework to get its value
#         next_state = torch.FloatTensor(next_state).to(device)
#         _, next_value = model(next_state)
#         returns = compute_gae(next_value, rewards, masks, values)
#
#         #256*8 = 2048
#         ### trajectory begin
#         returns   = torch.cat(returns).detach()
#         log_probs = torch.cat(log_probs).detach()
#         values    = torch.cat(values).detach()
#         states    = torch.cat(states)
#         actions   = torch.cat(actions)
#         ### trajectory end
#         #subtract returns from the network estimated values of each state
#         advantage = returns - values
#         advantage = normalize(advantage) #helps smooth training
#
#         ppo_update(frame_idx, states, actions, log_probs, returns, advantage)
#         train_epoch += 1
#
#         if train_epoch % TEST_EPOCHS == 0: #one test epoch is one entire update operation
#             #every few epochs we run a series of tests and average the rewards to see the agents performance
#             test_reward = np.mean([test_env(env, model, device) for _ in range(NUM_TESTS)])
#             writer.add_scalar("test_rewards", test_reward, frame_idx)
#             print('Frame %s. reward: %s' % (frame_idx, test_reward))
#             # Save a checkpoint every time we achieve a best reward
#             if best_reward is None or best_reward < test_reward:
#                 if best_reward is not None:
#                     print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
#                     name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
#                     fname = os.path.join('.', 'checkpoints', name)
#                     print(fname)
#                     torch.save(model, fname)
#                 best_reward = test_reward
#             if test_reward > TARGET_REWARD: early_stop = True