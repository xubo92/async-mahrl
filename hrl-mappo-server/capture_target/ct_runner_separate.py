# ct_runner_separate: runner for capture target
# this runner file uses multiple separate policy networks

import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

import copy
from misc import logger

def _t2n(x):
    return x.detach().cpu().numpy()

class CtRunnerSeparate(Runner):
    def __init__(self, config):
        super(CtRunnerSeparate, self).__init__(config)
        assert self.all_args.scheme in ["fully-dec", "partial-dec", "partial-cen", "sync-wait", "sync-cut"]
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_tracker = None   # XL: add a info tracker to track unfinished, old actions
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                
                if self.all_args.scheme == "fully-dec":
                    # XL: Some tricky things (important)
                    if info_tracker is not None:
                        for i in range(info_tracker.shape[1]):  # if single env collecting data
                            if not info_tracker[0, i]["act_finished"]:
                                actions[0, i, 0] = info_tracker[0, i]["act"]
                                action_log_probs[0, i, 0] = 0
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                available_actions = None  # XL: force it to be None so the buffer do not care about it

                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)
                
                # XL: assign to info tracker to use globally
                info_tracker = copy.deepcopy(infos)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logger.log("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, total basic steps {}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                infos[0, 0]["basic_steps"],
                                (total_num_steps / (end - start))))

                if self.all_args.env_name == "captureTargetSeparate":
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update({"average_step_rewards": np.mean(self.buffer[agent_id].rewards)})
                # self.log_train(train_infos, total_num_steps)
                self.log_train(train_infos, infos[0, 0]["basic_steps"])
                

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
            # self.buffer[agent_id].available_actions[0] = np.array(available_actions[:, agent_id]).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            None) # XL:  # available action uses None 
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id],
                                        available_actions=None)




 

  


  