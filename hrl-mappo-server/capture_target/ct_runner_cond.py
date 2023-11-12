# ct_runner_cond: runner for capture target
# this runner file uses one shared policy network


import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner
import copy

from misc import logger

def _t2n(x):
    return x.detach().cpu().numpy()

class CtRunnerCond(Runner):
    """Runner class to perform training, evaluation. and data collection for waterFill. See parent class for details."""
    def __init__(self, config):
        super(CtRunnerCond, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_tracker = None        # XL: add a info tracker to track unfinished, old actions
        # total_basic_num_steps = 0  # XL: a side info for calculating basic timesteps
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                if self.all_args.scheme in ["fully-cen"]:
                    if info_tracker is not None:
                        assert not isinstance(info_tracker, tuple) and info_tracker.shape[1] == 1
                        unfinished_info = []
                        for idx, ite in enumerate(info_tracker[0,0]["act_finished"]):
                            if not ite:
                                unfinished_info.append((idx, info_tracker[0,0]["act"][0,idx]))
                        if len(unfinished_info) == 1:
                            cond_idx = unfinished_info[0][0]
                            cond_value = unfinished_info[0][1]
                            uncond_idx = 1 - cond_idx
                            # 1.
                            # actions[0, 0, cond_idx]  = cond_value
                            # action_log_probs[0, 0, 0] -= self.trainer.policy.actor.act.action_out.fixnormal.log_probs_unsumed[0, cond_idx]

                            # 2.
                            orin_mean = self.trainer.policy.actor.act.action_out.action_mean
                            # orin_cov = torch.eye(2)  # by default use this
                            orin_cov = torch.tensor([[1.0, 0.45], [0.45, 1.0]]) # a bit dependency
                            new_mean = orin_mean[0, uncond_idx] + orin_cov[uncond_idx, cond_idx] * (cond_value - orin_mean[0, cond_idx]) / orin_cov[cond_idx, cond_idx]
                            new_var  = orin_cov[uncond_idx, uncond_idx] - orin_cov[uncond_idx, cond_idx] * orin_cov[cond_idx, uncond_idx] / orin_cov[cond_idx, cond_idx]

                            gauss_dist = torch.distributions.normal.Normal(torch.tensor([new_mean]), torch.tensor([new_var]))
                            m = gauss_dist.sample()
                            m_logprob = gauss_dist.log_prob(m)

                            actions[0, 0, uncond_idx] = m
                            actions[0, 0, cond_idx]  = cond_value
                            action_log_probs[0, 0, 0] = m_logprob[0] + 0
                            
                                

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                available_actions = None  # XL: force it to be None so the buffer do not care about it


                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

                # XL: assign to info trac ker to use globally
                info_tracker = copy.deepcopy(infos)
                # total_basic_num_steps += infos[0][0]['basic_steps']

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
                logger.log("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, total basic steps {},  FPS {}, walltime {}\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                infos[0, 0]["basic_steps"],
                                total_num_steps / (end - start),
                                (end - start)))

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                # self.log_train(train_infos, total_num_steps)
                self.log_train(train_infos, infos[0, 0]["basic_steps"])

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        # self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            None)  # available action uses None 
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break







