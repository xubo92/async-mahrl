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

class WfRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for waterFill. See parent class for details."""
    def __init__(self, config):
        super(WfRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_tracker = None   # XL: add a info tracker to track unfinished, old actions
        total_basic_num_steps = 0  # XL: a side info for calculating basic timesteps
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                if self.all_args.conditioned and self.all_args.ac_sp == "combi" and self.all_args.mode=="multi":
                    nn_new_act_names = self.envs.envs[0].action_list[actions[0][0][0]].split("_")
                    if info_tracker is not None:
                        assert not isinstance(info_tracker, tuple) and info_tracker.shape[1] == 1 and np.sum(info_tracker[0,0]["act_finished"]) != 0
                        probs = self.trainer.policy.actor.act.probs # obtain the probs before sampling action
                        for idx, ite in enumerate(info_tracker[0,0]["act_finished"]):
                            if not ite:
                                nn_new_act_names[idx] = info_tracker[0,0]["act_names"][idx] # nn predicted actions should be changed a bit
                                unfinished_act_name = info_tracker[0,0]["act_names"][idx]
                                probs_idx_subset = []
                                for jdx, act_item in enumerate(self.envs.envs[0].action_list):
                                    act_item_list = act_item.split("_")
                                    if unfinished_act_name in act_item_list:
                                        probs_idx_subset.append(jdx)

                                catg_dist = torch.distributions.categorical.Categorical(probs=probs[:, probs_idx_subset])
                                m = catg_dist.sample()
                                m_logprob = catg_dist.log_prob(m)
                                actions[0, 0, 0] = probs_idx_subset[m[0]]
                                action_log_probs[0, 0, 0] = m_logprob[0]
                                


                # nn_act_names = self.envs.envs[0].action_list[actions[0][0][0]].split("_")
                # print("action index: {}; action name: {}".format(actions, self.envs.envs[0].action_list[actions[0][0][0]]))
                
                # if self.all_args.conditioned and self.all_args.ac_sp == "combi" and self.all_args.mode=="multi":
                #     if info_tracker is not None:
                #         assert not isinstance(info_tracker, tuple) and info_tracker.shape[1] == 1
                #         for idx, ite in enumerate(info_tracker[0,0]["act_finished"]):
                #             if not ite:
                #                 nn_act_names[idx] = info_tracker[0,0]["act_names"][idx]
                #         actions[0, 0, 0] = self.envs.envs[0].action_list.index("_".join(nn_act_names))
                elif self.all_args.mode == "multi":
                    # XL: Some tricky things (important)
                    if info_tracker is not None:
                        if isinstance(info_tracker, tuple):  # if multi-envs collecting data
                            for m, it in enumerate(info_tracker):
                                for n, sit in enumerate(it):
                                    if not sit["act_finished"]:
                                        actions[m, n, 0] = info_tracker[m][n]["act"]
                                        action_log_probs[m, n, 0] = 0

                        else:
                            for i in range(info_tracker.shape[1]):  # if single env
                                if not info_tracker[0, i]["act_finished"]:
                                    actions[0, i, 0] = info_tracker[0, i]["act"]
                                    action_log_probs[0, i, 0] = 0
                else:
                    # print("running baseline")
                    assert self.all_args.mode == "single" and self.all_args.ac_sp == "basic"
                    pass

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

                # XL: assign to info tracker to use globally
                info_tracker = copy.deepcopy(infos)
                total_basic_num_steps += infos[0][0]['basic_steps']

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
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, real FPS {}, walltime {}\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                total_num_steps / (end - start),
                                total_basic_num_steps / (end - start),
                                (end - start)))

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                self.log_train(train_infos, total_num_steps)

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
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
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

