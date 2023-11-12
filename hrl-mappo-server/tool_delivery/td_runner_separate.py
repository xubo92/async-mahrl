# td_runner_separate: runner for tool_delivery
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

class TdRunnerSeparate(Runner):
    def __init__(self, config):
        super(TdRunnerSeparate, self).__init__(config)
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
                        if isinstance(info_tracker, tuple):  # if multi-envs collecting data
                            for m, it in enumerate(info_tracker):
                                for n, sit in enumerate(it):
                                    if not sit["act_finished"]:
                                        actions[m, n, 0] = info_tracker[m][n]["act"]
                                        action_log_probs[m, n, 0] = 0

                        else:
                            for i in range(info_tracker.shape[1]):  # if single env collecting data
                                if not info_tracker[0, i]["act_finished"]:
                                    actions[0, i, 0] = info_tracker[0, i]["act"]
                                    action_log_probs[0, i, 0] = 0

                elif self.all_args.scheme == "partial-dec":
                    if info_tracker is not None:
                        assert not isinstance(info_tracker, tuple)
                        # 1. find all unfinished agent ids and actions
                        unfinished_search = []
                        for idx, ite in enumerate(info_tracker[0]):
                            if not ite['act_finished']:
                                unfinished_act_idx = info_tracker[0,idx]["act"]
                                unfinished_agt_idx = idx
                                unfinished_search.append((unfinished_agt_idx, unfinished_act_idx))
                        # 2. for each agent, modify their action and log_prob according to its finished or not
                        for i in range(info_tracker.shape[1]):
                            if not info_tracker[0, i]["act_finished"]:
                                actions[0, i, 0] = info_tracker[0, i]["act"]
                                action_log_probs[0, i, 0] = 0
                            else:
                                if i in [0, 1]:  # for two turtlebots, use independent selection
                                    unfinished_search.append((i, actions[0, i, 0]))  
                                elif i in [2]:
                                    cur_probs = self.trainer[i].policy.actor.act.probs
                                    probs_idx_subset = []
                                    for k,v in self.envs.envs[0].act_table.items():
                                        flag = True
                                        for uf_i, uf in enumerate(unfinished_search):  # could be one or more agents unfinished
                                            if v[uf[0]] != uf[1]:
                                                flag = False
                                        if flag:
                                            probs_idx_subset.append(k)
                                    catg_dist = torch.distributions.categorical.Categorical(probs=cur_probs[:, probs_idx_subset])
                                    m = catg_dist.sample()
                                    m_logprob = catg_dist.log_prob(m)
                                    actions[0, i, 0] = self.envs.envs[0].act_table[probs_idx_subset[m]][i]
                                    action_log_probs[0, i, 0] = m_logprob[0] 
                    else:
                        unfinished_search = []
                        for i in range(self.num_agents):
                            if i in [0, 1]:  # for two turtlebots, use independent selection
                                unfinished_search.append((i, actions[0, i, 0]))  
                            elif i in [2]:
                                cur_probs = self.trainer[i].policy.actor.act.probs
                                probs_idx_subset = []
                                for k,v in self.envs.envs[0].act_table.items():
                                    flag = True
                                    for uf_i, uf in enumerate(unfinished_search):  # could be one or more agents unfinished
                                        if v[uf[0]] != uf[1]:
                                            flag = False
                                    if flag:
                                        probs_idx_subset.append(k)
                                catg_dist = torch.distributions.categorical.Categorical(probs=cur_probs[:, probs_idx_subset])
                                m = catg_dist.sample()
                                m_logprob = catg_dist.log_prob(m)
                                actions[0, i, 0] = self.envs.envs[0].act_table[probs_idx_subset[m]][i]
                                action_log_probs[0, i, 0] = m_logprob[0] 
                    
                    ## --- 1. use an order to determine partial dependency --- ##
                    # if info_tracker is not None:
                    #     assert not isinstance(info_tracker, tuple)
                    #     # 1. find all unfinished agent ids and actions
                    #     unfinished_search = []
                    #     for idx, ite in enumerate(info_tracker[0]):
                    #         if not ite['act_finished']:
                    #             unfinished_act_idx = info_tracker[0,idx]["act"]
                    #             unfinished_agt_idx = idx
                    #             unfinished_search.append((unfinished_agt_idx, unfinished_act_idx))
                    #     # 2. for each agent, modify their action and log_prob according to its finished or not
                    #     for i in range(info_tracker.shape[1]):
                    #         if not info_tracker[0, i]["act_finished"]:
                    #             actions[0, i, 0] = info_tracker[0, i]["act"]
                    #             action_log_probs[0, i, 0] = 0
                    #         else:
                    #             cur_probs = self.trainer[i].policy.actor.act.probs
                    #             probs_idx_subset = []
                    #             for k,v in self.envs.envs[0].act_table.items():
                    #                 flag = True
                    #                 for uf_i, uf in enumerate(unfinished_search):  # could be one or more agents unfinished
                    #                     if v[uf[0]] != uf[1]:
                    #                         flag = False
                    #                 if flag:
                    #                     probs_idx_subset.append(k)
                                
                    #             catg_dist = torch.distributions.categorical.Categorical(probs=cur_probs[:, probs_idx_subset])
                    #             m = catg_dist.sample()
                    #             m_logprob = catg_dist.log_prob(m)
                    #             actions[0, i, 0] = self.envs.envs[0].act_table[probs_idx_subset[m]][i]
                    #             action_log_probs[0, i, 0] = m_logprob[0] 

                    #             # update unfinished search with a newly choosed action for agent i
                    #             unfinished_search.append((i, actions[0, i, 0]))       
                    # else:
                    #     # first step when info_tracker is None, and select all new options (0~53) for all agents
                    #     unfinished_search = []
                    #     for i in range(self.num_agents):
                    #         if (len(unfinished_search) == 0):  # for the first agent, choose without unfinished dependency
                    #             tmp_idx = actions[0, i, 0]
                    #             actions[0, i, 0] = self.envs.envs[0].act_table[tmp_idx][i]
                    #         else:
                    #             cur_probs = self.trainer[i].policy.actor.act.probs
                    #             probs_idx_subset = []
                    #             for k,v in self.envs.envs[0].act_table.items():
                    #                 flag = True
                    #                 for uf_i, uf in enumerate(unfinished_search):  # could be one or more agents unfinished
                    #                     if v[uf[0]] != uf[1]:
                    #                         flag = False
                    #                 if flag:
                    #                     probs_idx_subset.append(k)
                                
                    #             catg_dist = torch.distributions.categorical.Categorical(probs=cur_probs[:, probs_idx_subset])
                    #             m = catg_dist.sample()
                    #             m_logprob = catg_dist.log_prob(m)
                    #             actions[0, i, 0] = self.envs.envs[0].act_table[probs_idx_subset[m]][i]
                    #             action_log_probs[0, i, 0] = m_logprob[0] 

                    #         # update unfinished search with a newly choosed action for agent i
                    #         unfinished_search.append((i, actions[0, i, 0]))

                    ### --- 2.
                    # if info_tracker is not None:
                    #     assert not isinstance(info_tracker, tuple)

                    #     # find all unfinished agent ids and actions
                    #     unfinished_search = []
                    #     for idx, ite in enumerate(info_tracker[0]):
                    #         if not ite['act_finished']:
                    #             unfinished_act_idx = info_tracker[0,idx]["act"]
                    #             unfinished_agt_idx = idx
                    #             unfinished_search.append((unfinished_agt_idx, unfinished_act_idx))

                    #     # for each agent, modify their action and log_prob according to its finished or not
                    #     for i in range(info_tracker.shape[1]):
                    #         if not info_tracker[0, i]["act_finished"]:
                    #             actions[0, i, 0] = info_tracker[0, i]["act"]
                    #             action_log_probs[0, i, 0] = 0
                    #         else:
                    #             cur_probs = self.trainer[i].policy.actor.act.probs
                    #             probs_idx_subset = []
                    #             for k,v in self.envs.envs[0].act_table.items():
                    #                 flag = True
                    #                 for uf_i, uf in enumerate(unfinished_search):  # could be one or more agents unfinished
                    #                     if v[uf[0]] != uf[1]:
                    #                         flag = False
                    #                 if flag:
                    #                     probs_idx_subset.append(k)
                                
                    #             catg_dist = torch.distributions.categorical.Categorical(probs=cur_probs[:, probs_idx_subset])
                    #             m = catg_dist.sample()
                    #             m_logprob = catg_dist.log_prob(m)
                    #             actions[0, i, 0] = self.envs.envs[0].act_table[probs_idx_subset[m]][i]
                    #             action_log_probs[0, i, 0] = m_logprob[0] 
                    
                    # else:
                    #     for i in range(self.num_agents):
                    #         tmp_idx = actions[0, i, 0]
                    #         actions[0, i, 0] = self.envs.envs[0].act_table[tmp_idx][i]


                    ### --- 3.
                    # if info_tracker is not None:
                    #     assert not isinstance(info_tracker, tuple)
                    #     unfinished_search = []
                    #     # find all unfinished agent ids and actions
                    #     for idx, ite in enumerate(info_tracker[0]):
                    #         if not ite['act_finished']:
                    #             unfinished_act_idx = info_tracker[0,idx]["act"]
                    #             unfinished_agt_idx = idx
                    #             unfinished_search.append((unfinished_agt_idx, unfinished_act_idx))

                    #     # for each agent, modify their action and log_prob according to its finished or not
                    #     for i in range(info_tracker.shape[1]):
                    #         if not info_tracker[0, i]["act_finished"]:
                    #             actions[0, i, 0] = info_tracker[0, i]["act"]
                    #             action_log_probs[0, i, 0] = 0
                    #         else:
                    #             cur_probs = self.trainer[i].policy.actor.act.probs
                    #             probs_idx_subset = []
                    #             for k,v in self.envs.envs[0].act_table.items():
                    #                 flag = True
                    #                 for uf_i, uf in enumerate(unfinished_search):  # could be one or more agents unfinished
                    #                     if v[uf[0]] != uf[1]:
                    #                         flag = False
                    #                 if flag:
                    #                     probs_idx_subset.append(k)
                            
                    #             # shrink probs_subset to marginal 
                    #             marginal_probs = [0] * len(self.envs.envs[0].marginal_act[i])
                    #             for pis in probs_idx_subset:
                    #                 tmp_idx = self.envs.envs[0].act_table[pis][i]
                    #                 marginal_probs[tmp_idx] += cur_probs[0][pis]
                    #             catg_dist = torch.distributions.categorical.Categorical(probs=torch.Tensor([marginal_probs]))
                    #             m = catg_dist.sample()
                    #             m_logprob = catg_dist.log_prob(m)
                    #             actions[0, i, 0] = m
                    #             action_log_probs[0, i, 0] = m_logprob[0] 
                    # else:
                    #     # first step when info_tracker is None, and select all new options (0~53) for all agents
                    #     for i in range(self.num_agents):
                    #         new_probs = [0] * len(self.envs.envs[0].marginal_act[i])
                    #         probs = self.trainer[i].policy.actor.act.probs
                    #         for j in range(probs.shape[1]):
                    #             tmp_idx = self.envs.envs[0].act_table[j][i]
                    #             new_probs[tmp_idx] += probs[0][j]

                    #         catg_dist = torch.distributions.categorical.Categorical(probs=torch.Tensor([new_probs]))
                    #         m = catg_dist.sample()
                    #         m_logprob = catg_dist.log_prob(m)
                    #         actions[0, i, 0] = m
                    #         action_log_probs[0, i, 0] = m_logprob[0]


                elif self.all_args.scheme == "partial-cen":
                    if info_tracker is not None:
                        assert not isinstance(info_tracker, tuple)
                        for i in range(info_tracker.shape[1]):  # if single env collecting data
                            if not info_tracker[0, i]["act_finished"]:
                                actions[0, i, 0] = info_tracker[0, i]["act"]
                                action_log_probs[0, i, 0] = 0


                elif self.all_args.scheme == "sync-wait":
                    if info_tracker is not None:
                        assert not isinstance(info_tracker, tuple)
                        for i in range(info_tracker.shape[1]):  # if single env collecting data
                            if not np.any(info_tracker[0, i]["termi_info"]):
                                assert info_tracker[0, i]["act_finished"]
                          
                
                elif self.all_args.scheme == "sync-cut":
                    pass
                else:
                    pass

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

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

                if self.all_args.env_name == "toolDeliverySeparate":
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
            self.buffer[agent_id].available_actions[0] = np.array(available_actions[:, agent_id]).copy()

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
                                                            self.buffer[agent_id].available_actions[step])
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
                                        available_actions=available_actions[:, agent_id])




 

  


  