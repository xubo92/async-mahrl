import numpy as np
from gym import spaces
from gym.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from pettingzoo.mpe.scenarios.simple import Scenario
from pettingzoo.mpe._mpe_utils import rendering

g_frames = []
def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env


class SimpleEnv(AECEnv):
    def __init__(self, scenario, world, max_cycles, continuous_actions=False, local_ratio=None):
        super().__init__()

        self.seed()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'is_parallelizable': True,
            'video.frames_per_second': 10
        }

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.world.agents)}

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(space_dim,))
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,), dtype=np.float32)

        self.state_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(state_dim,), dtype=np.float32)

        self.steps = 0

        self.current_actions = [None] * self.num_agents

        self.viewer = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32)

    def state(self):
        states = tuple(self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32) for agent in self.possible_agents)
        return np.concatenate(states, axis=None)

    def reset(self):
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0. for name in self.agents}
        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._reset_render()

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        # XL: rendering
        self.render()
        if self.viewer != None:
            g_frames.append(self.viewer.get_array())
            print("length of frames:", len(g_frames))
            if len(g_frames) >= 500:
                from moviepy.editor import ImageSequenceClip
                clip = ImageSequenceClip(g_frames, fps=5)
                clip.write_videofile("D:/xubo92/hrl-mappo-server/visuals/ct_frames.mp4")
                sys.pause()

    def render(self, mode='human'):
        # from . import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.state.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.state.c]) + "]"
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()




# class raw_env(SimpleEnv):
#     def __init__(self, max_cycles=25, continuous_actions=True):
#         scenario = Scenario()
#         world = scenario.make_world()
#         super().__init__(scenario, world, max_cycles, continuous_actions)
#         self.metadata['name'] = "simple_v2"




class captureTargetCond(object):
    def __init__(self, max_cycles=35, continuous_actions=True, **kwargs):
        scenario = Scenario()
        world = scenario.make_world()
        
        self.observation_space = [[8 + 8, []] * 1]  # each agent has obs: self_vel (2) + self_pos (2) + landmark_rel_pos (2) + the_other_agent_pos (2)
        self.share_observation_space = [[8 + 8, []] * 1] 
        self.action_space = [spaces.Box(low = -1.0, high = 1.0, shape=(2,), dtype=np.float32)]
        self.agt_avail_acts = None

        self.meta_env = SimpleEnv(scenario, world, max_cycles, continuous_actions)
        self.basic_steps = 0
        

    # action: [[agent_1_vel, agent_2_vel]]; agent_1_vel:[-1,1] -> [0, 0.8]; agent_2_vel: [-1,1] -> [0.5, 1.2]
    # [0.5, 1.3], [1,1.7]
    def step(self, action):
        a = np.clip(action[0], self.action_space[0].low, self.action_space[0].high) 

        # a0 = (a[0] - (-1)) * (0.8 - 0) / (1 - (-1))
        # a1 = (a[1] - (-1)) * (1.2 - 0.5) / (1 - (-1))

        a0 = (a[0] - (-1)) * (1.3 - 0.5) / (1 - (-1))
        a1 = (a[1] - (-1)) * (1.7 - 1.0) / (1 - (-1))

        new_act = [a0, a1]

        local_obs, share_obs= [], []
        dones = [False]
        infos = []
        cur_actions_done = np.array([0,0])
        while (not self.meta_env.dones['agent_0'] and not np.any(cur_actions_done)):
            cur_obs, cur_reward = [], []
            for j in range(2):
                landmark_pos  = self.meta_env.world.landmarks[0].state.p_pos
                cur_agent_pos = self.meta_env.world.agents[j].state.p_pos
                theta = np.arctan(np.abs(landmark_pos[1] - cur_agent_pos[1]) / np.abs(landmark_pos[0] - cur_agent_pos[0]))
                
                self.meta_env.world.agents[j].state.p_vel[0] = new_act[j] * np.cos(theta) * np.sign(landmark_pos[0] - cur_agent_pos[0])
                self.meta_env.world.agents[j].state.p_vel[1] = new_act[j] * np.sin(theta) * np.sign(landmark_pos[1] - cur_agent_pos[1])
                self.meta_env.step(np.array([0., 0., 0., 0., 0.]))

            for j in range(2):
                cur_obs.append(np.concatenate((self.meta_env.world.agents[j].state.p_vel, \
                                                self.meta_env.world.agents[j].state.p_pos, \
                                                self.meta_env.world.landmarks[0].state.p_pos - self.meta_env.world.agents[j].state.p_pos, \
                                                self.meta_env.world.agents[1-j].state.p_pos)))
                cur_reward.append(self.meta_env.rewards['agent_{}'.format(j)])

                if self.is_collision(self.meta_env.world.agents[j], self.meta_env.world.landmarks[0]):
                    cur_actions_done[j] = 1
            self.basic_steps += 1

            

        local_obs.append(np.concatenate(cur_obs))
        share_obs.append(np.concatenate(cur_obs))
        reward = [[np.mean(cur_reward)]] * 1
        
        if self.meta_env.dones['agent_0'] or np.any(cur_actions_done):
            dones = [True]

        infos.append({"bad_transition":False})
        infos[0]["act"] = action
        infos[0]["act_finished"] = []
        infos[0]["basic_steps"] = self.basic_steps
        for i in range(2):
            if cur_actions_done[i]:
                infos[0]["act_finished"].append(True)
            else:
                infos[0]["act_finished"].append(False)

        return local_obs, share_obs, reward, np.array(dones), infos, self.agt_avail_acts 

    def reset(self):
        local_obs, shared_obs = [], []
        self.meta_env.reset()
        cur_obs = []
        for j in range(2):
            cur_obs.append(np.concatenate((self.meta_env.world.agents[j].state.p_vel, \
                            self.meta_env.world.agents[j].state.p_pos, \
                            self.meta_env.world.landmarks[0].state.p_pos - self.meta_env.world.agents[j].state.p_pos, \
                            self.meta_env.world.agents[1-j].state.p_pos)))
        local_obs.append(np.concatenate(cur_obs))
        shared_obs.append(np.concatenate(cur_obs))

        return local_obs, shared_obs, self.agt_avail_acts

    def _get_avail_actions(self, agtId):
        return None

    def seed(self, seed=None):
        return seed

    def close(self):
        return 0

    def is_collision(self, agent, landmark):
        delta_pos = agent.state.p_pos - landmark.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + landmark.size
        return True if dist < dist_min else False


if __name__ == "__main__":
    # env = raw_env()
    # obs = env.reset()
    # acts = [None, None]
    # for i in range(100):
    #     print("step {}".format(i))
    #     for j in range(2):
    #         acts[j] = np.array([0., 0., 0., 0., 0.])
    #         env.world.agents[j].state.p_vel[0] = 0.45
    #         env.world.agents[j].state.p_vel[1] = 0.45
    #         obs = env.step(acts[j])
    #         if env.dones['agent_{}'.format(j)]:
    #             obs = env.reset()
    #         # if env.dones['agent_{}'.format(j)]:
    #         #     acts[j] = None
    #         # else:
    
    env = captureTargetCond()
    local_obs, shared_obs, agt_avail_acts = env.reset()
    for i in range(100):
        act = np.random.uniform(low=-1, high=1, size=2)
        local_obs, shared_obs, reward, done, infos, agt_avail_acts = env.step([act])
        if done[0,0]:
            local_obs, shared_obs, agt_avail_acts = env.reset()
            
    