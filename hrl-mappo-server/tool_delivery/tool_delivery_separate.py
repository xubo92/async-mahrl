#!/usr/bin/python
import gym
import numpy as np
import IPython
from gym import spaces
from gym.utils import seeding
from IPython.core.debugger import set_trace

import sys
sys.path.append("D:/xubo92/hrl-mappo-server/")
from misc import logger

# XL: store frames for creating mp4 video
g_frames = []

class AgentTurtlebot_v4(object):

    """Properties for a Turtlebot"""

    def __init__(self, 
                 idx, 
                 init_x, 
                 init_y,
                 beliefwaypoints,
                 MAs,
                 n_objs,
                 speed = 0.6):

        # unique agent's id
        self.idx = idx
        # agent's name
        self.name = 'Turtlebot'+ str(self.idx)
        # agent's 2D position x 
        self.xcoord = init_x
        # agent's 2D position y
        self.ycoord = init_y
        # applicable waypoints to move to
        self.BWPs = beliefwaypoints
        # record which belief waypoint the agent currently is
        self.cur_BWP = None
        # obtain applicable macro_actions
        self.macro_actions = MAs
        # agent's current macro_action 
        self.cur_action = None
        # how much time left to finish current macro_action
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        # turtlebot base movement speed
        self.speed = speed

        # communication info
        self.n_objs = n_objs
        # keep tracking the objects in the basket
        self.objs_in_basket = np.zeros(n_objs)
        # keep tracking the message of request objects received by Fetch robot
        self.request_objs = np.zeros(n_objs)
        
        # XL: add it for sync-wait
        self.sync_wait_flag = False
    def step(self, action, humans):

        """Depends on the input macro-action to run low-level controller to achieve 
           primitive action execution.
        """

        assert action < len(self.macro_actions), "The action received is out of the range"

        reward = 0.0

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False

        # action 0 - 2
        if action <= 2:
            bwpterm_idx = self.cur_action.ma_bwpterm
            if self.cur_action.expected_t_cost != 1:
                dist = round(self._get_dist(self.BWPs[bwpterm_idx].xcoord, self.BWPs[bwpterm_idx].ycoord),2)
                if dist < self.speed:
                    self.xcoord = self.BWPs[bwpterm_idx].xcoord
                    self.ycoord = self.BWPs[bwpterm_idx].ycoord
                    self.cur_BWP = self.BWPs[bwpterm_idx]
                    if self.cur_action_time_left >= 0.0:
                        self.cur_action_time_left = 0.0
                    if action < 2:
                        self.cur_action_done = True
                    else:
                        # XL: for sync-wait 
                        if self.sync_wait_flag:
                            self.cur_action_time_left = 0.0
                            self.cur_action_done = True
                        else:
                            # indicates turtlebot has been ready to get obj from fetch
                            self.cur_action_time_left -= 1.0
                            if self.cur_action_time_left < -10.0:   #----get_tool action automatically terminate after waiting for 10s
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                else:
                    delta_x = self.speed / dist * (self.BWPs[bwpterm_idx].xcoord - self.xcoord)
                    delta_y = self.speed / dist * (self.BWPs[bwpterm_idx].ycoord - self.ycoord)
                    self.xcoord += delta_x
                    self.ycoord += delta_y
                    self.cur_action_time_left = dist - self.speed
            else:
                self.xcoord = self.BWPs[bwpterm_idx].xcoord
                self.ycoord = self.BWPs[bwpterm_idx].ycoord
                self.cur_BWP = self.BWPs[bwpterm_idx]

        elif action >= 3:
            if self.xcoord < 3.5:
                self.request_objs[action-3] = 1.0
            self.cur_action_done = True

        # change the human's properties when turtlebot deliever correct objects
        if self.cur_BWP is not None and \
           (action == 0 and self.cur_BWP.idx == action):
            human = humans[self.cur_BWP.idx]
            if not human.next_requested_obj_obtained and \
                    self.objs_in_basket[human.next_request_obj_idx] > 0.0:
                        self.objs_in_basket[human.next_request_obj_idx] -= 1.0
                        reward += 100
                        human.next_requested_obj_obtained = True

        return reward

    def _get_dist(self, g_xcoord, g_ycoord):
        return np.sqrt((g_xcoord - self.xcoord)**2 + (g_ycoord - self.ycoord)**2)

class AgentFetch_v4(object):

    """Properties for a Fetch robot"""
    """Double Check for passing obj action, beginning and end"""

    def __init__(self, 
                 idx, 
                 init_x, 
                 init_y,
                 MAs,
                 n_objs,
                 n_each_obj):

        # unique agent's id
        self.idx = idx
        # agent's name
        self.name = 'Fetch'
        # agent's 2D position x
        self.xcoord = init_x
        # agent's 2D position y
        self.ycoord = init_y
        # obtain applicable macro_actions
        self.macro_actions = MAs
        # agent's current macro_action
        self.cur_action = None
        # how much time left to finish current macro_action
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        # the number of different objects in this env
        self.n_objs = n_objs
        # the amout of each obj in the env
        self.n_each_obj = n_each_obj
        # counte the object that has been found by fetch robot
        self.count_found_obj = np.zeros(n_objs)
        
        ################# communication info ######################
        # indicates if fetch is serving or not
        self.serving = False
        self.serving_failed = False
        # [0,0] means there is no any object ready for Turtlebot1 and Turtlebot2
        self.ready_objs = np.zeros(2)
        self.found_objs = []

    def step(self, action, agents):

        """Depends on the input macro-action to run low-level controller to achieve 
           primitive action execution.
        """

        reward = 0.0

        self.cur_action_time_left -= 1.0
        
        if self.cur_action_time_left  > 0.0:
            return reward
        else:
            if self.cur_action_done:
                self.cur_action = self.macro_actions[action]
                self.cur_action_time_left = self.cur_action.t_cost - 1.0
                # action 0 wait request
                if self.cur_action.idx == 0:
                    self.cur_action_done = True
                else:
                    self.cur_action_done = False

                # when fetch execute pass_obj action, the corresponding turtlebot has to stay beside table
                if self.cur_action.idx == 1:
                    self.serving = True
                    if agents[0].cur_BWP is None or \
                       agents[0].cur_BWP.name != "ToolRoomTable" or \
                       agents[0].cur_action_time_left > -1.0:
                        self.serving_failed = True
                elif self.cur_action.idx == 2:
                    self.serving = True
                    if agents[1].cur_BWP is None or \
                       agents[1].cur_BWP.name != "ToolRoomTable" or \
                       agents[1].cur_action_time_left > -1.0:
                        self.serving_failed = True

                return reward

            # action 1 Pass_obj_T0
            elif self.cur_action.idx == 1:
                self.serving = False
                if not self.serving_failed and agents[0].cur_action_time_left < 0.0:
                    if len(self.found_objs) > 0:
                        obj_idx = self.found_objs.pop(0)
                        agents[0].objs_in_basket[obj_idx] += 1.0
                    agents[0].cur_action_done = True
                    agents[0].cur_action_time_left = 0.0
                        
                    # check if there is still any other object ready for turtlebot 1
                    self.ready_objs = np.zeros(2)
                    if len(self.found_objs) == 1:
                        self.ready_objs[0]=1.0

                else:
                    reward += -10.0

                self.serving_failed = False

            # action 2 Pass_obj_T1
            elif self.cur_action.idx == 2:
                self.serving = False
                if not self.serving_failed and agents[1].cur_action_time_left < 0.0:
                    if len(self.found_objs) > 0:
                        obj_idx = self.found_objs.pop(0)
                        agents[1].objs_in_basket[obj_idx] += 1.0
                    agents[1].cur_action_done = True
                    agents[1].cur_action_time_left = 0.0

                    # check if there is still any other object ready for turtlebot 1
                    self.ready_objs = np.zeros(2)
                    if len(self.found_objs) == 1:
                        self.ready_objs[0] = 1.0

                else:
                    reward += -10.0
                
                self.serving_failed = False

            # action Look_for_T0_obj
            elif self.cur_action.idx < 3+self.n_objs: 
                found_obj_idx = self.cur_action.idx - 3
                if len(self.found_objs) < 2 and self.count_found_obj[found_obj_idx] < self.n_each_obj:   #---------------tweak 3
                    self.count_found_obj[found_obj_idx] += 1.0
                    self.found_objs.append(found_obj_idx)
                    if len(self.found_objs) == 2:
                        self.ready_objs[1] = 1.0
                        self.ready_objs[0] = 0.0
                    else:
                        self.ready_objs[0] = 1.0
                        self.ready_objs[1] = 0.0

            # indicate the current action finished
            self.cur_action_done = True

        return reward

class AgentHuman(object):

    """Properties for a Human in the env"""

    def __init__(self,
                 idx,
                 task_total_steps,
                 expected_timecost_per_task_step,
                 request_objs_per_task_step,
                 std=None,
                 seed=None):

        # unique human's id
        self.idx = idx
        # the total number of steps for finishing the task
        self.task_total_steps = task_total_steps
        # a vector to indicate the expected time cost for each human to finish each task step
        self.expected_timecost_per_task_step = expected_timecost_per_task_step
        # std is used to sample the actual time cost for each human to finish each task step
        self.time_cost_std_per_task_step = std
        # a vector to inidcate the tools needed for each task step
        self.request_objs_per_task_step = request_objs_per_task_step

        self.cur_step = 0 
        if std is None:
            self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
        else:
            # sample the time cost for the current task step, which will be counted down step by step
            self.cur_step_time_left = self.np.random.normal(self.expected_timecost_per_task_step[self.cur_step], 
                                                            self.time_cost_std_per_task_step)  

        # indicates the tool needed for next task step
        self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step]  
        # indicates if the tool needed for next step has been delivered
        self.next_requested_obj_obtained = False
        # indicates if the human has finished the whole task
        self.whole_task_finished = False

    def step(self):

        # check if the human already finished whole task
        if self.cur_step + 1 == self.task_total_steps:
            assert self.whole_task_finished == False
            self.whole_task_finished = True
        else:
            self.cur_step += 1
            if self.time_cost_std_per_task_step is None:
                self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
            else:
                self.cur_step_time_left = self.np.random.normal(self.expected_timecost_per_task_step[self.cur_step], 
                                                                self.time_cost_std_per_task_step)
            # update the request obj for next step
            if self.cur_step + 1 < self.task_total_steps:
                self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step] 
                self.next_requested_obj_obtained = False

    def reset(self):
        self.cur_step = 0
        if self.time_cost_std_per_task_step is None:
            self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
        else:
            # sample the time cost for the current task step, which will be counted down step by step
            self.cur_step_time_left = self.np.random.normal(self.expected_timecost_per_task_step[self.cur_step],
                                                            self.time_cost_std_per_task_step)

        # indicates the tool needed for next task step
        self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step]  
        # indicates if the tool needed for next step has been delivered
        self.next_requested_obj_obtained = False
        # indicates if the human has finished the whole task
        self.whole_task_finished = False

class MacroAction(object):

    """Properties for a macro_action"""

    def __init__(self, 
                 name,
                 idx,
                 expected_t_cost=None,
                 std=None,
                 ma_bwpterm=None):

        # the name of this macro-action
        self.name = name
        # the index of this macro-action
        self.idx = idx
        # None is for moving action. When it is done depends on the specify speed.
        self.expected_t_cost = expected_t_cost
        self.std = std
        if std is None:
            # the time cost of finishing this macro-action
            self.real_t_cost = expected_t_cost
        else:
            self.real_t_cost = np.random.normal(expected_t_cost, std)
        # used for moving action to indicate at which belief waypoint this macro-action will be terminated,
        # None means the terminate belief waypoint is same as where the action is initialized.
        self.ma_bwpterm = ma_bwpterm

    @property
    def t_cost(self):
        if self.std is None:
            # the time cost of finishing this macro-action
            return self.expected_t_cost
        else:
            # resample a time cost for the macro-action
            return round(np.random.normal(self.expected_t_cost, self.std),1)   
 
class BeliefWayPoint(object):

    """Properties for a waypoint in the 2D sapce"""

    def __init__(self,
                 name,
                 idx,
                 xcoord,
                 ycoord):
        
        self.name = name
        self.idx = idx
        self.xcoord = xcoord
        self.ycoord = ycoord


class ObjSearchDelivery(gym.Env):

    """Base class of object search and delivery domain"""

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, 
                 n_objs=3, 
                 n_each_obj=1,
                 terminate_step=150, 
                 tbot_speed=0.6, *args, **kwargs):
        """
        Parameters
        ----------
        n_objs : int
            The number of object's types in the domain.
        n_each_obj : int
            The number of objects per object's type
        terminate_step : int
            The maximal steps per episode.
        tbot_speed : float
            Turtlebot's moving speed m/s
        """

        self.n_agent = 3

        #-----------------basic settings for this domain
        # define the number of different objects needs for each human to finish the whole task
        self.n_objs = n_objs
        # total amount of each obj in the env
        self.n_each_obj = n_each_obj
        # define the number of steps for each human finishing the task 
        self.n_steps_human_task = self.n_objs + 1

        #-----------------def belief waypoints
        BWP0 = BeliefWayPoint('WorkArea0', 0, 6.0, 3.0)
        BWP1 = BeliefWayPoint('ToolRoomWait', 1, 2.5, 3.5)
        BWP2_T0 = BeliefWayPoint('ToolRoomTable', 2, 0.8, 3.0)
        BWP2_T1 = BeliefWayPoint('ToolRoomTable', 2, 1.2, 3.0)

        self.BWPs = [BWP0, BWP1, BWP2_T0, BWP2_T1]
        self.BWPs_T0 = [BWP0, BWP1, BWP2_T0]
        self.BWPs_T1 = [BWP0, BWP1, BWP2_T1]
        
        self.viewer = None
        self.terminate_step = terminate_step

        self.human_waiting_steps = []
        self.turtlebot_speed=tbot_speed

    @property
    def obs_size(self):
        return [self.observation_space_T.n] *2 + [self.observation_space_F.n]

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    @property
    def action_spaces(self):
        return [self.action_space_T] * 2 + [self.action_space_F]

    def create_turtlebot_actions(self):
        raise NotImplementedError

    def create_fetch_actions(self):
        raise NotImplementedError

    def createAgents(self):
        raise NotImplementedError

    def createHumans(self):

        #-----------------initialize Three Humans
        Human0 = AgentHuman(0, self.n_steps_human_task, [18,18,18,18], list(range(self.n_objs)))

        # recording the number of human who has finished his own task
        self.humans = [Human0]
        self.n_human_finished = []
        
    def step(self, actions, debug=False):
        raise NotImplementedError

    def reset(self, debug=False):
        
        # reset the agents in this env
        self.createAgents()

        # reset the humans in this env
        for human in self.humans:
            human.reset()
        self.n_human_finished = []
        
        self.t = 0   # indicates the beginning of one episode, check _getobs()
        self.count_step = 0

        if debug:
            self.render()

        return self._getobs()

    def _getobs(self, debug=False):
        raise NotImplementedError

    def render(self, mode='human'):

        screen_width = 700
        screen_height = 500

        if self.viewer is None:
            import misc.rendering as rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            self.pass_objs = 0

            line = rendering.Line((0.0, 0.0), (0.0, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_height), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #--------------------------------draw rooms' boundaries

            for i in range(0,100,2):
                line_tool_room = rendering.Line((350, i*5), (350, (i+1)*5))
                line_tool_room.set_color(0,0,0)
                line_tool_room.linewidth.stroke = 2
                self.viewer.add_geom(line_tool_room)

            for i in range(0,40,2):
                line_WA = rendering.Line((700, i*5), (700, (i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
            
            for i in range(0,40,2):
                line_WA = rendering.Line((700+i*5, 200), (700+(i+1)*5, 200))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
                
            for i in range(0,80,2):
                line_WA = rendering.Line((500+i*5, 300), (500+(i+1)*5, 300))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,80,2):
                line_WA = rendering.Line((700, 300+i*5), (700, 300+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
            
            for i in range(0,80,2):
                line_WA = rendering.Line((500, 300+i*5), (500, 300+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
            
            #---------------------------draw BW0
            for i in range(len(self.BWPs)):
                BWP = rendering.make_circle(radius=6)
                BWP.set_color(178.0/255.0, 34.0/255.0, 34.0/255.0)
                BWPtrans = rendering.Transform(translation=(self.BWPs[i].xcoord*100, self.BWPs[i].ycoord*100))
                BWP.add_attr(BWPtrans)
                self.viewer.add_geom(BWP)

            #-------------------------------draw table
            tablewidth = 60.0
            tableheight = 180.0
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.43,0.28,0.02)
            tabletrans = rendering.Transform(translation=(175, 180))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            tablewidth = 54.0
            tableheight = 174.0
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.67,0.43,0.02)
            tabletrans = rendering.Transform(translation=(175, 180))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            #-----------------------------draw Fetch
            fetch = rendering.make_circle(radius=28)
            fetch.set_color(*(0.0,0.0,0.0))
            self.fetchtrans = rendering.Transform(translation=(self.agents[2].xcoord*100, self.agents[2].ycoord*100))
            fetch.add_attr(self.fetchtrans)
            self.viewer.add_geom(fetch)

            #-----------------------------draw Fetch
            fetch_c = rendering.make_circle(radius=25)
            fetch_c.set_color(*(0.5, 0.5,0.5))
            self.fetchtrans_c = rendering.Transform(translation=(self.agents[2].xcoord*100, self.agents[2].ycoord*100))
            fetch_c.add_attr(self.fetchtrans_c)
            self.viewer.add_geom(fetch_c)

            #-----------------------------draw Fetch arms
            self.arm2 = rendering.FilledPolygon([(-5.0,-20.0,), (-5.0, 20.0), (5.0, 20.0), (5.0, -20.0)])
            self.arm2.set_color(0.0, 0.0, 0.0)
            self.arm2trans = rendering.Transform(translation=(self.agents[2].xcoord*10000+49, self.agents[2].ycoord*100), rotation = -90/180*np.pi)
            self.arm2.add_attr(self.arm2trans)
            self.viewer.add_geom(self.arm2)

            self.arm2_c = rendering.FilledPolygon([(-3.0,-18.0,), (-3.0, 18.0), (3.0, 18.0), (3.0, -18.0)])
            self.arm2_c.set_color(0.5, 0.5, 0.5)
            self.arm2trans_c = rendering.Transform(translation=(self.agents[2].xcoord*10000+48, self.agents[2].ycoord*100), rotation = -90/180*np.pi)
            self.arm2_c.add_attr(self.arm2trans_c)
            self.viewer.add_geom(self.arm2_c)

            self.arm1 = rendering.FilledPolygon([(-5.0,-38.0,), (-5.0, 38.0), (5.0, 38.0), (5.0, -38.0)])
            self.arm1.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 243), rotation = -15/180*np.pi)
            self.arm1.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1)

            self.arm1_c = rendering.FilledPolygon([(-3.0,-36.0,), (-3.0, 36.0), (3.0, 36.0), (3.0, -36.0)])
            self.arm1_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 243), rotation = -15/180*np.pi)
            self.arm1_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1_c)
            
            self.arm0 = rendering.FilledPolygon([(-5.0,-35.0,), (-5.0, 35.0), (5.0, 35.0), (5.0, -35.0)])
            self.arm0.set_color(1.0, 1.0, 1.0)
            arm0trans = rendering.Transform(translation=(82, 243), rotation = 5/180*np.pi)
            self.arm0.add_attr(arm0trans)
            self.viewer.add_geom(self.arm0)

            self.arm0_c = rendering.FilledPolygon([(-3.0,-33.0,), (-3.0, 33.0), (3.0, 33.0), (3.0, -33.0)])
            self.arm0_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(82, 243), rotation = 5/180*np.pi)
            self.arm0_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm0_c)

            #----------------------------draw Turtlebot_1
            turtlebot_1 = rendering.make_circle(radius=17.0)
            turtlebot_1.set_color(*(0.15,0.65,0.15))
            self.turtlebot_1trans = rendering.Transform(translation=(self.agents[0].xcoord*100, self.agents[0].ycoord*100))
            turtlebot_1.add_attr(self.turtlebot_1trans)
            self.viewer.add_geom(turtlebot_1)

            turtlebot_1_c = rendering.make_circle(radius=14.0)
            turtlebot_1_c.set_color(*(0.0,0.8,0.4))
            self.turtlebot_1trans_c = rendering.Transform(translation=(self.agents[0].xcoord*100, self.agents[0].ycoord*100))
            turtlebot_1_c.add_attr(self.turtlebot_1trans_c)
            self.viewer.add_geom(turtlebot_1_c)
            
            #----------------------------draw Turtlebot_2
            turtlebot_2 = rendering.make_circle(radius=17.0)
            turtlebot_2.set_color(*(0.15,0.15,0.65))
            self.turtlebot_2trans = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100))
            turtlebot_2.add_attr(self.turtlebot_2trans)
            self.viewer.add_geom(turtlebot_2)

            turtlebot_2_c = rendering.make_circle(radius=14.0)
            turtlebot_2_c.set_color(*(0.0,0.4,0.8))
            self.turtlebot_2trans_c = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100))
            turtlebot_2_c.add_attr(self.turtlebot_2trans_c)
            self.viewer.add_geom(turtlebot_2_c)

            #----------------------------draw human_2's status
            self.human0_progress_bar = []
            total_steps = self.humans[0].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10,-10), (-10,10), (10,10), (10,-10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26,480))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human0_progress_bar.append(progress_bar)

        # draw each robot's status
        self.turtlebot_1trans.set_translation(self.agents[0].xcoord*100, self.agents[0].ycoord*100)
        self.turtlebot_1trans_c.set_translation(self.agents[0].xcoord*100, self.agents[0].ycoord*100)
        self.turtlebot_2trans.set_translation(self.agents[1].xcoord*100, self.agents[1].ycoord*100)
        self.turtlebot_2trans_c.set_translation(self.agents[1].xcoord*100, self.agents[1].ycoord*100)
        self.fetchtrans.set_translation(self.agents[2].xcoord*100, self.agents[2].ycoord*100)

        for idx, bar in enumerate(self.human0_progress_bar):
            bar.set_color(0.8,0.8,0.8)

        # draw each human's status
        if self.humans[0].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx < self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[0].cur_step:
                    bar.set_color(0.0, 1.0, 0.0)
                    break
        else:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx <= self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # reset fetch arm
        self.arm0.set_color(1.0, 1.0, 1.0)
        self.arm0_c.set_color(1.0, 1.0, 1.0)
        self.arm1.set_color(1.0, 1.0, 1.0)
        self.arm1_c.set_color(1.0, 1.0, 1.0)

        self.arm2trans_c.set_translation(self.agents[2].xcoord*10000+48, self.agents[2].ycoord*100)
        self.arm2trans.set_translation(self.agents[2].xcoord*10000+49, self.agents[2].ycoord*100)


        if self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 1 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    # self.pass_objs += 1  # XL: comment to show pass all time
                    self.arm0.set_color(0.0, 0.0, 0.0)
                    self.arm0_c.set_color(0.5,0.5,0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 2 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    # self.pass_objs += 1  # XL: comment to show pass all time
                    self.arm1.set_color(0.0, 0.0, 0.0)
                    self.arm1_c.set_color(0.5, 0.5, 0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx > 2 and \
                np.sum(self.agents[2].count_found_obj) < self.n_objs:
                    self.arm2trans_c.set_translation(self.agents[2].xcoord*100+48, self.agents[2].ycoord*100)
                    self.arm2trans.set_translation(self.agents[2].xcoord*100+49, self.agents[2].ycoord*100)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class ObjSearchDelivery_v4(ObjSearchDelivery):

    """1) Not distinguish Look_for_obj to different robot.
       2) Turtlebot's macro-action "get tool" has two terminate conditions:
            a) wait besides the table until fetch pass any obj to it;
            b) terminates in 10s.
       3) Turtlebot observes human working status when locates at workshop room"""

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, *args, **kwargs):

        super(ObjSearchDelivery_v4, self).__init__(*args, **kwargs)

        self.create_turtlebot_actions()
        self.create_fetch_actions()
        self.createAgents()
        self.createHumans()

    def create_turtlebot_actions(self):

        self.action_space_T = spaces.Discrete(3)
        self.observation_space_T = spaces.MultiBinary(4+self.n_steps_human_task+self.n_objs+2)     # Descrete location areas: 3 
                                                                                                   # human's working step
                                                                                                   # which object in the basket: n_objs
                                                                                                   # number of object is ready: 2
        #-----------------def single step macro-actions for Turtlebot 
        T_MA0 = MacroAction('Go_WA0', 0, expected_t_cost = None, ma_bwpterm = 0)
        T_MA1 = MacroAction('Go_Tool_Room', 1, expected_t_cost = None, ma_bwpterm = 1)
        T_MA2 = MacroAction('Get_Tool', 2, expected_t_cost = None, ma_bwpterm = 2)

        self.T_MAs = [T_MA0, T_MA1, T_MA2]

    def create_fetch_actions(self):
        self.action_space_F = spaces.Discrete(3+self.n_objs)
        self.observation_space_F = spaces.MultiBinary(2+2)                            # how many object has been found
                                                                                      # which turtlebot is beside the table: 2
       
        #-----------------def single step macro-actions for Fetch Robot
        F_MA0 = MacroAction('Wait_Request', 0, expected_t_cost = 1)
        F_MA1 = MacroAction('Pass_Obj_T0', 1, expected_t_cost = 4)
        F_MA2 = MacroAction('Pass_Obj_T1', 2, expected_t_cost = 4)
        self.F_MAs = [F_MA0, F_MA1, F_MA2]
        for i in range(self.n_objs):
            F_MA = MacroAction('Look_For_obj'+str(i), i+3, expected_t_cost=6)
            self.F_MAs.append(F_MA)

    def createAgents(self):

        #-----------------initialize Two Turtlebot agents
        Turtlebot1 = AgentTurtlebot_v4(0, 3.0, 1.5, self.BWPs_T0, self.T_MAs, self.n_objs, speed=self.turtlebot_speed)
        Turtlebot2 = AgentTurtlebot_v4(1, 3.0, 2.5, self.BWPs_T1, self.T_MAs, self.n_objs, speed=self.turtlebot_speed)
        
        #-----------------initialize One Fetch Robot agent
        Fetch_robot = AgentFetch_v4(2, 0.9, 1.8, self.F_MAs, self.n_objs, self.n_each_obj)

        self.agents = [Turtlebot1, Turtlebot2, Fetch_robot]

    def step(self, actions, debug=False):

        """
        Parameters
        ----------
        actions : int | List[..]
           The discrete macro-action index for one or more agents. 
        Returns
        -------
        cur_actions : int | List[..]
            The discrete macro-action indice which agents are executing in the current step.
        observations : ndarry | List[..]
            A list of  each agent's macor-observation.
        rewards : float | List[..]
            A global shared reward for each agent.
        done : bool
            Whether the current episode is over or not.
        cur_action_done : binary (1/0) | List[..]
            Whether each agent's curent macro-action is done or not.
        """

        rewards = -1.0
        terminate = 0
        cur_actions= []
        cur_actions_done= []

        self.count_step += 1

        # Each Turtlebot executes one step
        for idx, turtlebot in enumerate(self.agents[0:2]):
            # when the previous macro-action has not been finished, and return the previous action id
            if not turtlebot.cur_action_done:
                reward = turtlebot.step(turtlebot.cur_action.idx, self.humans)
                cur_actions.append(turtlebot.cur_action.idx)
            else:
                reward = turtlebot.step(actions[idx], self.humans)
                cur_actions.append(actions[idx])
            rewards += reward

        # Fetch executes one step
        if not self.agents[2].cur_action_done:
            reward = self.agents[2].step(self.agents[2].cur_action.idx, self.agents)
            cur_actions.append(self.agents[2].cur_action.idx)
        else:
            reward = self.agents[2].step(actions[2], self.agents)
            cur_actions.append(actions[2])
        rewards += reward

        # collect the info about the cur_actions' execution status
        for idx, agent in enumerate(self.agents):
            cur_actions_done.append(1 if agent.cur_action_done else 0)
        
        if (self.humans[0].cur_step == self.n_steps_human_task-2) and self.humans[0].next_requested_obj_obtained:
            terminate = 1

        # each human executes one step
        for idx, human in enumerate(self.humans):
            if idx in self.n_human_finished:
                continue
            human.cur_step_time_left -= 1.0
            if human.cur_step_time_left <= 0.0 and human.next_requested_obj_obtained:
                if human.cur_step_time_left < 0.0:
                    self.human_waiting_steps.append(human.cur_step_time_left)
                human.step()
            if human.whole_task_finished:
                self.n_human_finished.append(idx)

        self.render()
        # save the rendering
        if self.viewer != None:
            g_frames.append(self.viewer.get_array())
            print("length of frames:", len(g_frames))
            if len(g_frames) >= 500:
                from moviepy.editor import ImageSequenceClip
                clip = ImageSequenceClip(g_frames, fps=5)
                clip.write_videofile("D:/xubo92/hrl-mappo-server/visuals/td_frames.mp4")
                sys.pause()

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            print("Turtlebot0 \t action \t\t{}".format(self.agents[0].cur_action.name))
            print("           \t action_t_left \t\t{}".format(self.agents[0].cur_action_time_left))
            print("           \t action_done \t\t{}".format(self.agents[0].cur_action_done))
            print("Turtlebot1 \t action \t\t{}".format(self.agents[1].cur_action.name))
            print("           \t action_t_left \t\t{}".format(self.agents[1].cur_action_time_left))
            print("           \t action_done \t\t{}".format(self.agents[1].cur_action_done))
            print("Fetchrobot \t action \t\t{}".format(self.agents[2].cur_action.name))
            print("           \t action_t_left \t\t{}".format(self.agents[2].cur_action_time_left))
            print("           \t action_done \t\t{}".format(self.agents[2].cur_action_done))
            print("           \t is_serving \t\t{}".format(self.agents[2].serving))
            print("           \t serving_failed \t{}".format(self.agents[2].serving_failed))

        observations = self._getobs(debug)

        # reset Turtlebot request.
        self.agents[0].request_objs = np.zeros(self.n_objs)
        self.agents[1].request_objs = np.zeros(self.n_objs)

        if debug:
            print("")
            print("Humans status:")
            for idx, human in enumerate(self.humans):
                print("Human" + str(idx) + " \t\t cur_step  \t\t\t{}".format(human.cur_step))
                print("      " + " \t\t cur_step_t_left  \t\t{}".format(human.cur_step_time_left))
                print("      " + " \t\t next_request_obj  \t\t{}".format(human.next_request_obj_idx))
                print("      " + " \t\t requested_obj_obtain  \t\t{}".format(human.next_requested_obj_obtained))
                print("      " + " \t\t whole_task_finished  \t\t{}".format(human.whole_task_finished))
                print(" ")

        return cur_actions, observations, [rewards]*self.n_agent, terminate or self.count_step == self.terminate_step, cur_actions_done, [terminate, self.count_step == self.terminate_step]
         

    def _getobs(self, debug=False):

        #--------------------get observations at the beginning of each episode
        if self.t == 0:
            # get initial observation for turtlebot0
            T_obs_0 = np.zeros(self.observation_space_T.n)
            T_obs_0[len(self.BWPs_T0)] = 1.0

            # get initial observation for turtlebot1
            T_obs_1 = np.zeros(self.observation_space_T.n)
            T_obs_1[len(self.BWPs_T1)] = 1.0

            # get initial observaion for fetch robot
            F_obs = np.zeros(self.observation_space_F.n) 

            observations = [T_obs_0, T_obs_1, F_obs]
            self.t = 1
            self.old_observations = observations

            return observations

        #---------------------get observations for the two turtlebots
        if debug:
            print("")
            print("observations list:")

        observations = []
        for idx, agent in enumerate(self.agents[0:2]):

            # won't get new obs until current macro-action finished
            if not agent.cur_action_done:
                observations.append(self.old_observations[idx])
                if debug:
                    print("turtlebot" + str(idx) + " \t loc  \t\t\t{}".format(self.old_observations[idx][0:(len(self.BWPs_T0)+1)]))
                    print("          " + " \t hm_cur_step \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1):(len(self.BWPs_T0)+1)+self.n_steps_human_task]))
                    print("          " + " \t basket_objs \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1)+self.n_steps_human_task:(len(self.BWPs_T0)+1)+self.n_steps_human_task+self.n_objs]))
                    print("          " + " \t obj_ready_for_t# \t{}".format(self.old_observations[idx][-2:]))
                    print("")

                continue

            # get observation about location
            T_obs_0 = np.zeros(len(self.BWPs_T0)+1)
            if agent.cur_BWP is not None:
                T_obs_0[agent.cur_BWP.idx] = 1.0
            else:
                T_obs_0[-1] = 1.0
            BWP =agent.cur_BWP

            if debug:
                print("Turtlebot" + str(idx) + " \t loc  \t\t\t{}".format(T_obs_0))

            # get observation about the human's current working step
            T_obs_1 = np.zeros(self.n_steps_human_task)
            if BWP is not None and BWP.idx < len(self.humans):               #tweak depends on number of humans
                T_obs_1[self.humans[BWP.idx].cur_step] = 1.0

            if debug:
                print("          " + " \t Hm_cur_step \t\t{}".format(T_obs_1))

            # get observation about which obj is in the basket
            T_obs_3 = agent.objs_in_basket

            if debug:
                print("          " + " \t Basket_objs \t\t{}".format(T_obs_3))

            # get observation about which turtlebot's tool is ready (This message only can be received in Tool Room)
            if BWP is None or BWP.idx > 0:
                T_obs_4 = self.agents[2].ready_objs
            else:
                T_obs_4 = np.zeros(2)

            if debug:
                print("          " + " \t Obj_ready \t\t{}".format(T_obs_4))
                print("")

            # collect obs to be an array with shape (self.observation_space_T.n, )
            T_obs = np.hstack((T_obs_0, T_obs_1, T_obs_3, T_obs_4))
            assert len(T_obs) == self.observation_space_T.n

            observations.append(T_obs)
            self.old_observations[idx] = T_obs

        #--------------------get observations for Fetch robot
        if not self.agents[2].cur_action_done:
            observations.append(self.old_observations[2])
            if debug:
                print("Fetchrobot" + " \t obj_ready  \t\t{}".format(self.old_observations[2][0:2]))
                print("          " + " \t T#_beside_table  \t{}".format(self.old_observations[2][2:4]))
                print(" ")
                print("          " + " \t Found_objs  \t{}".format(self.agents[2].found_objs))
        else:
            # get observation about how many objects are ready
            F_obs_0 = self.agents[2].ready_objs
            
            if debug:
                print("Fetchrobot" + " \t obj_ready  \t\t{}".format(F_obs_0))

            # get observation about which turtlebot is beside the table
            F_obs_1 = np.zeros(2)
            for idx, agent in enumerate(self.agents[0:2]):
                if agent.xcoord == agent.BWPs[-1].xcoord and agent.ycoord == agent.BWPs[-1].ycoord:
                    F_obs_1[idx] = 1.0
            
            if debug:
                print("          " + " \t T#_beside_table  \t{}".format(F_obs_1))

            if debug:
                print("          " + " \t Found_objs  \t{}".format(self.agents[2].found_objs))

            # collect obs to be an array with shape (self.observation_space_F.n, )
            F_obs = np.hstack((F_obs_0, F_obs_1))
            assert len(F_obs) == self.observation_space_F.n
            self.old_observations[2] = F_obs
            observations.append(F_obs)

        return observations


from gym.spaces import Discrete

class toolDeliverySeparate(ObjSearchDelivery_v4):
    def __init__(self, *args, **kwargs):
        super(toolDeliverySeparate, self).__init__(*args, **kwargs)
        self.agent_names = ["turtlebot1", "turtlebot2", "fetchbot"]
        self.agt_avail_acts = []
        self.kwargs = kwargs

        self.frames = []

        if kwargs["all_args"].scheme == "partial-dec": # \pi(a1,a2|o1) and \pi(a1,a2|o2); 
            # assert kwargs["all_args"].use_centralized_V
            self.observation_space = [[13], [13], [4]]
            self.share_observation_space = [[13 + 13 + 4]] *3
            self.action_space = [Discrete(54), Discrete(54), Discrete(3*3*6)]
            for i in range(len(self.agent_names)):
                if self.agent_names[i] in ["turtlebot1" , "turtlebot2"]:
                    self.agt_avail_acts.append(np.concatenate(([1,1,1], [0]*51)))
                else:
                    self.agt_avail_acts.append([1] * 54)
            # self.action_space = [Discrete(3*3*6), Discrete(3*3*6), Discrete(3*3*6)]
            # for i in range(len(self.agent_names)):
            #     self.agt_avail_acts.append([1] * 3 * 3 * 6)
            self.act_table = {}
            for i in range(3):
                for j in range(3):
                    for k in range(6):
                        self.act_table[i*18+j*6+k] = [i, j, k]
            self.marginal_act = [[0,1,2], [0,1,2], [0,1,2,3,4,5]]

        elif kwargs["all_args"].scheme == "partial-cen":
            assert not kwargs["all_args"].use_centralized_V  # make it not use centralized V, otherwise the share observation is too high-dimensional
            self.observation_space = [[13 + 13 + 4], [13 + 13 + 4], [13 + 13 + 4]]
            self.share_observation_space = [[13 + 13 + 4]] *3
            self.action_space = [Discrete(6), Discrete(6), Discrete(6)]

            for i in range(len(self.agent_names)):
                if self.agent_names[i] in ["turtlebot1" , "turtlebot2"]:
                    self.agt_avail_acts.append([1, 1, 1, 0, 0, 0])
                elif self.agent_names[i] == "fetchbot":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1, 1])

        elif kwargs["all_args"].scheme == "fully-dec":
            if kwargs["all_args"].use_centralized_V:
                self.observation_space = [[13], [13], [4]]
                self.share_observation_space = [[13 + 13 + 4]]*3
                self.action_space = [Discrete(6), Discrete(6), Discrete(6)]
            else:
                self.observation_space = [[13], [13], [4]]
                self.share_observation_space = [[13], [13], [4]]
                self.action_space = [Discrete(6), Discrete(6), Discrete(6)]

            for i in range(len(self.agent_names)):
                if self.agent_names[i] in ["turtlebot1" , "turtlebot2"]:
                    self.agt_avail_acts.append([1, 1, 1, 0, 0, 0])
                elif self.agent_names[i] == "fetchbot":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1, 1])

        elif kwargs["all_args"].scheme == "sync-wait":
            assert kwargs["all_args"].use_centralized_V
            self.observation_space = [[13], [13], [4]]
            self.share_observation_space = [[13 + 13 + 4]]*3
            self.action_space = [Discrete(6), Discrete(6), Discrete(6)]

            for i in range(len(self.agent_names)):
                if self.agent_names[i] in ["turtlebot1" , "turtlebot2"]:
                    self.agt_avail_acts.append([1, 1, 1, 0, 0, 0])
                elif self.agent_names[i] == "fetchbot":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1, 1])

        elif kwargs["all_args"].scheme == "sync-cut":
            assert kwargs["all_args"].use_centralized_V
            self.observation_space = [[13], [13], [4]]
            self.share_observation_space = [[13 + 13 + 4]]*3
            self.action_space = [Discrete(6), Discrete(6), Discrete(6)]

            for i in range(len(self.agent_names)):
                if self.agent_names[i] in ["turtlebot1" , "turtlebot2"]:
                    self.agt_avail_acts.append([1, 1, 1, 0, 0, 0])
                elif self.agent_names[i] == "fetchbot":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1, 1])
        else:
            raise ValueError("invalid training scheme for tool delivery task ...")

        self.OSDv4_parent = ObjSearchDelivery_v4(*args, **kwargs)
        if self.kwargs["all_args"].scheme == "sync-cut":
            self.OSDv4_parent.terminate_step = 100
            self.OSDv4_parent.turtlebot_speed = 0.4

        self.cur_ep_steps = 0
        self.max_ep_steps = 100

        self.basic_steps = 0

    def step(self, actions):
        local_obs, shared_obs, reward = None, None, None
        dones = [False] * len(self.agent_names)
        infos = []
        terminate = False
        cur_actions_done = np.array([0,0,0])
        if self.kwargs["all_args"].scheme == "sync-wait":
            self.OSDv4_parent.agents[0].sync_wait_flag = False
            self.OSDv4_parent.agents[1].sync_wait_flag = False

        while (not terminate and not np.any(cur_actions_done)):
            cur_actions, observations, rewards, terminate, cur_actions_done, termi_info = self.OSDv4_parent.step(np.squeeze(actions), debug=False)
            local_obs = observations
            shared_obs = observations
            reward = rewards
            self.basic_steps += 1
            

        if self.kwargs["all_args"].scheme == "sync-wait":
            while (not terminate and not np.all(cur_actions_done)):
                if cur_actions_done[2]:
                    actions[2, 0] = 0
                if cur_actions_done[0] and cur_actions[0] == 2:
                    self.OSDv4_parent.agents[0].sync_wait_flag = True
                if cur_actions_done[1] and cur_actions[1] == 2:
                    self.OSDv4_parent.agents[1].sync_wait_flag = True

                cur_actions, observations, rewards, terminate, cur_actions_done, termi_info = self.OSDv4_parent.step(np.squeeze(actions), debug=False)
                local_obs = observations
                shared_obs = observations
                reward = rewards
                self.basic_steps += 1
                

        if self.kwargs["all_args"].scheme == "partial-cen":
            local_obs  = [np.concatenate(observations)] * len(self.agent_names)
            shared_obs = [np.concatenate(observations)] * len(self.agent_names)

        if self.cur_ep_steps >= self.max_ep_steps or terminate:
            dones = [True] * len(self.agent_names)
            if self.cur_ep_steps >= self.max_ep_steps:
                logger.log("episode done because of max option step")  
            elif termi_info[0]:
                logger.log("episode done because of task complete")
            elif termi_info[1]:
                logger.log("episode done because of max primitive steps")

        
        for i in range(len(self.agent_names)):
            infos.append({"bad_transition":False})
            infos[i]["act"] = cur_actions[i]
            if cur_actions_done[i]:
                infos[i]["act_finished"] = True
            else:
                infos[i]["act_finished"] = False
            infos[i]["termi_info"] = termi_info
            infos[i]["basic_steps"] = self.basic_steps
        
        return local_obs, shared_obs, reward, np.array(dones), infos, self.agt_avail_acts 

    def reset(self):
        observations = self.OSDv4_parent.reset()
        local_obs, shared_obs = observations, observations

        if self.kwargs["all_args"].scheme == "partial-cen":
            local_obs  = [np.concatenate(observations)] * len(self.agent_names)
            shared_obs = [np.concatenate(observations)] * len(self.agent_names)

        return local_obs, shared_obs, self.agt_avail_acts

    def _get_avail_actions(self, agtId):
        return self.agt_avail_acts[agtId]

    def seed(self, seed):
        return seed

    def close(self):
        return 0

if __name__ == "__main__":
    test_env = ObjSearchDelivery_v4()
    ob = test_env.reset()
    i = 0
    while (i <= 150):
        turtlebot_1_act = np.random.choice([0,1,2])
        turtlebot_2_act = np.random.choice([0,1,2])
        fetch_act = np.random.choice([0,1,2,3,4,5])
        cur_actions, observations, rewards, done, cur_action_done = test_env.step([turtlebot_1_act, turtlebot_2_act, fetch_act])
        i += 1
