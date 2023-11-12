import sys
sys.path.append("/home/xubo92/ai2thor")
sys.path.append("/home/xubo92/hrl-mappo")
import ai2thor
from ai2thor.controller import Controller
import time
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import copy
import torch
from gym.spaces import Discrete

from misc.utils import get_shortest_path, get_rot, dist2d, ResNetEmbedder
from misc import logger as logger

resnet = ResNetEmbedder()

# Bottle:33; Cup:51; Mug:53
class waterFillSeparate(object):
    """
    A customized env for two agents water-filling task
    :: agent 1: locobot
    :: agent 2: drone
    :: scene: loorPlan_Train3_1
    """
    def __init__(self, args, scene="FloorPlan_Train3_1"):
        
        self.args = args
        self.scene_width  = 224
        self.scene_height = 224

        # agent and object typest
        self.agent_types = ['locobot', 'drone']
        self.container_types = ['Bottle', 'Cup', 'Mug']

        # setup agent action space
        if self.args.ac_sp == "full":
            self.action_list = ["idle", "go2cup", "go2mug", "go2btl", "go2cup2fill", "go2mug2fill", "go2btl2fill", "MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]
        elif self.args.ac_sp == "partial":
            self.action_list = ["idle", "go2cup", "go2mug", "go2btl", "go2cup2fill", "go2mug2fill", "go2btl2fill"]
        elif self.args.ac_sp == "combi":
            self.action_list = []
            for i, locobot_act in enumerate(["go2cup2fill", "go2mug2fill", "go2btl2fill"]):
                for j, drone_act in enumerate(["go2cup", "go2mug", "go2btl"]):
                    self.action_list.append(locobot_act + "_" + drone_act)
        elif self.args.ac_sp == "basic":
            self.action_list = ["idle", "MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]
        else:
            raise ValueError("invalid action space type")
        

        # num of agents and objects
        self.num_agents = len(self.agent_types)
        self.num_containers = len(self.container_types)

        # initialize controller, scene and two agents
        self.scene = scene 
        self.controller = Controller(agentMode=self.agent_types[0], agentCount=2, scene=self.scene,  width=self.scene_width, height=self.scene_height)
        event = self.controller.step(action="GetMapViewCameraProperties")
        event = self.controller.step(action="AddThirdPartyCamera", agentId=0, **event.metadata["actionReturn"])
        
        # Set up top-down view frame
        self.top_down_frame = event.events[0].third_party_camera_frames[0]

        sce_xmin = event.events[0].metadata["sceneBounds"]["cornerPoints"][-1][0]
        sce_xmax = event.events[0].metadata["sceneBounds"]["cornerPoints"][0][0]
        sce_zmin = event.events[0].metadata["sceneBounds"]["cornerPoints"][-1][2]
        sce_zmax = event.events[0].metadata["sceneBounds"]["cornerPoints"][0][2]
        self.scene_size = {"x_min": sce_xmin, "x_max":sce_xmax, "z_min":sce_zmin, "z_max": sce_zmax}

        self.container_poses = {}
        for ctn in self.container_types: 
            for k, obj in enumerate(event.events[0].metadata["objects"]):
                if obj["objectType"] == ctn:
                    self.container_poses[ctn] =  obj["position"]
                    break
        # container index in metadata["objects"]            
        self.container_index = {"Bottle":33, "Cup":51, "Mug":53}
        self.container_Ids   = {"Bottle":'Bottle|+03.05|+00.67|-03.03', "Cup": 'Cup|+05.06|+00.60|-01.21', "Mug":'Mug|+08.34|+00.51|-02.65'}

        # for 640 * 480 frame
        if self.scene_width == 640 and self.scene_height == 480:
            self.container_poses_pix = {"Cup":(296,183), "Bottle":(198,274), "Mug":(456,254)}  
            self.container_draw_color = {"Cup":(255,255,255), "Bottle":(255,0,255), "Mug":(0,255,0)}
            self.container_rect_start = {"Cup":(150,50), "Bottle":(300,50), "Mug":(450,50)}
            self.container_rect_end = {"Cup":(180,100), "Bottle":(330,100), "Mug":(480,100)}
        elif self.scene_width == 224 and self.scene_height == 224:
            # for 224*224 frame
            self.container_poses_pix = {"Cup":(100,84), "Bottle":(54,127), "Mug":(175,118)}  
            self.container_draw_color = {"Cup":(255,255,255), "Bottle":(255,0,255), "Mug":(0,255,0)}
            self.container_rect_start = {"Cup":(50,20), "Bottle":(100,20), "Mug":(150,20)}
            self.container_rect_end = {"Cup":(70,40), "Bottle":(120,40), "Mug":(170,40)}
        else:
            raise ValueError("Invald scene size")

        

        # Specify the slow and fast agent speed (w.r.t grid points)
        self.speeds = {"locobot": 1, "drone": 4}

        

        # initialize the water levels for all containers (not visible to agents)
        self.g_water_levels = [1.0] * self.num_containers # 1 as full
        self.g_water_levels_hist = [1.0] * self.num_containers
        
        
        self.water_dec_rate = None   # how much water will decrease in this step
        self.water_dec_intv = None   # interval between current and next water dec

        # Max episodic steps
        self.cur_ep_steps = 0                          # this is Option step
        self.max_ep_steps = 200                        # Maximum number of Option steps
        self.basic_steps  = 0
        
        # Engineered state variables
        self.last_known_water_levels = copy.deepcopy(self.g_water_levels)
        self.elapsed_time_from_lkwl = [0.] * self.num_containers # elapsed timesteps from last known water levels

        # specially for two agents and 3 cups
        self.agt_avail_acts = []
        for i in range(self.num_agents):
            if self.agent_types[i] == "locobot":
                if self.args.ac_sp == "full":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # previous good tests by default think the locobot cannot seperate reach and fill
                elif self.args.ac_sp == "partial":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1, 1, 1])
                elif self.args.ac_sp == "basic":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1])
                else:
                    raise ValueError("invalid action space type")
            elif self.agent_types[i] == "drone":
                if self.args.ac_sp == "full":
                    self.agt_avail_acts.append([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
                elif self.args.ac_sp == "partial":
                    self.agt_avail_acts.append([1, 1, 1, 1, 0, 0, 0])
                elif self.args.ac_sp == "basic":
                    self.agt_avail_acts.append([1, 1, 1, 1, 1])
                else:
                    raise ValueError("invalid action space type")
            else:
                raise ValueError("invalid agent type")
        
        if self.args.share_policy:
            self.observation_space = [[518, []]*self.num_agents]
            self.share_observation_space = [[518, []]*self.num_agents]
            self.action_space = [Discrete(len(self.action_list))]*self.num_agents
        else:
            if self.args.use_centralized_V:
                self.observation_space = [[518]]*self.num_agents
                self.share_observation_space = [[518*self.num_agents]]*self.num_agents
                self.action_space = [Discrete(len(self.action_list))]*self.num_agents
            else:
                self.observation_space = [[518]]*self.num_agents
                self.share_observation_space = [[518]]*self.num_agents
                self.action_space = [Discrete(len(self.action_list))]*self.num_agents

    def reset(self):
        """
        :: forced reset, only when max_ep_steps is reached
        :: reset scene, water level, agent initial pos
        :: return: obs ->  [img * num_agents; timestep_to_now; -1 * num_containers]
        :: -1 as unknown water level; 0 as empty; 0~1 as known water level (when agent is around)
        """
        # reset scene
        self.controller.reset(agentMode=self.agent_types[0], scene=self.scene, agentCount=2)
        event = self.controller.step(action="GetMapViewCameraProperties")
        event = self.controller.step(action="AddThirdPartyCamera", agentId=0, **event.metadata["actionReturn"])

        # reset global water info
        self.g_water_levels = [1.0] * self.num_containers
        self.g_water_levels_hist = [1.0] * self.num_containers

        # retrieve rgb imgs for each agent
        multi_evts = self.controller.last_event.events
        rgb_imgs = [evt.frame for evt in multi_evts]

        # retrieve local water level, visible to agents, as partial state  
        self.last_known_water_levels = copy.deepcopy(self.g_water_levels)
        self.elapsed_time_from_lkwl  = [0] * self.num_containers
       
        # Update the frame
        self.top_down_frame = multi_evts[0].third_party_camera_frames[0]
        self.cur_ep_steps = 0
        self.basic_steps = 0
        

        # render
        # cv2.imshow("locobot view", rgb_imgs[0].astype(np.uint8))
        # cv2.imshow("drone view", rgb_imgs[1].astype(np.uint8))
        # cv2.waitKey(1)


        # retrieve obs
        local_obs, shared_obs, available_actions = [], [], []  # len = 3 for each of it
        for i, obs_img in enumerate(rgb_imgs):
            torch_im = torch.from_numpy(np.transpose(obs_img, (2,0,1)).copy()).float()
            torch_imvec = resnet(torch_im.unsqueeze(0))  # return (1, 512)
            np_imvec = torch_imvec.numpy()
            np_imvec = np.squeeze(np_imvec)  # return (512, )
            tot_info = np.concatenate((np_imvec, self.last_known_water_levels, self.elapsed_time_from_lkwl))
            local_obs.append(tot_info)
            shared_obs.append(tot_info)
            available_actions.append(self.agt_avail_acts[i])
        
        # generate random params for water level changes, for all containers
        base_dura = np.random.randint(5,15, size=self.num_containers)
        dura = np.random.exponential(base_dura, size=self.num_containers)
        dura = np.round(dura)
        self.water_dec_intv = dura
        base_water_dec = self.water_dec_intv * 1.0 / 500.0
        self.water_dec_rate = np.random.normal(base_water_dec, scale=0.2)
        self.water_dec_rate = np.abs(self.water_dec_rate)
        
        logger.log("reset... current water level:", self.g_water_levels)
        # print("reset... current water level:", self.g_water_levels)

        
        return local_obs, shared_obs, available_actions

    
    def step(self, acts):
        """
        acts: shape: [num_agent,1]; nparray; index of available actions for each agent
        "Collision as soft reset, which means only give penalty but no pose reset"
        """
        isActsFinished = [False] * self.num_agents
        while not np.any(isActsFinished):
            # update global water info
            for i in range(self.num_containers):
                if self.water_dec_intv[i] != 0:
                    self.water_dec_intv[i] -= 1
                else:
                    self.g_water_levels_hist[i] = self.g_water_levels[i] # update g_water_levels
                    self.g_water_levels[i] -= self.water_dec_rate[i]
                    if self.g_water_levels[i] <= 0:
                        self.g_water_levels[i] = 0
    
                    # generate random params of water change for the specific container
                    base_dura = np.random.randint(5,15)
                    dura = np.random.exponential(base_dura)
                    dura = np.round(dura)
                    self.water_dec_intv[i] = dura
                    base_water_dec = self.water_dec_intv[i] * 1.0 / 500.0
                    self.water_dec_rate[i] = np.random.normal(base_water_dec, scale=0.2)
                    self.water_dec_rate[i] = np.abs(self.water_dec_rate[i])
        
            # apply one Option for each agent
            isCollided, isLwUpdated, isCtnVis = self._take_moves(acts)
            self.basic_steps += 1
            # print("basic step num:", self.basic_steps)
    
            # Check if current action is finished (specially for multi-step Option)
            if self.args.mode == "single":
                isActsFinished = [True] * self.num_agents
            elif self.args.mode == "multi":
                isActsFinished = [False] * self.num_agents
    
                # convert acts to the format "mover_move"
                str_acts = []
                for idx in range(acts.shape[0]):
                    str_acts.append(self.agent_types[idx] + "_" + self.action_list[acts[idx][0]])
    
                for ia, act in enumerate(str_acts):
                    if len(act.split("2")) > 1:
                        act_ctn = act.split("2")[1]
                        act_ctn = act_ctn.capitalize() if act_ctn != "btl" else "Bottle"
                        if isLwUpdated[act_ctn] and act_ctn in isCtnVis[self.agent_types[ia]].keys():
                            isActsFinished[ia] = True
                    else:
                        isActsFinished[ia] = True
            else:
                raise ValueError("invalid mode")
    
            # Update observations
            rgb_imgs = [evt.frame for evt in self.controller.last_event.events] # retrieve imgs 
            for k,v in isLwUpdated.items():
                ctn_index = self.container_types.index(k)
                if isLwUpdated[k]:
                    self.last_known_water_levels[ctn_index] = self.g_water_levels[ctn_index] # update local wate levels
                    self.elapsed_time_from_lkwl[ctn_index] = 0  # update elapsed time
                else:
                    self.elapsed_time_from_lkwl[ctn_index] += 1.0 / self.max_ep_steps
            
            # calculate rew
            rew_tot, rew_wat, rew_coll, rew_targ_view, rew_fw_pel = 0, 0, 0, 0, 0
            if self.args.rew_type == "abs":
                # - use min water level
                # rew_wat = -1.0/(np.min(self.g_water_levels)+1e-3) + 1
                
                # - use sum of inverse function
                for i in range(len(self.g_water_levels)):
                    rew_wat += (-1.0/(self.g_water_levels[i]+1e-3)+1)
                
                # - use inverse of avg 
#                rew_wat += -1.0/(np.mean(self.g_water_levels)+1e-3) + 1
                
                # total rew
                rew_tot +=  rew_wat + rew_coll + rew_targ_view + rew_fw_pel
                
            elif self.args.rew_type == "inc":
                sum_water = np.sum(self.g_water_levels)
                sum_water_hist = np.sum(self.g_water_levels_hist)
                rew_tot = (sum_water - sum_water_hist)*10
                # TODO: consider when water level is zero
            else:
                raise ValueError("invalid reward type")
            reward = [[rew_tot]] * self.num_agents
                
    
            # update top-down frame
            self.top_down_frame = self.controller.last_event.events[0].third_party_camera_frames[0]
            # cv2.imshow("locobot view", rgb_imgs[0].astype(np.uint8))
            # cv2.imshow("drone view", rgb_imgs[1].astype(np.uint8))
            # self.render()
    
            # retrieve obs
            local_obs, shared_obs, available_actions = [], [], []  # len = 3 for each of it
            for i, obs_img in enumerate(rgb_imgs):
                torch_im = torch.from_numpy(np.transpose(obs_img, (2,0,1)).copy()).float()
                torch_imvec = resnet(torch_im.unsqueeze(0))  # return torch_imvec as [1, 512]
                np_imvec = torch_imvec.numpy()
                np_imvec = np.squeeze(np_imvec)  # return (512, )
                tot_info = np.concatenate((np_imvec, self.last_known_water_levels, self.elapsed_time_from_lkwl))
                local_obs.append(tot_info)
                shared_obs.append(tot_info)
                available_actions.append(self.agt_avail_acts[i])
    
                
                # hidd_feat = resnet.vis_any_feature(torch_im.unsqueeze(0), 'conv1')
                # hidd_feat = hidd_feat[:, 63, :, :]
                # hidd_feat = hidd_feat.view(hidd_feat.shape[1], hidd_feat.shape[2])
                # hidd_feat = hidd_feat.data.numpy()
    
                # cv2.imshow("hidden feat " + str(i), hidd_feat)
    
    
        # Update step counter. 
        self.cur_ep_steps += 1
        # print("ep step:", self.cur_ep_steps)
            
        # Update done flag
        done = [False] * self.num_agents
        if self.cur_ep_steps >= self.max_ep_steps:
            done = [True] * self.num_agents
    
        # update info (to comply with mappo code)
        info = []
        for i in range(self.num_agents):
            info.append({"bad_transition":False})
            info[i]["act"] = acts[i][0]
            if isActsFinished[i]:
                info[i]["act_finished"] = True
            else:
                info[i]["act_finished"] = False
            info[i]["basic_steps"] = self.basic_steps
        
        logger.log(reward[0])
        logger.log("current water level:", self.g_water_levels)
        # print(reward[0])
        # print("current water level:", self.g_water_levels)
    
        return local_obs, shared_obs, reward, np.array(done), info, self.agt_avail_acts 
    
    def render(self):
        intf = self.top_down_frame.astype(np.uint8).copy() 
        for i, cnt in enumerate(self.container_types):
            center = self.container_poses_pix[cnt]
            color  = self.container_draw_color[cnt]
            start = self.container_rect_start[cnt] 
            end = self.container_rect_end[cnt] 
            intf = cv2.circle(intf, center, 5, color, thickness=2)
            intf = cv2.rectangle(intf, start, end, color=color, thickness=2)
            # simulated water level
            wat_start = (start[0], int(end[1]-self.g_water_levels[i]*(end[1]-start[1])))
            wat_end   = (end[0], end[1])
            intf = cv2.rectangle(intf, wat_start, wat_end, color=(255,255,0), thickness=-1)
        cv2.imshow("topdown", intf)
        cv2.waitKey(1)
        return intf
    
    def _take_moves(self, acts):
        """
        acts: [num_agents, 1], nparray
        """
        isCollided = [False] * self.num_agents # agent collision info
        isLwUpdated = None # local water update info
        isCtnVis = None  # containers in sight of agents info

        # convert acts to the format "mover_move"
        str_acts = []
        for idx in range(acts.shape[0]):
            str_acts.append(self.agent_types[idx] + "_" + self.action_list[acts[idx][0]])
        
        logger.log(str_acts)
        # print(str_acts)

        # run one step for all agents
        for i, act_i in enumerate(str_acts):
            mover = act_i.split("_")[0]
            move  = act_i.split("_")[1]
            # logger.log("mover: {}; move: {}".format(mover, move))
            reachable_positions = self.controller.step("GetReachablePositions", agentId=i).metadata["actionReturn"]
            agt_pos = self.controller.last_event.events[i].metadata["agent"]["position"]
            agt_rot = self.controller.last_event.events[i].metadata["agent"]["rotation"]
            # not going anywhere
            if move == "idle":
                continue
            # go to one of the containers
            elif move in ["go2cup", "go2mug", "go2btl", "go2cup2fill", "go2mug2fill", "go2btl2fill"]:
                move_info = move.split("2")
                isFill = False
                if move_info[-1] == "fill":
                    ctn_name = move_info[1]
                    ctn_name = ctn_name.capitalize()
                    if ctn_name == "Btl":
                        ctn_name = "Bottle" 
                    isFill = True
                else:
                    ctn_name = move_info[-1].capitalize()
                    if ctn_name == "Btl":
                        ctn_name = "Bottle"
                path = get_shortest_path(agt_pos, self.container_poses[ctn_name], reachable_positions)  
                if len(path) > self.speeds[mover]:
                    final_pos = path[0 + self.speeds[mover]]
                    final_pos_before = path[0 + self.speeds[mover]-1]
                    final_pos_dict = {"x":final_pos[0], "y":final_pos[1], "z":final_pos[2]}
                    final_pos_before_dict = {"x":final_pos_before[0], "y":final_pos_before[1], "z":final_pos_before[2]}
                    rot = get_rot(agt_rot, final_pos_before_dict, final_pos_dict)
                    self.controller.step(action="Teleport", position=final_pos_dict, rotation=rot, agentId=i)  
                elif len(path) <= self.speeds[mover] and len(path) >= 2:
                    final_pos = path[-1]
                    final_pos_before = path[-2]
                    final_pos_dict = {"x":final_pos[0], "y":final_pos[1], "z":final_pos[2]}
                    final_pos_before_dict = {"x":final_pos_before[0], "y":final_pos_before[1], "z":final_pos_before[2]}
                    rot = get_rot(agt_rot, final_pos_before_dict, final_pos_dict)
                    self.controller.step(action="Teleport", position=final_pos_dict, rotation=rot, agentId=i)
                else: 
                    # add last rotation to make sure object is in sight
                    metaobjs = self.controller.last_event.events[i].metadata["objects"]
                    while not metaobjs[self.container_index[ctn_name]]["visible"]:    
                        self.controller.step(action="RotateLeft", agentId=i)
                        metaobjs = self.controller.last_event.events[i].metadata["objects"]
                
                # if fillwater is needed, do it
                if isFill:
                    isFill = False
                    assert mover == "locobot" and i == 0
                    isVisible = self.controller.last_event.events[i].metadata["objects"][self.container_index[ctn_name]]['visible']
                    isReachable = self.controller.last_event.events[i].metadata["objects"][self.container_index[ctn_name]]['distance'] <= 1.0
                    if isVisible and isReachable:
                        self.g_water_levels[self.container_types.index(ctn_name)] = 1.0
            else:
                self.controller.step(action=move, agentId=i)
                if self.args.ac_sp == "basic":
                    # print("running baseline")
                    if mover == "locobot" and i == 0:
                        all_objects = self.controller.last_event.events[i].metadata["objects"]
                        for ctn_name, ctn_id in self.container_index.items():
                            isVisible   = all_objects[ctn_id]['visible']
                            isReachable = all_objects[ctn_id]['distance'] <= 1.0
                            if isVisible and isReachable:
                                self.g_water_levels[self.container_types.index(ctn_name)] = 1.0
                                print(ctn_name + " is filled")
                                break
        
        isLwUpdated, isCtnVis = self._get_local_water_level()

        return isCollided, isLwUpdated, isCtnVis

    def _get_local_water_level(self):
        # Check water level in each container & Check if the agent's location is around the cup
        LwUpdated = {}
        for i, it in enumerate(self.container_types):
            LwUpdated[it] = False
        # for each agent, save the distance of visible objects (for rew compu afterwards)
        VisUpdated = {}
        for i, it in enumerate(self.agent_types):
            VisUpdated[it] = {}

        multi_evts = self.controller.last_event.events
        for i, evt in enumerate(multi_evts):
            for j, ctn in enumerate(self.container_types):
                for k, obj in enumerate(evt.metadata["objects"]):
                    # grid_size: 0.25m; Visible distance: 1.5m (too far); use 0.6 to close enough for check the water level
                    if obj['objectType'] == ctn and obj['visible']:
                        dist = dist2d(evt.metadata["agent"]["position"], obj["position"])
                        VisUpdated[self.agent_types[i]][ctn] = dist
                        # logger.log("agent: {}; visible container: {}; distance: {}".format(self.agent_types[i], ctn, dist))
                        if dist <= 1.0:
                            LwUpdated[ctn] = True
        return LwUpdated, VisUpdated
        

    def _get_avail_actions(self, agtId):
        return self.agt_avail_acts[agtId]

    def _get_env_info(self):
        """
        obs_shape: shape of (compressed visual embedding + last_known_water_levels + elapsed_time_from_lkwl)
        """
        return {"n_actions":len(self.action_list), "n_agents":self.num_agents, "state_shape":None, "obs_shape":512+6}

    def seed(self, seed):
        return seed

    def close(self):
        return 0




if __name__ == "__main__":
    resnet = ResNetEmbedder()
    # (resnet)
    # for i, submodule in enumerate(resnet.model.modules()):
    #     print(i, submodule)
    for key,value in resnet.model._modules.items():
        print(key, value)
