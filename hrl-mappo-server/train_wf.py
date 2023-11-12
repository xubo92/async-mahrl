#!/usr/bin/env python
import sys
sys.path.remove('d:\\xubo92\\smac')
sys.path.remove('d:\\xubo92\\on-policy')
sys.path.append("D:/xubo92/hrl-mappo-server/on-policy")
# sys.path.append("D:/xubo92/ai2thor-docker")
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from misc import logger
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from water_fill import waterFill
from water_fill_cond import waterFillCond
from water_fill_separate import waterFillSeparate
# from ai2thor_docker.x_server import startx

"""Train script for waterFill."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "waterFill":
                env = waterFill(all_args)
            elif all_args.env_name == "waterFillCond":
                env = waterFillCond(all_args)
            elif all_args.env_name == "waterFillSeparate":
                env = waterFillSeparate(all_args)
            else:
                logger.log("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "waterFill":
                env = waterFill(all_args)
            elif all_args.env_name == "waterFillCond":
                env = waterFillCond(all_args)
            elif all_args.env_name == "waterFillSeparate":
                env = waterFillSeparate(all_args)
            else:
                logger.log("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='FloorPlan_Train3_1', help="Which indoor map to run on")
    parser.add_argument('--mode', type=str, default='multi')
    parser.add_argument("--ac_sp", type=str, default='full', help="full: basic actions included; partial: only high-level Options included; combi: joint combination of action space")
    parser.add_argument("--rew_type", type=str, default='abs')

    parser.add_argument("--conditioned", action="store_true", default=False, help="if use conditional distribution when one agent is not choosing new action")
    parser.add_argument("--baseline", action="store_true", default=False, help="if run baseline setup")
    all_args = parser.parse_known_args(args)[0]

    

    return all_args


def main(args):
    # startx()
    parser = get_config()
    all_args = parse_args(args, parser)


    all_args.env_name = "waterFill"
    all_args.algorithm_name = "mappo"
    all_args.experiment_name = "mlp"
    all_args.map_name = "FloorPlan_Train3_1"
    all_args.n_training_threads = 1
    all_args.n_rollout_threads = 1 
    all_args.num_mini_batch = 1 
    all_args.num_env_steps = 200000 # default use 200000
    all_args.ppo_epoch = 10 
    all_args.episode_length = 1600  # default use 1600
    all_args.log_interval = 1

    # load pre-trained model to evaluate
    # all_args.model_dir = "./results/waterFill/FloorPlan_Train3_1/mappo/mlp/run27/models"
 
    # Uncomment these 4 lines if conditional case is used
    # all_args.env_name = "waterFillCond"
    # all_args.mode = "multi"
    # all_args.ac_sp = "combi"
    # all_args.conditioned = True

    # Uncomment these 3 lines if baseline is used
    # all_args.baseline = True
    # all_args.mode = "single"
    # all_args.ac_sp = "basic"

    # Uncomment these lines if use separate policy (shared policy is not used)
    # all_args.share_policy = False
    # all_args.env_name = "waterFillSeparate"
    # all_args.use_centralized_V = True   # run3,4,5,6 are all using false; can be true
    # all_args.use_policy_active_masks = False
    # all_args.use_value_active_masks = False


    if all_args.conditioned:
        assert all_args.mode == "multi"
        assert all_args.ac_sp == "combi"
        assert all_args.env_name == "waterFillCond"

    if all_args.baseline:
        assert all_args.mode == "single"
        assert all_args.ac_sp == "basic"
        assert all_args.conditioned == False
    
    if not all_args.share_policy:
        assert all_args.env_name == "waterFillSeparate"
        assert all_args.use_policy_active_masks == False
        assert all_args.use_value_active_masks == False 

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logger.log("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logger.log("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path("./" + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    # setup logger
    logger.configure(dir=str(run_dir), format_strs=['log'])
    logger.log(all_args)
    
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    if all_args.conditioned and  all_args.mode == "multi" and all_args.ac_sp == "combi":
        num_agents = 1
    else:
        num_agents = 2

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from wf_runner import WfRunner as Runner
    else:
        from wf_runner_separate import WfRunnerSeparate as Runner
    
    print(all_args)

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])
