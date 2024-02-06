
import numpy as np

from ppo.model import get_train_fn, get_act_fn
from ppo.runner import Runner
from ppo.policy import GPTPolicy
from utils.util import get_device
import torch
from tqdm import trange
import wandb
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from loader import SizeConstrainedLoader
import numpy as np
import gymnasium as gym

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from collections import OrderedDict

def solve(cfg, env):
    tcfg = cfg.train
    device = get_device(tcfg.gpu_num)
    policy = GPTPolicy(cfg, device).to(device)
    model_state = torch.load("/data/arcle-trajectory/withbbox/saved_parameter_1410.pth")
    new_model_state = OrderedDict([(key.replace('_orig_mod.', ''), value) for key, value in model_state.items()])
    policy.load_state_dict(new_model_state)
    policy = torch.compile(policy)

    nenvs = tcfg.nenvs
    nbatch = nenvs * tcfg.nsteps

    nbatch_train = nbatch // tcfg.nminibatches
    nupdates = tcfg.total_timesteps // nbatch

    optimizer = policy.configure_optimizers()
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1000,
                                          cycle_mult=1.0,
                                          max_lr=5e-4,
                                          min_lr=5e-5,
                                          warmup_steps=500,
                                          gamma=1.0)
    train_fn = get_train_fn(policy, optimizer, tcfg)
    act_fn = get_act_fn(policy)

    # Instantiate the runner object
    runner = Runner(env, cfg, device, act_fn)
    all_ep_rets = []

    (ob_acs, returns, values, neglogpacs, rtm1, rtm1_pred, norm_rew, rpred, gpred, gtp1,
         ep_rets, success_ts, ebbox, unwrapped, clip, input, answer) = runner.run()
    
    return unwrapped, clip, input, answer, success_ts

def render_ansi(env, grid):
        if env.rendering is None:
            env.rendering = True
            print('\033[2J',end='')

        print(f'\033[{env.H+3}A\033[K', end='')
        print('Problem Description:')
        print(env.description, '\033[K')

        grid = grid.squeeze()

        for i,dd in enumerate(grid):
            for j,d in enumerate(dd):
                
                if i >= 5 or j>= 5:
                    print('\033[47m  ', end='')
                else:
                    print("\033[48;5;"+str(env.ansi256arc[d])+"m  ", end='')

            print('\033[0m')

def render_ansi_with_grids(env, grids, clips):
        if env.rendering is None:
            env.rendering = True
            print('\033[2J',end='')
        
        print(f'\033[{env.H+3}A\033[K', end='')
        print('Problem Description:')
        print(env.description, '\033[K')

        idx = 1
        for grid, clip in zip(grids, clips):
            print("{} grid : \n".format(idx))
            idx += 1
            grid = grid.squeeze()
            clip = clip.squeeze()

            grid_dim = env.current_state['grid_dim']
            sel = env.current_state['selected']
            clip_dim = env.current_state['clip_dim']

            for i,dd in enumerate(grid):
                for j,d in enumerate(dd):
                    
                    if i >= 5 or j>= 5:
                        print('\033[47m  ', end='')
                    else:
                        print("\033[48;5;"+str(env.ansi256arc[d])+"m  ", end='')

                print('\033[0m')

@hydra.main(config_path="ppo", config_name="ppo_config")
def main(cfg: DictConfig) -> None:
    # wandb.init(
    #     project="arc_traj_gen",
    #     config=OmegaConf.to_container(cfg)
    # )
    env = gym.make(
        'ARCLE/O2ARCv2Env-v0', 
        data_loader = SizeConstrainedLoader(cfg.env.grid_x),
        max_trial = 3,
        max_grid_size=(cfg.env.grid_x, cfg.env.grid_y), 
        colors=cfg.env.num_colors,
        render_mode="ansi")
    
    gtp1, clip, input, answer, success_ts = solve(cfg, env)
    print("Input: \n\n")
    render_ansi(env, input)

    print("\n\n Answer: \n\n")
    render_ansi(env, answer)

    render_ansi_with_grids(env, gtp1, clip)

if __name__ == "__main__":
    main()
