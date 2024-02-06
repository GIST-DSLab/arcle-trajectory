import torch
import numpy as np
from copy import deepcopy
from tqdm import trange
from utils.util import RunningMeanStd

def batch_act_env(env, b_inp, b_inp_dim, b_answer, b_answer_dim,
    b_grid, b_grid_dim, b_selected, b_clip, b_clip_dim, b_terminated, b_trials_remain,
    b_active, b_object_, b_object_sel, b_object_dim, b_object_pos, b_background, b_rotation_parity,
    b_selection, b_operation):
    nb_grid = b_grid.copy()
    nb_grid_dim = b_grid_dim.copy()
    nb_selected = b_selected.copy()
    nb_clip = b_clip.copy()
    nb_clip_dim = b_clip_dim.copy()
    nb_terminated = b_terminated.copy()
    nb_trials_remain = b_trials_remain.copy()
    nb_active = b_active.copy()
    nb_object_ = b_object_.copy()
    nb_object_sel = b_object_sel.copy()
    nb_object_dim = b_object_dim.copy()
    nb_object_pos = b_object_pos.copy()
    nb_background = b_background.copy()
    nb_rotation_parity = b_rotation_parity.copy()
    reward = b_active.copy()
    for i, (
        inp, inp_dim, answer, answer_dim, grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain,
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity,
        selection, operation
    ) in enumerate(zip(
        b_inp, b_inp_dim, b_answer, b_answer_dim, b_grid, b_grid_dim, b_selected, b_clip, b_clip_dim, b_terminated, b_trials_remain,
        b_active, b_object_, b_object_sel, b_object_dim, b_object_pos, b_background, b_rotation_parity, 
        b_selection, b_operation)):

        state = {
            "trials_remain": trials_remain,
            "terminated": terminated,
            "input":inp,
            "input_dim": inp_dim,    
            "grid": grid,
            "grid_dim": grid_dim,
            "selected": selected,
            "clip" : clip,
            "clip_dim" : clip_dim,
            "object_states": {
                "active": active, 
                "object": object_,
                "object_sel": object_sel,
                "object_dim": object_dim,
                "object_pos": object_pos, 
                "background": background, 
                "rotation_parity": rotation_parity,
            }
        }
        action = {
            "selection" : selection,
            "operation" : operation,
        }
        # render_ansi(env, state)
        env.unwrapped.transition(state, action)
        rwd = env.unwrapped.reward(state)
        # render_ansi(env, state)

        (nb_grid[i], nb_grid_dim[i], nb_selected[i], nb_clip[i], nb_clip_dim[i], nb_terminated[i], nb_trials_remain[i],
        nb_active[i], nb_object_[i], nb_object_sel[i], nb_object_dim[i], nb_object_pos[i],
        nb_background[i], nb_rotation_parity[i], reward[i]) = (state['grid'], state['grid_dim'], 
        state['selected'], state['clip'], state['clip_dim'], state['terminated'], state['trials_remain'], 
        state['object_states']['active'], state['object_states']['object'], state['object_states']['object_sel'],
        state['object_states']['object_dim'], state['object_states']['object_pos'], state['object_states']['background'], state['object_states']['rotation_parity'], 
        rwd)

    return (
        nb_grid, nb_grid_dim, nb_selected, nb_clip, nb_clip_dim, nb_terminated, nb_trials_remain,
        nb_active, nb_object_, nb_object_sel, nb_object_dim, nb_object_pos, 
        nb_background, nb_rotation_parity, reward
    )

def render_ansi_with_grids(env, grids, clips):
        if env.rendering is None:
            env.rendering = True
            print('\033[2J',end='')
        
        print(f'\033[{env.H+3}A\033[K', end='')
        print('Problem Description:')
        print(env.description, '\033[K')

        for grid, clip in zip(grids, clips):
            grid = grid.squeeze()
            clip = clip.squeeze()

            grid_dim = env.current_state['grid_dim']
            sel = env.current_state['selected']
            clip_dim = env.current_state['clip_dim']

            for i in range(env.H):
                for j in range(env.W):
                    d = grid[i,j]
                    st = "[]" if sel[i,j] else "  " 
                    if i >= grid_dim[0] or j>= grid_dim[1]:
                        print(f'\033[47m{st}', end='')
                    else:
                        print("\033[48;5;"+str(env.ansi256arc[d])+f"m{st}", end='')

                print("\033[0m  ",end='')
                for j in range(env.W):
                    d = clip[i,j]
                    
                    if i >= clip_dim[0] or j>= clip_dim[1]:
                        print('\033[47m  ', end='')
                    else:
                        print("\033[48;5;"+str(env.ansi256arc[d])+"m  ", end='')               
        
                print('\033[0m')

def render_ansi(env, grid):
        if env.rendering is None:
            env.rendering = True
            print('\033[2J',end='')

        print(f'\033[{env.H+3}A\033[K', end='')
        print('Problem Description:')
        print(env.description, '\033[K')

        grid = grid.squeeze()

        state = env.current_state
        grid_dim = state['grid_dim']

        for i,dd in enumerate(grid):
            for j,d in enumerate(dd):
                
                if i >= grid_dim[0] or j>= grid_dim[1]:
                    print('\033[47m  ', end='')
                else:
                    print("\033[48;5;"+str(env.ansi256arc[d])+"m  ", end='')

            print('\033[0m')

class Runner:
    GRIDS = [
        "input", "answer", "grid", "selected", "clip",
        "object", "object_sel","background", "gpred", "gtp1"]
    TUPLES = [
        "input_dim", "answer_dim", "grid_dim", 
        "clip_dim", "object_dim", "object_pos",
        "bbox", "ebbox"]
    NUMBERS = [
        "terminated", "trials_remain",
        "active", "rotation_parity",
        "operation", "reward", "rtm1", "rpred", "rtm1_pred",
        "neglogpac", "vpred"
    ]
    INFO_KEYS = ["input", "input_dim", "answer", "answer_dim"]
    STATE_KEYS = ["grid", "grid_dim", "selected", "clip", "clip_dim",
                  "terminated", "trials_remain", "active",
                  "object", "object_sel", "object_dim", "object_pos", 
                  "background", "rotation_parity"]
    ACTION_KEYS = ["operation", "bbox"]

    def __init__(self, env, cfg, device, act_fn):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.att_set = set(self.GRIDS + self.TUPLES + self.NUMBERS)
        self.act_fn = act_fn
        self.rew_rms = RunningMeanStd(shape=(), clip=cfg.train.cliprew)
        self.ret_rms = RunningMeanStd(shape=(), clip=cfg.train.cliprew)
        self.reset()

    def reset(self):
        for att in self.att_set:
            setattr(self, att, [])

        state_infos = {key: [] for key in self.STATE_KEYS + self.INFO_KEYS}
        for _ in range(self.cfg.train.nenvs):
            raw_state, info = self.env.reset()
            state_info = flatten_and_copy(raw_state) | info
            for key in self.STATE_KEYS + self.INFO_KEYS:
                state_infos[key].append(state_info[key])

        for key in self.STATE_KEYS + self.INFO_KEYS:
            getattr(self, key).append(np.stack(state_infos[key]))
        self.timesteps = np.zeros(self.cfg.train.nenvs)
        self.sum_rewards = np.zeros(self.cfg.train.nenvs)
        self.success = np.zeros(self.cfg.train.nenvs)
        self.disc_sum_rewards = np.zeros(self.cfg.train.nenvs)

        rewards, _ = self._augmented_reward()
        self.rtm1.append(rewards)

    def _get_selection_from_bbox(self, bboxes):
        selection = np.zeros((len(bboxes), self.cfg.env.grid_x, self.cfg.env.grid_y)).astype(np.uint8)
        # mult = np.array([self.cfg.env.grid_x, self.cfg.env.grid_y, self.cfg.env.grid_x, self.cfg.env.grid_y])
        # bias = np.array([0, 0, - self.cfg.env.grid_x // 2, - self.cfg.env.grid_y // 2])

        mult = np.concatenate((self.answer_dim[-1], self.answer_dim[-1]), axis=1)
        zero_array = np.zeros(self.answer_dim[-1].shape, dtype=int)
        bias = np.concatenate ((zero_array, - self.answer_dim[-1] // 2), axis=1)
        
        bboxes = bboxes * mult + bias
        bboxes[:, (2, 3)] = bboxes[:, (2, 3)] + bboxes[:, (0, 1)]
        bboxes[:, (0, 2)] = np.clip(bboxes[:, (0, 2)], 0, self.answer_dim[-1][:, (0,0)] - 1e-3)
        bboxes[:, (1, 3)] = np.clip(bboxes[:, (1, 3)], 0, self.answer_dim[-1][:, (1,1)] - 1e-3)
        bboxes = np.floor(bboxes).astype(int)
        for i, bbox in enumerate(bboxes):
            min_x, min_y = min(bbox[0], bbox[2]), min(bbox[1], bbox[3])
            max_x, max_y = max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
            selection[i, min_x:max_x + 1, min_y:max_y + 1] = 1
        return selection, bboxes

    def _augmented_reward(self):
        rewards = []
        success = []
        for g, ad, a in zip(self.grid[-1], self.answer_dim[-1], self.answer[-1]):
            dist = np.mean(g[:ad[0].item(), :ad[1].item()] != a[:ad[0], :ad[1]])
            rewards.append((dist == 0) - dist)
            success.append((dist == 0))
        return np.array(rewards), np.array(success)

    def run(self):
        lten = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        ften = lambda x: torch.tensor(x, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            for step in trange(self.cfg.train.nsteps):
                # Given observations, get action value and neglopacs
                # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
                policy_inp = {key: lten(getattr(self, key)[-1]) for key in self.STATE_KEYS + self.INFO_KEYS}
                operation, bbox, neglogpac, vpred, rtm1_pred, rpred, gpred = self.act_fn(**policy_inp)
                self.rtm1_pred.append(fnpy(rtm1_pred))
                self.rpred.append(fnpy(rpred))
                self.vpred.append(fnpy(vpred))
                self.gpred.append(fnpy(gpred))
                self.neglogpac.append(fnpy(neglogpac))
                self.bbox.append(fnpy(bbox))
                self.operation.append(npy(operation))
                selection, ebbox = self._get_selection_from_bbox(fnpy(bbox))
                self.ebbox.append(ebbox)

                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                """
                (grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain, active,
                 object, object_sel, object_dim, object_pos, background, rotation_parity, env_reward) = batch_act(
                    self.input[-1].astype(np.uint8), self.input_dim[-1], self.answer[-1].astype(np.uint8), self.answer_dim[-1],
                    self.grid[-1].astype(np.uint8), self.grid_dim[-1], self.selected[-1].astype(bool), self.clip[-1].astype(np.uint8), 
                    self.clip_dim[-1], self.terminated[-1], self.trials_remain[-1], 
                    self.active[-1], self.object[-1].astype(np.uint8), self.object_sel[-1].astype(np.uint8), self.object_dim[-1], 
                    self.object_pos[-1], self.background[-1].astype(np.uint8), self.rotation_parity[-1], 
                    selection.astype(bool), operation
                )"""

                (grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain, active,
                 object, object_sel, object_dim, object_pos, background, rotation_parity, env_reward) = batch_act_env(self.env,
                    self.input[-1].astype(np.uint8), self.input_dim[-1], self.answer[-1].astype(np.uint8), self.answer_dim[-1],
                    self.grid[-1].astype(np.uint8), self.grid_dim[-1], self.selected[-1].astype(bool), self.clip[-1].astype(np.uint8), 
                    self.clip_dim[-1], self.terminated[-1], self.trials_remain[-1], 
                    self.active[-1], self.object[-1].astype(np.uint8), self.object_sel[-1].astype(np.uint8), self.object_dim[-1], 
                    self.object_pos[-1], self.background[-1].astype(np.uint8), self.rotation_parity[-1], 
                    selection.astype(bool), operation
                )

                # append states
                self.grid.append(grid); self.grid_dim.append(grid_dim); self.selected.append(selected)
                self.clip.append(clip); self.clip_dim.append(clip_dim); self.terminated.append(terminated)
                self.trials_remain.append(trials_remain); self.active.append(active); self.object.append(object)
                self.object_sel.append(object_sel); self.object_dim.append(object_dim); self.object_pos.append(object_pos)
                self.background.append(background); self.rotation_parity.append(rotation_parity)
                # append infos
                self.input.append(deepcopy(self.input[-1])); self.input_dim.append(deepcopy(self.input_dim[-1]))
                self.answer.append(deepcopy(self.answer[-1])); self.answer_dim.append(deepcopy(self.answer_dim[-1]))

                rewards, success = self._augmented_reward()

                self.timesteps += 1
                self.sum_rewards += rewards
                self.success += success
                self.disc_sum_rewards = self.cfg.train.gamma * self.disc_sum_rewards + rewards
                self.ret_rms.update(self.disc_sum_rewards)
                
                # normalize reward
                self.reward.append(rewards)
                self.rtm1.append(rewards)

        policy_inp = {key: lten(getattr(self, key)[-1]) for key in self.STATE_KEYS + self.INFO_KEYS}
        _, _, _, nextvalues, _, _, _ = self.act_fn(**policy_inp)
        nextvalues = fnpy(nextvalues)

        gtp1 = np.stack(self.grid[1:])
        clip = np.stack(self.grid[1:])
        # render_ansi_with_grids(self.env, gtp1, clip)
        for att in self.STATE_KEYS + self.INFO_KEYS + self.ACTION_KEYS + [
            "vpred", "neglogpac", "reward", "rpred", "rtm1", "rtm1_pred", "ebbox", "gpred"]:
            if att in self.STATE_KEYS + self.INFO_KEYS + ["rtm1"]:
                setattr(self, att, np.stack(getattr(self, att)[:-1]))
            else:
                setattr(self, att, np.stack(getattr(self, att)))

        # normalize reward
        self.rew_rms.update(self.rtm1.reshape(-1))
        self.rtm1 = self.rew_rms.normalize(self.rtm1, use_mean=True)
        norm_rew =  self.rew_rms.normalize(self.reward, use_mean=True)
        self.reward = self.ret_rms.normalize(self.reward, use_mean=True)

        # discount/bootstrap off value fn
        self.returns = np.zeros_like(self.reward)
        advs = np.zeros_like(self.reward)
        lastgaelam = 0
        for t in reversed(range(self.cfg.train.nsteps)):
            if t != self.cfg.train.nsteps - 1:
                nextvalues = self.vpred[t + 1]

            delta = self.reward[t] + self.cfg.train.gamma * nextvalues - self.vpred[t]
            advs[t] = lastgaelam = delta + self.cfg.train.gamma * self.cfg.train.lam * lastgaelam
        self.returns = advs + self.vpred

        ret_ob_acs = {key: lten(sf01(getattr(self, key))) for key in self.STATE_KEYS + self.INFO_KEYS + ["operation"]}
        ret_ob_acs = ret_ob_acs | {"bbox": ften(sf01(self.bbox))}
        ret = (
            ret_ob_acs, 
            ften(sf01(self.returns)), 
            ften(sf01(self.vpred)), 
            ften(sf01(self.neglogpac)), 
            ften(sf01(self.rtm1)), 
            ften(sf01(self.rtm1_pred)), 
            ften(sf01(norm_rew)),
            ften(sf01(self.rpred)),
            ften(sf01(self.gpred, no_prod=True)),
            lten(sf01(gtp1)),
            list(self.sum_rewards),
            list(self.success),
            self.ebbox,
            gtp1,
            clip,
            self.input[-1],
            self.answer[-1])
        self.reset()

        return ret

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr, no_prod=False):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    if not no_prod:
        shape = (s[0] * s[1], int(np.prod(s[2:]))) if int(np.prod(s[2:])) != 1 else (s[0] * s[1],)
        return arr.swapaxes(0, 1).reshape(*shape)
    else:
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def flatten_and_copy(state):
    new_state = deepcopy(state)
    object_state = new_state.pop("object_states")
    return new_state | object_state

def fnpy(tensor):
    return tensor.detach().cpu().numpy().astype(float)

def npy(tensor):
    return tensor.detach().cpu().numpy().astype(int)

def unpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.uint8)
