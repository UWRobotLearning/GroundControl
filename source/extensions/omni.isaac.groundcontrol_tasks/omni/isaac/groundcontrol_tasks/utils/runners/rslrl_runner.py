import warnings
import rsl_rl
from rsl_rl import runners
import os

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

import time
import torch
from collections import deque
from typing import Dict

from .common.rl_runner import RLRunner


class ExtraRslRlRolloutStorage(rsl_rl.storage.rollout_storage.RolloutStorage):
    def __init__(self, extras_shape: Dict[str, torch.Size],
                 *args, **kwargs):
        '''
        example of extras_info:
        {
            "reward": (torch.Size([64, 2]), lambda infos: infos["observations"]["reward"]),
            ...
        }
        The name is the name of the extras tensor and the size is the size of this tensor.
        The callable is the mapping from the infos dict to the tensor.
        '''
        super().__init__(*args, **kwargs)
        self.extras = {name: torch.empty(self.num_transitions_per_env, self.num_envs, *shape) for name, shape in extras_shape.items()}

    def add_transitions(self, transition, infos):
        super().add_transitions(transition)
        self.step -= 1 # Undo super step
        for name in self.extras.keys():
            self.extras[name][self.step] = infos['observations'][name]
        self.step += 1 # Redo super step


class OnPolicyRunner(runners.OnPolicyRunner, RLRunner):
    ''' Override for logging purposes '''

    def __init__(self, env, cfg, log_dir=None, device="cpu"):
        super().__init__(env, cfg, log_dir, device)
        self.no_log = self.cfg.get("rl_no_log", True)
        self.no_wandb = self.cfg.get("no_wandb", False)
        self.logger_type = None
        if not self.no_wandb:
            self.logger_type = "wandb"
        # self.pbar = tqdm(total=self.cfg.get("rl_max_iterations", 0))

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if not self.no_log and self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in tqdm(range(start_iter, tot_iter)):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    if actions.isnan().any():
                        raise ValueError("NaN in actions")
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if not self.no_log:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if not self.no_log:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", f"model_{it}.pt"))
            ep_infos.clear()

        # self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def collect_rollouts(self, max_steps: int) -> ExtraRslRlRolloutStorage:
        alg = self.alg

        rollouts_storage = ExtraRslRlRolloutStorage(
            extras_shape={
                "reward": torch.Size([self.env.num_envs, 1]),
            },
            num_envs=self.storage.num_envs,
            num_transitions_per_env=self.storage.num_transitions_per_env,
            actor_obs_shape=self.storage.actor_obs_shape,
            critic_obs_shape=self.storage.critic_obs_shape,
            action_shape=self.storage.action_shape,
            device=self.device,
        )
        self.storage.clear()
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        with torch.inference_mode():
            self.env.reset()
            for i in range(max_steps):
                actions = alg.act(obs, critic_obs)
                obs, rewards, dones, infos = self.env.step(actions)
                if actions.isnan().any():
                    raise ValueError("NaN in actions")
                obs = self.obs_normalizer(obs)
                if "critic" in infos["observations"]:
                    critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                else:
                    critic_obs = obs
                obs, critic_obs, rewards, dones = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )

                ## on_policy_runner.OnPolicyRunner.process_env_step:
                alg.transition.rewards = rewards.clone()
                alg.transition.dones = dones
                        # Bootstrapping on time outs
                if "time_outs" in infos:
                    alg.transition.rewards += alg.gamma * torch.squeeze(
                        alg.transition.values * infos["time_outs"].unsqueeze(1).to(alg.device), 1
                    )

                # Record the transitions to both buffers
                # alg.storage.add_transitions(alg.transition)
                rollouts_storage.add_transitions(alg.transition, infos)

                # Reset
                alg.transition.clear()
                alg.actor_critic.reset(dones)

        return rollouts_storage

    def reset(self):
        self.__init__(self.env, self.cfg, self.log_dir, self.device)
