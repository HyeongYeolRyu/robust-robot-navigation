from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import datetime
import rospy
import spinup.algos.pytorch.ddpg.core_fc_lstm_v1 as core
from spinup.utils.logx import EpochLogger


use_lstm = True
seq_len = 50


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.ptr = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.episodes_list = [None for _ in range(100)]
        self.episodes_list_max_size = 100
        self.epi_ptr = -1
        self.max_ep_len = 1000
        self.episode_buffer_dict = {
            'obs': np.zeros(core.combined_shape(self.max_ep_len, obs_dim), dtype=np.float32),
            'obs2': np.zeros(core.combined_shape(self.max_ep_len, obs_dim), dtype=np.float32),
            'act': np.zeros(core.combined_shape(self.max_ep_len, act_dim), dtype=np.float32),
            'rew': np.zeros(self.max_ep_len, dtype=np.float32),
            'done': np.zeros(self.max_ep_len, dtype=np.float32),
        }

    def count_epi_ptr(self):
        # import ipdb
        # ipdb.set_trace()

        if self.epi_ptr != -1:
            if self.ptr > seq_len:
                self.episodes_list[self.epi_ptr] = {}
                for key in self.episode_buffer_dict.keys():
                    self.episodes_list[self.epi_ptr][key] = self.episode_buffer_dict[key][:self.ptr]
            else:
                self.epi_ptr -= 1

        self.epi_ptr += 1
        self.ptr = 0
        if self.epi_ptr == self.episodes_list_max_size:
            self.epi_ptr = 0

    def store(self, obs, act, rew, next_obs, done):
        self.episode_buffer_dict['obs'][self.ptr] = obs
        self.episode_buffer_dict['obs2'][self.ptr] = next_obs
        self.episode_buffer_dict['act'][self.ptr] = act
        self.episode_buffer_dict['rew'][self.ptr] = rew
        self.episode_buffer_dict['done'][self.ptr] = done
        self.ptr += 1

    def __repr__(self):
        string1 = f'epi_ptr : {self.epi_ptr}\n'
        string2 = ''
        # import ipdb
        # ipdb.set_trace()

        try:
            for key in self.episode_buffer_dict.keys():
                string2 += f'episodes[{key}] : {self.episodes_list[self.epi_ptr][key].shape}\n'
        except:
            pass

        return string1 + string2

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_seq_batch(self, batch_size=32):
        # import ipdb
        # ipdb.set_trace()

        out_dict = {
            'obs': np.zeros((batch_size, seq_len, self.obs_dim[0])),
            'obs2': np.zeros((batch_size, seq_len, self.obs_dim[0])),
            'act': np.zeros((batch_size, seq_len, self.act_dim)),
            'rew': np.zeros((batch_size, seq_len)),
            'done': np.zeros((batch_size, seq_len))
        }

        # sample_idx = np.random.randint(0,len(self.episodes_list),batch_size)
        # sampled_episodes = np.array(self.episodes_list)[sample_idx]

        sample_count = 0
        # sampled_episodes = np.random.choice(np.array(self.episodes_list), batch_size, replace=False)
        candidate = []
        for i in range(self.episodes_list_max_size):
            if isinstance(self.episodes_list[i], dict):
                candidate.append(i)

        while sample_count < batch_size:
            sampled_episode_idx = int(np.random.choice(candidate, 1))
            sampled_episode = self.episodes_list[sampled_episode_idx]
            # import ipdb
            # ipdb.set_trace()
            # print("======================")
            # print(type(sampled_episode))
            # print(sampled_episode)
            # print("======================")
            if not isinstance(sampled_episode, dict):
                continue

            max_len = sampled_episode['rew'].shape[0]
            first_idx = np.random.randint(0, max_len - seq_len, size=1)[0]
            last_idx = first_idx + seq_len
            for k in sampled_episode.keys():
                out_dict[k][sample_count] = sampled_episode[k][first_idx:last_idx]
            sample_count += 1

        result = []
        for seq_idx in range(seq_len):
            batch_dict = {
                'obs': torch.Tensor(out_dict['obs'][:, seq_idx]),
                'obs2': torch.Tensor(out_dict['obs2'][:, seq_idx]),
                'act': torch.Tensor(out_dict['act'][:, seq_idx]),
                'rew': torch.Tensor(out_dict['rew'][:, seq_idx]),
                'done': torch.Tensor(out_dict['done'][:, seq_idx]),
            }
            result.append(batch_dict)

        return result


def ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=1,
         steps_per_epoch=2000, epochs=10000, replay_size=int(1e5), gamma=0.99,
         polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=64, start_steps=1000,
         update_after=1000, update_every=500, act_noise=0.05, num_test_episodes=1,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    rospy.init_node('DDPG_Train')
    env = env_fn()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    print(f"[DDPG] obs dim: {obs_dim} action dim: {act_dim}")

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # ac.apply(init_weights)
    ac_targ = deepcopy(ac)
    ac.eval()  # in-active training BN
    print(f"[MODEL] Actor_Critic: {ac}")

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # import ipdb
        # ipdb.set_trace()
        q = ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        total_loss_q = 0.
        ac.reset_lstm()
        for i in range(seq_len):
            # loss_q, loss_info = compute_loss_q(data[i])
            loss_q, loss_info = compute_loss_q(data[i])
            total_loss_q += loss_q
        total_loss_q /= seq_len
        total_loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        total_loss_pi = 0.
        ac.reset_lstm()
        for i in range(seq_len):
            loss_pi = compute_loss_pi(data[i])
            total_loss_pi += loss_pi

        total_loss_pi /= seq_len
        total_loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

    def soft_target_update():
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        o = torch.as_tensor(o, dtype=torch.float32)
        if o.dim() == 1:
            o = o.unsqueeze(0)
        a = ac.act(o)[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, env.act_limit_min, env.act_limit_max)

    def test_agent():
        print("[DDPG] eval......")
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            if use_lstm:
                ac.reset_lstm()

            # while not(d or (ep_len == max_ep_len)):
            while not (d or (ep_len == 10)):
                # Take deterministic actions at test time (noise_scale=0)
                a = get_action(o, 0)
                print(f"[Eval] a: {a}")
                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    if use_lstm:
        ac.reset_lstm()
        replay_buffer.count_epi_ptr()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).

        print(f"O {o[-4]:.3f} {o[-3]:.3f} {o[-2]:.3f} {o[-1]:.3f} ")
        if t > start_steps:
            # if np.random.rand() > 0.3:
            a = get_action(o, act_noise)
            # else:
            # a = env.action_space.sample()
        else:
            a = env.action_space.sample()
        print(f't {t:7.0f} | a [{a[0]:.3f},{a[1]:.3f}]')

        # Step the env
        o2, r, d, info = env.step(a)
        # print(f"O {o[-4:]} |A {a} |O2 {o2[-4:]} |R {r} |D {d} |Info {info}")
        print(f"          ------------------> R: {r:.3f}")
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            env.pause_pedsim()
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            if use_lstm:
                ac.reset_lstm()
                replay_buffer.count_epi_ptr()
            env.unpause_pedsim()

        # Update handling
        if t >= update_after and t % update_every == 0:
            env.pause_pedsim()
            ac.train()  # active training BN
            ac_targ.train()
            if torch.cuda.is_available():
                ac.cuda()
                ac_targ.cuda()
            for k in range(update_every-400):
                if use_lstm:
                    batch = replay_buffer.sample_seq_batch(batch_size)
                else:
                    batch = replay_buffer.sample_batch(batch_size)
                if torch.cuda.is_available():
                    for i in range(seq_len):
                        for key, value in batch[i].items():
                            batch[i][key] = value.cuda()
                update(data=batch)
                soft_target_update()
            ac.eval()
            ac_targ.eval()
            if torch.cuda.is_available():
                ac.cpu()
                ac_targ.cpu()
            env.unpause_pedsim()

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            if use_lstm:
                ac.reset_lstm()
                replay_buffer.count_epi_ptr()

            # Transform human-readable time
            sec = time.time() - start_time
            elapsed_time = str(datetime.timedelta(seconds=sec)).split('.')[0]

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('Time', elapsed_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
