import time
import joblib
import os
import os.path as osp
import torch
from spinup import EpochLogger
from spinup.envs.NAVEnv import env as Env
import numpy as np
import rospy

def get_action(ac, o):
    o = torch.as_tensor(o, dtype=torch.float32)
    if o.dim() == 1:
        o = o.unsqueeze(0)
    a = ac.act(o)[0]
    return a

def run_policy(fpath, max_ep_len=None, num_episodes=100):

    # assert env is not None, \
    #     "Environment not found!\n\n It looks like the environment wasn't saved, " + \
    #     "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
    #     "page on Experiment Outputs for how to handle this situation."
    rospy.init_node('DDPG_Test')
    ac_model = torch.load(osp.join(fpath, 'pyt_save', 'model.pt'))
    env = Env(is_training=False)
    print(ac_model)

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        a = get_action(ac_model,o)
        print(f'a: {a}')
        o, r, d, _ = env.step(a)
        print(f"o {o} /r {r} /d {d}")
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    print("****************************************************************")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=1000)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    args = parser.parse_args()
    run_policy(args.fpath, args.len, args.episodes)