#!/usr/bin/env python3
import rospy
import gym
import gym_gazebo
import numpy as np
import os
import datetime
import time
from ddpg import *
from environment import Env
from torch.utils.tensorboard import SummaryWriter


log_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'logs')  # tensorboard log
exploration_decay_start_step = 50000
state_dim = 6 + 720  # current state + Lidar dim.
action_dim = 2
action_linear_max = 1.25  # m/s
action_linear_min = 0.0  # m/s
action_angular_max = 0.5  # rad/s
action_angular_min = -0.5  # rad/s
is_training = True
max_episode_length = 500
# for debugging
is_debugging = False
cur_robot_state = None


def main():
    rospy.init_node('ddpg_stage_1')
    env = Env(is_training)
    agent = DDPG(env, state_dim, action_dim)
    
    import ipdb
    ipdb.set_trace()

    past_action = np.array([0., 0.])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')
    print('Action Min: ' + str(action_linear_min) + ' m/s and ' + str(action_angular_min) + ' rad/s')

#########################################################################################
#                                 Training
#########################################################################################
    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        action_var = 0.2
        success_rate = 0

        # Log path setting
        now = datetime.datetime.now()
        logdir = now.strftime('%Y-%M-%d') + '_' + now.strftime('%H-%M')
        logdir = os.path.join(log_dir, logdir)
        # tb_writer = SummaryWriter(logdir)

        # Start training
        start_time = time.time()
        for itr in range(10000):
            state = env.reset()

            # episode_reward = 0.0
            # For each episode
            for cur_step in range(max_episode_length):
                action = agent.action(state)
                action[0] = np.clip(np.random.normal(action[0], action_var), action_linear_min, action_linear_max)
                action[1] = np.clip(np.random.normal(action[1], action_var), action_angular_min, action_angular_max)

                state_, reward, done, arrive = env.step(action, past_action)
                time_step = agent.perceive(state, action, reward, state_, done)

                ########################################################################################
                #                                   debugging environment
                ########################################################################################
                if is_debugging:
                    print('cur_step: {}'.format(cur_step))
                    print('action: {}'.format(action))
                    print('goal position: x:{}, y:{}'.format(env.goal_position.position.x, env.goal_position.position.y))
                    print('r: {}, done: {}, arrive: {}'.format(reward, done, arrive))
                ########################################################################################

                result = 'Success' if arrive else 'Fail'

                if time_step > 0:
                    total_reward += reward

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward: {}'.format(avg_reward))
                    avg_reward_his.append(round(avg_reward, 2))
                    # writer.add_scalar('avg_reward', avg_reward, time_step)
                    print('Overall average Reward: {}'.format(avg_reward_his))
                    total_reward = 0

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    action_var *= 0.9999

                past_action = action
                state = state_

                if arrive or done or cur_step >= max_episode_length:
                    if result == 'Success':
                        success_rate += 1
                    sec = time.time() - start_time
                    elapsed_time = str(datetime.timedelta(seconds=sec)).split('.')[0]
                    print('Num_episode: {}, Full steps: {}, Result: {}, Elapsed time: {}'.format(itr, cur_step, result, elapsed_time))

                    if itr % 20 == 0 and itr > 0:
                        print('Total: {}/20, Success rate: {}'.format(success_rate, round(success_rate/20), 2))
                        success_rate = 0

                    break


#########################################################################################
#                                 Testing
#########################################################################################
    else:
        print('Testing mode')
        while True:
            state = env.reset()
            one_round_step = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)
                state_, reward, done, arrive = env.step(a, past_action)
                past_action = a
                state = state_
                one_round_step += 1

                if arrive:
                    print('Step: %3i' % one_round_step, '| Arrive!!!')
                    one_round_step = 0

                if done:
                    print('Step: %3i' % one_round_step, '| Collision!!!')
                    break


if __name__ == '__main__':
    main()