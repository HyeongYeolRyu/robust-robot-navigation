#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from gym import core, spaces
from geometry_msgs.msg import Twist, Pose, PointStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from spinup.envs.damaged_sensor import *


class env(core.Env):
    """
    Class for Gazebo Environment

    Main function:
        1. step(): Execute new action and return state
        2. reset(): Rest environment at the end of each episode
        and generate new goal position for next episode
        3. _process_scan_obs(): Receive and process raw scan observation
        4. _compute_reward(): Set a reward value given a state
    """

    def __init__(self, is_training):
        print("[ENV] NAVEnv has been being loaded!")
        # Sensor specification
        self.sensor_range = 8
        self.sensor_timeout = 0.5
        self.sensor_dim = 480
        self.stack_size = 30
        self.stacked_scan_obs = [self.sensor_range for _ in range(self.stack_size * self.sensor_dim)]
        self.use_stacked_scan_obs = False
        self.damage_sensor = False

        # Sensor visualization
        self.visualize_scan_obs = False
        self.visualize_stacked_scan_obs = False
        self.vis_window = None
        self.visualize_y_axis_size = 100
        self.plt_pause_time = 0.0000000001

        # Robot state
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_point_stamped = PointStamped()
        self.robot_scan = np.zeros(self.sensor_dim)
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_position = PoseStamped()
        self.past_action = np.array([0., 0.])
        self.pitch = 0.
        self.past_yaw_list = [0 for _ in range(self.stack_size)]
        self.yaw_ptr = 0.
        self.past_reward = 0
        self.past_info = {'is_arrive': False}
        self.init_position = [17, 11]
        self.state_dim = 6

        # Input state
        self.past_distance = 0.
        self.goal_distance = 0.

        # Goal position
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.

        # Goal arrival threshold
        self.threshold_arrive = 0.6
        self.diagonal_dis = 15

        # Service
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.pause_pedsim = rospy.ServiceProxy('/pedsim_simulator/pause_simulation', Empty)
        self.unpause_pedsim = rospy.ServiceProxy('/pedsim_simulator/unpause_simulation', Empty)

        # Publisher
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.pub_goal = rospy.Publisher('goal_position', PointStamped, queue_size=1)
        self.pub_robot_position = rospy.Publisher('robot_position', PoseStamped, queue_size=1)

        # Subscriber
        self.sub_odom = rospy.Subscriber('odom', Odometry, self._get_odometry)
        # self.sub_odom = rospy.Subscriber('/ground_truth_pose', Odometry, self._get_odometry)
        # self.sub_robot_position = rospy.Subscriber('/ground_truth_pose', Odometry, self._get_gt_position_cb)
        # self.sub_model_state = rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        # self.sub_scan = rospy.Subscriber('scan', LaserScan, self._robot_scan_cb)

        if self.visualize_scan_obs or self.visualize_stacked_scan_obs:
            dummy = np.zeros((self.visualize_y_axis_size, self.sensor_dim))
            plt.title('Scan observation')
            self.vis_window = plt.imshow(dummy, cmap='gray', vmin=0, vmax=self.sensor_range)

        self._generate_goal()

        # state = self.reset()
        self.pub_cmd_vel.publish(Twist())
        self.act_limit_max = np.array([0.25, 0.5])
        self.act_limit_min = np.array([0.0, -0.5])
        self.action_space = spaces.Box(self.act_limit_min, self.act_limit_max, dtype='float32')

        if self.use_stacked_scan_obs:
            self.past_state = self.stacked_scan_obs + [0, 0, 0, 0]
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.stack_size * self.sensor_dim + self.state_dim, ), dtype='float32')
        else:
            self.past_state = np.concatenate((np.full(self.sensor_dim, self.sensor_range),
                                          np.array([0, 0, 0, 0]))
                                        )
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.sensor_dim + self.state_dim, ), dtype='float32')

        # Initialize damage sensor
        if self.damage_sensor:
            self.damage_module = ConstDamageModule(p=0.5, occ_range=(0,240))
            # self.damage_module = SplitDamageModule(p=0.9, splits_num=12, occ_num=6)
            self.damage_module.reset()

    def _get_goal_distance(self):
        goal_distance = 0.
        while True:
            try:
                goal_distance = math.sqrt((self.goal_position.position.x - self.position.x)**2 + 
                                          (self.goal_position.position.y - self.position.y)**2)
                self.past_distance = goal_distance
                break
            except Exception as e:
                print("Calculating goal distance doesn't work well!")
                print(e)
                continue

        return goal_distance

    def _get_odometry(self, odom):
        self.position = odom.pose.pose.position

        # For publishing current position of the robot
        self.robot_position.pose.position = odom.pose.pose.position
        self.robot_position.pose.orientation = odom.pose.pose.orientation
        self.robot_position.header.frame_id = 'odom'

        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))
        # self.pitch = math.degrees(math.atan2((2 * q_x * q_w) - (2 * q_y * q_z), 1 - (2 * q_x*q_x + 2 * q_z*q_z)))

        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_angle = round(math.degrees(theta), 2)

        diff_angle = abs(rel_angle - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_angle = rel_angle
        self.yaw = yaw
        self.diff_angle = diff_angle

    def _process_scan_obs(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_angle = self.rel_angle
        diff_angle = self.diff_angle
        min_range = 0.3
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(self.sensor_range)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            print("==============================================================================================")
            print("                                                                                  Collision!!!")
            print("==============================================================================================")
            done = True

        # damage
        if self.damage_sensor:
            scan_range = self.damage_module(np.array(scan_range))

        rel_dist = math.sqrt((self.goal_position.position.x - self.position.x)**2 +
                              (self.goal_position.position.y - self.position.y)**2)

        if rel_dist <= self.threshold_arrive:
            print("==============================================================================================")
            print("                                                                         Arrive at the goal!!!")
            print("==============================================================================================")
            done = True
            arrive = True

        if abs(self.pitch) > math.degrees(180):
            print("==============================================================================================")
            print("Fall down!")
            print("Pitch value: {}".format(self.pitch))
            print("==============================================================================================")
            done = True
            arrive = False

        return scan_range, rel_dist, yaw, rel_angle, diff_angle, done, arrive

    def _compute_reward(self, done, arrive):
        r_base = -0.05  # Small negative base reward that makes robot move
        r_goal = 0.
        r_done = 0.
        r_rotation = 0.

        current_distance = math.sqrt((self.goal_position.position.x - self.position.x)**2 + 
                                     (self.goal_position.position.y - self.position.y)**2)

        distance_rate = (self.past_distance - current_distance)
        r_distance = 300.*distance_rate

        self.past_distance = current_distance

        r_heading = -self.diff_angle/360

        if abs(self.past_action[1]) >= 0.4:
            r_rotation = -0.3 * abs(self.past_action[1])

        if arrive:
            r_goal = 120.
            self.pub_cmd_vel.publish(Twist())
            # return reward
        elif done:
            r_done = -350.
            self.pub_cmd_vel.publish(Twist())
            # return r_done

        # Total reward
        print(f"          ------------------>  base  / dist  / heading / rot   / goal  / done")
        print(f"          ------------------> {r_base:.3f} / {r_distance:.3f} / {r_heading:.3f} / {r_rotation:.3f} / {r_goal:.3f} / {r_done:.3f}")

        reward = r_base + r_distance + r_goal + r_done + r_heading + r_rotation

        return reward

    def step(self, action):
        self._unpause_gazebo()

        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        self.pub_robot_position.publish(self.robot_position)  # Publish current position of the robot
        self.pub_goal.publish(self.goal_point_stamped)  # Publish Goal position.

        # This should be included for processing the delay
        rospy.sleep(0.001)

        try:
            data = rospy.wait_for_message('scan', LaserScan, timeout=self.sensor_timeout)
            scan, rel_dist, yaw, rel_angle, diff_angle, done, arrive = self._process_scan_obs(data)
            state = [i for i in scan]

            if self.use_stacked_scan_obs:
                state = self._stack_scan_obs(state)
                # self._calibrate_sensor()

            if self.visualize_scan_obs or self.visualize_stacked_scan_obs:
                self._visualize_sensor(scan)

            state = state + [self.past_action[0], self.past_action[1],
                             rel_dist / self.diagonal_dis,
                             rel_angle / 360,
                             yaw / 360,
                             diff_angle / 180]
            reward = self._compute_reward(done, arrive)

            info = {
                'is_arrive': arrive
            }
            self.past_action = action
            self.past_state = state
            self.past_reward = reward
            self.past_info = info
            self.past_yaw = yaw
        except Exception as e:
            print(f"[Env] Error in step function: {e} --> use stored past value instead.")
            state = self.past_state
            reward = self.past_reward
            info = self.past_info
            done = True

        self._pause_gazebo()

        return np.asarray(state), reward, done, info

    def reset(self):
        """
        Reset the environment

        1. Unpause gazebo
        2. Reset gazebo
          2-1. Delete target
          2-2. Reset simulation
        3. Generate target
        4. Get state
        5. Pause gazebo
        6. Return state
        """

        print("==============================================================================================")
        print("                                                                            Reset environment!")
        print("==============================================================================================")

        # 0. Initialize stacked_scan_obs
        self.stacked_scan_obs = [self.sensor_range for _ in range(self.stack_size * self.sensor_dim)]

        # 1. Unpause gazebo
        self._unpause_gazebo()
        # self.unpause_pedsim()
        # 2-1. Delete target
        rospy.wait_for_service('/gazebo/delete_model')
        # try:
        #     self.del_model('target')
        # except rospy.ServiceException as e:
        #     print("gazebo/delete_model service call failed: {}".format(e))

        # 2-2. Reset simulation
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed: {}".format(e))

        while True:
            try:
                if abs(self.init_position[0] - self.position.x) < 0.1 and\
                   abs(self.init_position[1] - self.position.y) < 0.1:
                    break
            except:
                pass
            # else:
                # print("Resetting simulation...")

        # 3. Generate target
        self._generate_goal()

        # Publish goal position
        self.pub_goal.publish(self.goal_point_stamped)

        # 4. Get state
        try:
            data = rospy.wait_for_message('scan', LaserScan, timeout=self.sensor_timeout)
            self.goal_distance = self._get_goal_distance()

            scan, rel_dist, yaw, rel_angle, diff_angle, done, arrive = self._process_scan_obs(data)
            state = [i for i in scan]  # Normalize scan data

            if self.use_stacked_scan_obs:
                state = self._stack_scan_obs(state)
                # self._calibrate_sensor()

            if self.visualize_scan_obs or self.visualize_stacked_scan_obs:
                self._visualize_sensor(scan)

            # Past action: Initially, linear & angular velocity: 0, 0
            state = state + [0, 0,
                             rel_dist / self.diagonal_dis,  # Relative distance to goal
                             rel_angle / 360,  # Relative angle to goal
                             yaw / 360,  # Current yaw of robot
                             diff_angle / 180]

            reward = self._compute_reward(done, arrive)
            info = {
                'is_arrive': arrive
            }
            self.past_state = state
            self.past_reward = reward
            self.past_info = info
            self.past_yaw = yaw
        except Exception as e:
            print(f"[Env] Error in step function: {e} --> use stored past value instead.")
            state = self.past_state
            reward = self.past_reward
            info = self.past_info
            done = True

        # 5. Pause gazebo
        self._pause_gazebo()
        # self.pause_pedsim()

        # Initialize damage sensor
        if self.damage_sensor:
            self.damage_module.reset()

        # 6. Return state
        return np.asarray(state)

    def _pause_gazebo(self):
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException as e:
            print("pause Service Failed: {}".format(e))

    def _unpause_gazebo(self):
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: {}".format(e))

    def _generate_goal(self):
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            # goal_urdf = open(goal_model_dir, "r").read()
            # target = SpawnModel
            # target.model_name = 'target'  # the same with sdf name
            # target.model_xml = goal_urdf
            # offset_from_wall = 5
            self.goal_position.position.x = random.uniform(12, 13)
            self.goal_position.position.y = random.uniform(15, 16)
            # self.goal_position.position.x = 20 - 2.5
            # self.goal_position.position.y = 15 - 2.5

            # For publishing goal position
            self.goal_point_stamped.point.x = self.goal_position.position.x
            self.goal_point_stamped.point.y = self.goal_position.position.y
            self.goal_point_stamped.header.frame_id = 'odom'

            # self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except rospy.ServiceException as e:
            print("/gazebo/failed to generate the target: {}".format(e))

    def _stack_scan_obs(self, obs):
        self.stacked_scan_obs[self.sensor_dim:] = self.stacked_scan_obs[:-self.sensor_dim]
        self.stacked_scan_obs[:self.sensor_dim] = obs
        return self.stacked_scan_obs

    def _visualize_sensor(self, scan):
        if self.visualize_scan_obs:
            scan_obs = np.expand_dims(np.array(scan), axis=0)
            scan_obs = np.repeat(scan_obs, self.visualize_y_axis_size, axis=0)
            scan_obs = np.fliplr(scan_obs)
            self.vis_window.set_data(scan_obs)
            plt.pause(self.plt_pause_time)
            plt.draw()

        if self.visualize_stacked_scan_obs:
            stacked_scan_obs_2d = np.array(self.stacked_scan_obs[::-1]).reshape(self.stack_size, self.sensor_dim)
            # Multiplying 'self.sensor_range' makes back to the original sensor data (before it, we normalized)
            stacked_scan_obs_2d = stacked_scan_obs_2d  # * self.sensor_range
            self.vis_window.set_data(stacked_scan_obs_2d)
            plt.pause(self.plt_pause_time)
            plt.draw()

    def _calibrate_sensor(self):
        shift_weight = (self.yaw - self.past_yaw) / 60
        shift_cnt = int(self.sensor_dim * shift_weight)
        for i in range(self.stack_size):
            if shift_cnt < 0:
                for _ in range(shift_cnt):
                    self.stacked_scan_obs[i*self.sensor_dim: (i+1)*self.sensor_dim].pop(-1)
                    self.stacked_scan_obs[i*self.sensor_dim: (i+1)*self.sensor_dim].insert(0, 0)
            elif shift_cnt > 0:
                for _ in range(shift_cnt):
                    self.stacked_scan_obs[i*self.sensor_dim: (i+1)*self.sensor_dim].pop(0)
                    self.stacked_scan_obs[i*self.sensor_dim: (i+1)*self.sensor_dim].insert(-1, 0)
            else:
                return

    def _store_yaw(self, current_yaw):
        self.past_yaw_list[self.yaw_ptr] = current_yaw
        self.yaw_ptr += 1
        if self.yaw_ptr == self.stack_size:
            self.yaw_ptr = 0

    ##########################################################################################################
    #                                           UNUSED
    ##########################################################################################################
    def _get_gt_position_cb(self, odom):
        self.robot_gt_position = odom.pose.pose.position
        x, y, z = self.robot_gt_position.x, self.robot_gt_position.y, self.robot_gt_position.z

        print('x:{}, y:{}, z:{}'.format(x, y, z))

    def _robot_state_cb(self, msg):
        """
        Callback function for robot state
        :param msg: message
        """

        if self.robot_state_init is False:
            self.robot_state_init = True

        quat = [msg.pose[-1].orientation.x,
                msg.pose[-1].orientation.y,
                msg.pose[-1].orientation.z,
                msg.pose[-1].orientation.w]

        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        linear_spd = math.sqrt(msg.twist[-1].linear.x**2 + msg.twist[-1].linear.y**2)
        self.robot_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]
        self.robot_speed = [linear_spd, msg.twist[-1].angular.z]

    def _robot_scan_cb(self, msg):
        """
        Callback function for robot scan
        :param msg: message
        """

        if self.robot_scan_init is False:
            self.robot_scan_init = True
        self.robot_scan = msg