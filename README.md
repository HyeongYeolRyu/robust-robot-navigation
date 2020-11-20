# CS686 Term Project 
## Robust robot navigation against imperfect sensor data

### Dependencies
- python3
- gym_gazebo: [link](https://github.com/erlerobot/gym-gazebo/blob/master/INSTALL.md#ubuntu-1804)
- tensorflow-1.14


- defusedxml
```
$ pip install defusedxml
```

### Installation
```
$ cd YOUR_WS
$ mkdir src && cd src
$ git clone https://github.com/HyeongYeolRyu/robust-robot-navigation.git
$ cd ..
$ catkin_make
```

Add target robot to ~/.bashrc
```
$ echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
```

Please refer to the following repo & install

https://github.com/srl-freiburg/pedsim_ros

Once install it, follow the usages of 'pedsim_gazebo_plugin' package




### How to run?

Step 1.
```
$ roslaunch pedsim_simulator simple_pedestrians.launch
```

Step 2.
```
$ roslaunch pedsim_gazebo_plugin <scenario_name>.launch
```

Step 3.
```
$ roslaunch project ddpg_stage_1.launch
```

