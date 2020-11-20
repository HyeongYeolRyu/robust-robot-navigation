# CS686 Term Project 
## Robust robot navigation against imperfect sensor data

### Installation
```
$ cd YOUR_WS
$ mkdir src && cd src
$ git clone https://github.com/HyeongYeolRyu/robust-robot-navigation.git
$ cd ..
$ catkin_make
```
Please refer to the following repo & install, too.

https://github.com/srl-freiburg/pedsim_ros


Add following code in ~/.bashrc
```
$ export TURTLEBOT3_MODEL=burger
```

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

