from itertools import repeat, product
from typing import Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Landmark

import re
#文件位置C:\Users\Administrator\AppData\Local\Programs\Python\Python36\Lib\site-packages\highway_env\envs\lvxinfei_env

class lvxinfeiv2(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-5, 5], [-7, 7]],
                "grid_step": [2, 2],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "collision_reward": -100,
            "lane_centering_cost": 1,
            "action_reward": 0.05,
            "arrival_reward": 50,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
        })
        return config


    def str_to_number(self,str1):#将landmark的坐标提取出来
        a = []
        a = re.findall("\d+\.?\d*", str1)  # 正则表达式
        a = list(map(float,a))
        b = [a[1],a[2]]
        return b

    def is_success(self):
    #判断车辆是否到达标记点
        resh1 = self.str_to_number(str(self.goal1))#目标位置的坐标
        x = resh1 - self.vehicle.position
        x = np.square(x)
        x = np.sum(x)
        x = np.sqrt(x)
        resh2 = self.str_to_number(str(self.goal2))#目标位置的坐标
        y = resh2 - self.vehicle.position
        y = np.square(y)
        y = np.sum(y)
        y = np.sqrt(y)
        if x<=3 or y<= 3:
          return True
        else :
          return False



    def _reward(self, action: np.ndarray) -> float:#奖励函数部分
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        speed_reward = 1/(1+self.config["lane_centering_cost"]*(self.vehicle.speed * np.sin(self.vehicle.heading))**2)
        reward = \
            + (self.config["arrival_reward"]) * self.is_success() \
            + lane_centering_reward

        if self.vehicle.crashed or not self.vehicle.on_road:
          reward = self.config["collision_reward"]
        # print(np.cos(self.vehicle.heading))
        return reward


    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        # success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        # print(self.vehicle.position)
        success = self.is_success()
        return self.vehicle.crashed or self.steps >= self.config["duration"] or not self.vehicle.on_road or success



    def _reset(self) -> None:
        self._create_road()
        self._make_vehicles()

    def _create_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [60, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([42, 5], [60, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [60, -70]
        radii1 = 70
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

       
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.randint(1)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20,50))#20,50
            controlled_vehicle.SPEED_MIN = 0
            controlled_vehicle.SPEED_MAX = 10
            controlled_vehicle.SPEED_COUNT = 3

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.randint(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)
        
        # lane = self.np_random.choice(self.road.network.lanes_list())
        lane = self.road.network.lanes_list()[2]
        lane2 = self.road.network.lanes_list()[3]
        self.goal1 = Landmark(self.road, lane.position(lane.length, 0))#, heading=lane.heading
        self.goal2 = Landmark(self.road, lane2.position(lane2.length, 0))#, heading=lane.heading
        # print(self.goal1)
        # print(self.goal2)
        self.road.objects.append(self.goal1)
        self.road.objects.append(self.goal2)


register(
    id='lvxinfei-v2',
    entry_point='highway_env.envs:lvxinfeiv2',
)
