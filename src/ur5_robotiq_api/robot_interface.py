import sys
import rospy
import numpy as np
import copy
import os
import time
import json
from future.utils import viewitems
import cloudpickle


import tf2_ros
import tf
from visualization_msgs.msg import *
from  geometry_msgs.msg import *

from utils.utils import *
        
default_config = {
    "robot": 'jaco',
    "rate": 10,
    "env": "ros",
    "init_node": False,
    "damping": None,
    "natural_freq": None,
    'DriverUtils': {
        'type': None,
        'config':  {}
    },
    'MoveitUtils': {
        'type': None,
        'config':  {}
    }
}

class RobotCookingInterface(object):

    def __init__(self, config={}):
        self.RobotCookingInterface_config = default_config
        self.RobotCookingInterface_config.update(config)

        if self.RobotCookingInterface_config['init_node']:
            rospy.init_node("robot_cooking", anonymous=False)
            rospy.on_shutdown(self.cleanup)

        self.driver_utils = self.RobotCookingInterface_config['DriverUtils']['type'](self.RobotCookingInterface_config['DriverUtils']['config'])

        self.moveit_utils = None
        if self.RobotCookingInterface_config['MoveitUtils']['type'] is not None:
            self.moveit_utils = self.RobotCookingInterface_config['MoveitUtils']['type'](self.RobotCookingInterface_config['MoveitUtils']['config'])

        self.robot_name = self.RobotCookingInterface_config['robot']
        self.rate = rospy.Rate(self.RobotCookingInterface_config['rate'])


        if self.RobotCookingInterface_config['env'] == 'ros':
            #### Initialize tf ####
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_quat_distance = None

        #### load json config ####
        with open(os.path.join(os.environ['RC_PATH'], 'src', 'robot_cooking', 'env', 'config', self.robot_name+'_env_config.json')) as f:
            self.config_json = json.loads(f.read())

        #### all info ####
        self.all_info = None

        self.goal = None
        self.target = None
        self.obs_info = None

        time.sleep(2.5)

    def update_all_info(self):
        if self.target is None:
            target_pos, target_quat = self.driver_utils.get_ee_pose()
            self.target = np.concatenate([target_pos, target_quat])

        self.all_info = {
            'obs_info': self.obs_info,
            'target_pose': self.target,
            'goal_pose': self.goal
        }
        
    def update_tf(self, pt, tf_target, tf_src='world'):
        if tf_target == 'goal':
            self.goal = pt

        if tf_target == 'pose_target':
            self.target = pt
            
        self.update_all_info()
        
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = tf_src
        t.child_frame_id = tf_target

        t.transform.translation.x = pt[0]
        t.transform.translation.y = pt[1]
        t.transform.translation.z = pt[2]
        
        t.transform.rotation.x = pt[3]
        t.transform.rotation.y = pt[4]
        t.transform.rotation.z = pt[5]
        t.transform.rotation.w = pt[6]

        self.tf_broadcaster.sendTransform(t)
       
        
    def servo_to_pose_target(self, pt, pos_th=0.01, quat_th=0.1, dry_run=True):
        assert len(pt) == 7

        # update tf
        self.update_tf(pt,tf_target='pose_target', tf_src='world')
        
        # calculate the joint positions using ik
        ik_sol = self.moveit_utils.ik(position=pt[:-4], orientation=pt[-4:])
        ik_jp = ik_sol

        damping = self.RobotCookingInterface_config['damping']
        natural_freq = self.RobotCookingInterface_config['natural_freq']

        if damping is None or natural_freq is None:
            raise ValueError('need to specify damping and natural freq for servoing')
        
        kp = (2 * np.pi * natural_freq) ** 2
        kd = 2 * damping * 2 * np.pi * natural_freq

        joint_angle_diff = ik_jp - self.driver_utils.get_joint_values()
        joint_vel = self.driver_utils.get_joint_velocities()

        jv = kp * joint_angle_diff + kd * joint_vel
        
        ee_pos, ee_quat = self.driver_utils.get_ee_pose()

        ee_pose = np.concatenate([ee_pos, ee_quat])
        pos_distance, quat_distance = pose_distance(ee_pose, pt)
            
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')
            return True

        # start servoing
        if (pos_distance > pos_th or quat_distance > quat_th) and self.driver_utils.is_tool_in_safe_workspace():
            if not dry_run:
                self.driver_utils.pub_joint_velocity(jv, duration_sec=1./self.RobotCookingInterface_config['rate'])
            self.rate.sleep()
            return False
        else:
            return("servo reached goal")
            return True

                        
    def cleanup(self):
        pass

    ##############################
    # Common interface functions #
    ##############################
    def get_target_pose(self):
        pos, quat = self.driver_utils.get_ee_pose()
        return pos, quat

    def get_target_velocity(self):
        lv, av = self.driver_utils.get_ee_velocity()
        return lv, av

    def set_gripper_state(self, gs):
        self.driver_utils.set_finger_positions(gs * np.ones(3))

    def get_switch_state(self):
        '''
        positive is on, negative is off
        '''

        if self.RobotCookingInterface_config['env'] == 'ros':
            ## this is relative to grill
            switch_pose_tf_stamped = self.tf_buffer.lookup_transform('grill_mapped', 'switch_mapped', rospy.Time())
            switch_pose = np.array([
                switch_pose_tf_stamped.transform.translation.x,
                switch_pose_tf_stamped.transform.translation.y,
                switch_pose_tf_stamped.transform.translation.z,
                switch_pose_tf_stamped.transform.rotation.x,
                switch_pose_tf_stamped.transform.rotation.y,
                switch_pose_tf_stamped.transform.rotation.z,
                switch_pose_tf_stamped.transform.rotation.w,
            ])

            switch_angle_rel_grill = tf.euler_from_quaternion(switch_pose[3:])[1]
        else:
            switch_angle_rel_grill = self.driver_utils.get_switch_angle()
            
        # print(switch_angle_rel_grill)
        if switch_angle_rel_grill > 0.15:
            return 10.
        else:
            return -10.

    def move_to(self, pt, dry_run=True):
        ee_pos, ee_quat = self.driver_utils.get_ee_pose()
        ee_linear_vel, ee_angular_vel = self.driver_utils.get_ee_velocity()

        curr_pose = np.concatenate([ee_pos, ee_quat])
        curr_vel = np.concatenate([ee_linear_vel, ee_angular_vel])

        pos_distance, quat_distance = pose_distance(curr_pose, pt)
            
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')
            return True

        if (pos_distance > 0.005 or quat_distance > 0.1) and self.driver_utils.is_tool_in_safe_workspace():
            if not dry_run:
                self.servo_to_pose_target(pt, pos_th=0.005, quat_th=0.1, dry_run=dry_run)  
            else:
                self.update_tf(pt, tf_target='pose_target', tf_src='world')
                self.rate.sleep()
            return False
        else:
            # print("plan reached goal")
            return True


    def pub_ee_frame_velocity(self, direction, vel_scale, duration_sec):
        self.driver_utils.pub_ee_frame_velocity(direction, vel_scale, duration_sec)
            
    def set_goal_pose(self, pt):
        # self.wp_gen.set_goal(pt)
        self.update_tf(pt, tf_target='goal', tf_src='world')

        
    def get_obstacle_info(self):
        fitted_obstacles = self.config_json['fitted_elliptical_obstacles']['fitted_obstacles']
        obs_info = []
        for k, v in viewitems(fitted_obstacles):
            try:
                obs_pose_tf_stamped = self.tf_buffer.lookup_transform(v['parent_frame_id'][1:], v['child_frame_id'][1:], rospy.Time(), rospy.Duration(3.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                continue
           
            obs_pos = [
                obs_pose_tf_stamped.transform.translation.x,
                obs_pose_tf_stamped.transform.translation.y,
                obs_pose_tf_stamped.transform.translation.z
            ]
            obs_info.append({ 'name': k, 'position': obs_pos, 'scale':v['scale']})


        table_pose_tf_stamped  = self.tf_buffer.lookup_transform('world', 'table_mapped', rospy.Time())
        table_pos = [
            table_pose_tf_stamped.transform.translation.x,
            table_pose_tf_stamped.transform.translation.y,
            table_pose_tf_stamped.transform.translation.z
        ]
          
        table_info = {'name': 'table', 'position': table_pos}
        obs_info.append(table_info)

        self.obs_info = obs_info

        self.update_all_info()
        
        return self.obs_info

    def get_gripper_state(self):
        finger_positions =  self.driver_utils.get_finger_positions()
        if isinstance(finger_positions, np.ndarray) or isinstance(finger_positions, list):
            return finger_positions[0]
        else:
            return finger_positions
            
    def get_object_pose(self):
        object_poses = {}
        for k, v in viewitems(self.config_json):
            if v['require_motive2robot_transform'] == "True" and 'obstacles' not in k:
                try:
                    obj_pose_tf_stamped = self.tf_buffer.lookup_transform(v['parent_frame_id'][1:], v['child_frame_id'][1:], rospy.Time(), rospy.Duration(3.0))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    self.rate.sleep()
                    continue
                    
                obj_pose = np.array([
                    obj_pose_tf_stamped.transform.translation.x,
                    obj_pose_tf_stamped.transform.translation.y,
                    obj_pose_tf_stamped.transform.translation.z,
                    obj_pose_tf_stamped.transform.rotation.x,
                    obj_pose_tf_stamped.transform.rotation.y,
                    obj_pose_tf_stamped.transform.rotation.z,
                    obj_pose_tf_stamped.transform.rotation.w,
                ])
                object_poses.update({k:obj_pose})

        self.object_poses = object_poses

        self.update_all_info()

        return self.object_poses

    def home_robot(self):
        self.driver_utils.home_robot()
        
    def test(self):
        #### go to pose rel curr pose
        # curr_pos, curr_quat = self.driver_utils.get_ee_pose()
        # curr_pose = np.concatenate([curr_pos, curr_quat])

        # goal_pose = curr_pose
        # goal_pose[2] += 0.15
        # goal_pose[1] -= 0.15

        #### go to an object pose
        # from backup.waypoints import waypoints_dict
        # rel_goal_pose = waypoints_dict['relative_toaster']

        
        # from utils.utils import get_object_goal_pose

        # grill_mapped_pose = self.get_object_pose()['grill']

        # goal_pose = get_object_goal_pose(grill_mapped_pose, rel_goal_pose)
        
        # self.set_gripper_state(0.4)
        # while not self.move_to(goal_pose, dry_run=False):
        #     pass

        #### misc
        print(self.get_switch_state())
        
if __name__ == "__main__":
    robot_name = str(sys.argv[1])

    from config import RobotCookingInterfaceConfig

    robot_cooking_interface_config = RobotCookingInterfaceConfig(config={'robot': robot_name}).get_robot_cooking_interface_config()
    robot_cooking_interface_config['init_node'] = True
    
    #### Initialize ####
    cls = RobotCookingInterface(robot_cooking_interface_config)
    time.sleep(.5)
    
    #### test ####
    #cls.test()
    # cls.home_robot()
    #cls.get_switch_state()
    cls.set_gripper_state(0.)
    
    #### test IK ####
    # curr_ee_pos, curr_ee_quat = cls.driver_utils.get_ee_pose()
    # curr_jp = cls.driver_utils.get_joint_values()
    # ik_sol = cls.moveit_utils.ik(position=curr_ee_pos, orientation=curr_ee_quat)
    # ik_jp = ik_sol
    # print(curr_jp)
    # print(ik_jp)
