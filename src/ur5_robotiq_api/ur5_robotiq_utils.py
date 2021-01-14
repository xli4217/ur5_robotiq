import sys

import rospy
from std_msgs.msg import UInt16
from sensor_msgs.msg import JointState
import numpy as np
import copy

import moveit_commander
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint, ExecuteTrajectoryActionGoal, ExecuteTrajectoryActionFeedback
from actionlib_msgs.msg import GoalID, GoalStatusArray
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from  geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list

import json
from future.utils import viewitems
import numpy as np
import tf
import tf2_ros
import geometry_msgs.msg
import time


default_config = {
    "arm": 'right',
    "env_json_path": "",
}

class UR5RobotiqUtils(object):
    def __init__(self, config={}):
        self.UR5RobotiqUtils_config = default_config
        self.UR5RobotiqUtils_config.update(config)

        rospy.init_node('ur5_robotiq_utils', anonymous=True)
        rospy.on_shutdown(self.cleanup)

        self.create_publishers_and_subscribers()

        #############################
        # Initiate member variables #
        #############################
        self.action_feedback = None
        self.current_goal_id = None
        
        ###############
        # Moveit init #
        ###############

        #### initialize move_group API ####
        moveit_commander.roscpp_initialize(sys.argv)

        #### initialize robot ####
        self.robot = moveit_commander.RobotCommander()
        
        #### initialize move group ####
        self.arm = moveit_commander.MoveGroupCommander('manipulator') 

        #### set tool link ####
        self.arm .set_end_effector_link("tool0")
        
        #### get name of end-effector link ####
        self.ee_link = self.arm.get_end_effector_link()
        
        #### get reference frame for pose targets ####
        self.reference_frame = "base_link"

        #### setup ur5 reference frame ####
        self.arm.set_pose_reference_frame(self.reference_frame)

        #### allow replanning ####
        self.arm.allow_replanning(True)

        #### allow some leeway in position (m) and orientation (rad) ####
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.1)

        #### neutral position for reset ####
        self.neutral_joint_positions = [2.996885061264038,
                                        -1.3486784140216272,
                                        1.112940788269043,
                                        -0.805460278187887,
                                        -0.4705269972430628,
                                        -1.7602842489825647]
        
        #### joint names ####
        self.joint_names=['shoulder_pan_joint',
                          'shoulder_lift_joint',
                          'elbow_joint',
                          'wrist_1_joint',
                          'wrist_2_joint',
                          'wrist_3_joint']

        #### Initialize IK Service ####
        try:
            self.moveit_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
            self.moveit_ik.wait_for_service()
   
        except rospy.ServiceException, e:
            rospy.logerror("Service call failed: %s" %e)


        #############
        # tf buffer #
        #############
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # time.sleep(2)

        # # Get info about environment through json (mush be the same one passed to baxter_api.launch)
        # with open(self.UR5RobotiqUtils_config.get('env_json_path'), 'rb') as f:
        #     self.env = json.loads(f.read())

            
    def create_publishers_and_subscribers(self):
        ##############
        # Publishers #
        ##############

        #### display trajectory publisher ####
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        #### cancel current path execution publisher ####
        self.cancel_execution_publisher = rospy.Publisher('/execute_trajectory/cancel', GoalID, queue_size=10)
        
        ###############
        # Subscribers #
        ###############

        #### execute cartesian path action feedback ####
        self.execute_traj_status_sub = rospy.Subscriber('/execute_trajectory/status', GoalStatusArray, self.cb_execute_traj_status)

        time.sleep(1.)

    def cb_execute_traj_status(self, msg):
        self.current_goal_id = msg.status_list[0].goal_id.id
        
    def get_joint_values(self):
        return self.arm.get_current_joint_values()

    def get_robot_state(self):
        return self.robot.get_current_state()

    def get_rpy(self):
        return self.arm.get_current_rpy()
        
    def get_planning_frame(self):
        return self.arm.get_planning_frame()

    def get_current_pose(self):
        return self.arm.get_current_pose()

        
    def ik(self, pose_stamped=None, position=None, orientation=None):
        '''
        poisiton = [x,y,z], orientation = [i,j,k,w]
        '''

        if pose_stamped is not None:
            ps = pose_stamped
        elif position is not None and orientation is not None:
            ps = PoseStamped()

            ps.header.time = rospy.get_rostime()
            ps.header.frame_id = "world"
            
            ps.pose.position.x = position[0]
            ps.pose.position.y = position[1]
            ps.pose.position.z = position[2]
            
            ps.pose.orientation.x = orientation[0]
            ps.pose.orientation.y = orientation[1]
            ps.pose.orientation.z = orientation[2]
            ps.pose.orientation.w = orientation[3]
        else:
            raise ValueError("need to provide at either pose_stamped or position and orientation")
            
        req = GetPositionIKRequest()
        req.ik_request.group_name = "manipulator"
        req.ik_request.robot_state = self.get_robot_state()
        req.ik_request.pose_stamped = ps
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.timeout = rospy.Duration(1.0)
        req.ik_request.attempts = 1
        req.ik_request.avoid_collisions = False
        
        try:
            resp = self.moveit_ik.call(req)
            joint_names = resp.solution.joint_state.name
            joint_values = resp.solution.joint_state.position
            return joint_values
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999  # Failure
            return resp

    # def fk(self, joint_angles):
    #     '''
    #     joint_angles is a list of length 6
    #     '''
    #     arm = self.UR5RobotiqUtils_config['arm']
    #     rospy.wait_for_service('compute_fk')
        # try:
        #     moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
        # except rospy.ServiceException, e:
        #     rospy.logerror("Service call failed: %s" %e)

    #     fkln = [arm+"_lower_elbow",
    #             arm+"_lower_shoulder",
    #             arm+"_lower_forearm",
    #             arm+"_upper_elbow",
    #             arm+"_upper_shoulder",
    #             arm+"_upper_forearm",
    #             arm+"_wrist"]

    #     # in robot_state this should be [s0, s1, e0, e1, w0, w1, w2]
    #     joint_names=[arm+"_e0",arm+"_e1",arm+"_s0",arm+"_s1",arm+"_w0",arm+"_w1",arm+"_w2"]

    #     header = Header(0, rospy.Time.now(), "/world")
    #     rs = RobotState()
    #     rs.joint_state.name = joint_names
    #     rs.joint_state.position = joint_angles

    #     fk = moveit_fk(header,fkln,rs)
    #     #print fk.pose_stamped[1]
    #     #rospy.loginfo(["FK LOOKUP:", moveit_fk(header, fkln, rs)])
    #     return fk

    def set_joint_velocity(self, jv):
        pass
        
    def move_to_joint_target(self, jp, wait=True):
        self.arm.set_joint_value_target(jp)
        self.arm.go(wait=wait)

    def move_to_pose_target(self, pt, wait=True):
        self.arm.clear_pose_targets()
        self.arm.set_pose_target(pt)
        self.arm.go(wait=wait)

    def execute_cartesian_path(self, wp=None):
        scale = 1.
        waypoints = []
        # waypoints.append(copy.deepcopy(wp))

        wpose = self.get_current_pose().pose

        waypoints.append(copy.deepcopy(wpose))

        wpose.position.z -= scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)

        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))
        
        wpose.position.y += scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y += scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))


        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0 disabling:
        (plan, fraction) = self.arm.compute_cartesian_path(waypoints,   # waypoints to follow
                                                           0.01,        # eef_step
                                                           0.0)         # jump_threshold
        
        # self.publish_display_trajectory(plan)
        self.arm.execute(plan, wait=False)

        
        
    def cancel_current_cartesian_path_execution(self):
        cancel_goal_id = GoalID()
        cancel_goal_id.stamp = rospy.Time.now()
        cancel_goal_id.id = self.current_goal_id

        self.cancel_execution_publisher.publish(cancel_goal_id)
        
    def publish_display_trajectory(self, plan):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory);
        
    def reset(self):
        self.move_to_joint_target(self.neutral_joint_positions, wait=True)
        
    def get_interactive_object_position(self):
        for k, v in viewitems(self.env):
            if v['marker_type'] == "interactive":
                object_pose = self.tf_buffer.lookup_transform(
                    value['parent_frame_id'][1:], value['child_frame_id'][1:],
                    rospy.Time())
                object_pose = np.array([object_pose.transform.translation.x,
                                        object_pose.transform.translation.y,
                                        object_pose.transform.translation.z])

                return object_pose


    def cleanup(self):
        # rospy.loginfo('stopping the robot')
        # self.arm.stop()
        pass
        
        
if __name__ == "__main__":
    import time
    from std_msgs.msg import Float32MultiArray

    cls = UR5RobotiqUtils()
    pub = rospy.Publisher('comparison', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(20.)
    cls.reset()

    cls.execute_cartesian_path()
    # cls.cancel_current_cartesian_path_execution()

    #### test IK ####
    # print("joint angles")
    # print(cls.get_joint_values())
    # ps = cls.get_current_pose()
    # print("ik solution")
    # print(cls.ik(ps))
    
    
    # for i in range(1000):
    #     d = Float32MultiArray()

    #     p = cls.get_current_pose()
    #     p.pose.position.x += 0.007 * np.sin(0.1*i)
    #     p.pose.position.z += 0.007 * np.sin(0.1*i)

    #     d.data.append(p.pose.position.x)
    #     d.data.append(p.pose.position.z)
        
    #     #### test IK pose target ####
    #     jv = cls.ik(p)
    #     cls.move_to_joint_target(jv, wait=False)
    #     pose = cls.get_current_pose()
    #     d.data.append(pose.pose.position.x)
    #     d.data.append(pose.pose.position.z)
    #     pub.publish(d)
    #     rate.sleep()
        
    #     #### test joint target ####
    #     joint_p = cls.get_joint_values()
    #     joint_p[0] += 0.1 * np.sin(i)
    #     cls.move_to_joint_target(joint_p, wait=True)
        
        #### test pose target ####
        # cls.move_to_pose_target(p, wait=True)
        # print((p.position.x, p.position.z))
        # print((cls.get_current_pose().pose.position.x, cls.get_current_pose().pose.position.z))
        # print("-----")

        #### test cartesian path ####
        # cls.execute_cartesian_path(p)
        
        # time.sleep(0.02)