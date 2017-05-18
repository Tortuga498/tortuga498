#! /usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from math import radians, degrees


def coord_client(x, y, yaw):
  goal = MoveBaseGoal()
  goal.target_pose.header.frame_id = '/map'
  goal.target_pose.pose.position.x = x
  goal.target_pose.pose.position.y = y
  goal.target_pose.pose.position.z = 0.0

  angle = radians(yaw)
  quat = quaternion_from_euler(0.0,0.0,angle)
  goal.target_pose.pose.orientation = Quaternion(*quat.tolist())

  return goal

def callback_pose(data):
  x = data.pose.pose.position.x
  y = data.pose.pose.position.y
  roll, pitch, yaw = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
  

if __name__ == '__main__':
  rospy.init_node("coord_nav")
  rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback_pose)  

  nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
  nav_as.wait_for_server()
  
  nav_goal = coord_client(2.7,-1.6,0.0)
  nav_as.send_goal(nav_goal)
  nav_as.wait_for_result()

  nav_goal = coord_client(0.4,0.0,0.0)
  nav_as.send_goal(nav_goal)
  nav_as.wait_for_result()

  nav_goal = coord_client(0.7,-1.6,0.0)
  nav_as.send_goal(nav_goal)
  nav_as.wait_for_result()