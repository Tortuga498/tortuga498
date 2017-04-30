#!/usr/bin/env python
import rospy, pcl_ros, tf
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Odometry
from tf.transformations import quaternion_from_euler

import cv2, math, pcl
import numpy as np

pub = rospy.Publisher('/slam_debug', MarkerArray)


def init() :
    global odom_pub, br
    global lines
    global t0, x0, y0, th0
    global odom, odom_trans

    tfListener = tf.TransformListener()
    ready = False
    while not ready :
        try:
            (position, orientation) = tfListener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
            ready = True
        except :
            ready = False
            continue

    (x, y, _) = position
    (_, _, theta) = tf.transformations.euler_from_quaternion(orientation)  
    x0 = x
    y0 = y
    th0 = theta
    t0 = None
    lines = []

    odom_pub = rospy.Publisher('/odom_visual/vo', Odometry, queue_size=50)
    br = tf.TransformBroadcaster()

    odom = Odometry()
    odom.header.frame_id = 'odom_visual'
    odom.child_frame_id = 'vo'


def pub_odom(t, ls):
    global odom_pub, br
    global lines
    global t0, x0, y0, th0
    global odom, odom_trans

    dx = 0
    dy = 0
    dth = 0
    if len(ls) > 0 and len(lines) > 0:
        cnt = 0
        for ([[x11, y11], [x12, y12]]) in ls:
            for ([[x21, y21], [x22, y22]]) in lines:        
                k1 = 1.0
                k2 = 2.0
                a = abs(x12 - x11)
                b = abs(x22 - x21)
                if a <= 0.001 and b <= 0.001:
                    k1 = k2 = 1000000.0
                elif a > 0.001 and b > 0.001:
                    k1 = (y12 - y11) / a
                    k2 = (y22 - y21) / b
                if abs(k2 - k1) < 0.02:
                    a = y12 - y11
                    b = x11 - x12
                    c = x12 * y11 - x11 * y12
                    d = abs(a * x21 + b * y21 + c) / math.hypot(a, b)
                    if d < 0.05:
                        cnt += 1
                        dx = (x11 + x12) / 2.0 - (x21 + x22) / 2.0
                        dy = (y11 + y12) / 2.0 - (y21 + y22) / 2.0
                        dth = math.atan2(k1 - k2, 1 + k1 * k2)
        if cnt > 0:
            dx /= cnt
            dy /= cnt
            dth /= cnt
    if t0 == None:
        t0 = t - 0.1
    dt = t - t0
    t0 = t
    x0 += dx
    y0 += dy
    th0 += dth
    q = quaternion_from_euler(0, 0, th0)
    vx = dx / dt
    vy = dy / dt
    vth = dth / dt
    lines = ls

    current_time = rospy.Time.now()

    br.sendTransform((x0, y0, 0.0), q, current_time, 'odom_visual', 'vo')

    odom.header.stamp = current_time
    odom.pose.pose.position.x = x0
    odom.pose.pose.position.y = y0
    odom.pose.pose.position.z = 0.0
    odom.pose.pose.orientation = q
    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.angular.z = vth;
    odom_pub.publish(odom)


def get_line(p1, v1, id_, color=(0,0,1)):
    marker = Marker()
    marker.header.frame_id = "camera_depth_frame"
    marker.header.stamp = rospy.Time()
    marker.lifetime = rospy.Duration(1)
    marker.ns = "slam_debug_ns"
    marker.id = id_
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 0.0
    marker.scale.x = 0.01
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.points.append(Point(p1[0] + 100 * v1[0], p1[1] + 100 * v1[1], 0))
    marker.points.append(Point(p1[0] - 100 * v1[0], p1[1] - 100 * v1[1], 0))
    return marker


def laser_callback(scan):
    global lines
    newls = [] 
    t = scan.header.stamp.secs + scan.header.stamp.nsecs / 1000000000.0   
    #print scan

    marker_array = MarkerArray()

    # Convert the laserscan to coordinates
    angle = scan.angle_min
    points1 = []    
    for r in scan.ranges:
        theta = angle
        angle += scan.angle_increment
        if (r > scan.range_max) or (r < scan.range_min):
            continue
        if (math.isnan(r)):
            continue

        points1.append([r * math.sin(theta), r * math.cos(theta)]) 

    # Fit the line
    ## convert points to pcl type
    
    while True:
        points = np.array(points1, dtype=np.float32)
        pcl_points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)    
        p = pcl.PointCloud(np.array(pcl_points, dtype=np.float32))
    
    ## create a segmenter object
        seg = p.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_LINE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold (0.003)
    
    ## apply RANSAC
        indices, model = seg.segment()
    # print "Found", len(indices), "inliers", model

        if len(indices) >= 10:         
    # OpenCV line fitter - least squares
    # line = cv2.fitLine(points, 2, 0, 0.01, 0.01)
    # Publish detected lines so we can see them in Rviz
    # marker_array.markers.append(get_line((line[3], line[2]), (line[1], line[0]), 1, (0,1,0)))
    # pub.publish(marker_array)

            marker_array.markers.append(get_line((model[1], model[0]), (model[4], model[3]), 0))
            pub.publish(marker_array)

            l = [[model[1] + 100 * model[4], model[0] + 100 * model[3]],
                 [model[1] - 100 * model[4], model[0] - 100 * model[3]]]
            newls.append(l)

            for i in indices:
                (x, y, _) = p[i]
                #print (x, y)
                ps = points1
                for j in range(len(points1)-1):
                    if abs(points1[j][0] - x) < 0.001 and abs(points1[j][1] - y) < 0.001:
                        ps.remove(points1[j])
                points1 = ps
        elif len(indices) > 0:
            for i in indices:
                (x, y, _) = p[i]
                ps = points1
                for j in range(len(points1)-1):
                    if abs(points1[j][0] - x) < 0.001 and abs(points1[j][1] - y) < 0.001:
                        ps.remove(points1[j])
                points1 = ps
        else:
            break
        if len(points1) < 16:
            break
    #print "======================"
    #print len(lines)
    #print "======================\n"
    pub_odom(t, newls)

def main():
    rospy.init_node('adventure_slam', anonymous=True)

    init()

    rospy.Subscriber("/scan", LaserScan, laser_callback)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
