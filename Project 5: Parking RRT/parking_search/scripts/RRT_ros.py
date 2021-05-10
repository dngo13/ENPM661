#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math
import random
import rospy
import tf
from geometry_msgs.msg import Twist, Point
from tf.transformations import euler_from_quaternion
import computation

# Define initial variables and initial node
rospy.init_node("rrt")
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
start_time = time.time()
ws_width = 500
ws_height = 300
start_location = (20, 20)
img = np.zeros([ws_height, ws_width, 3], dtype=np.uint8)
radius = 33  # cm
wheel_d = 89  # cm
clr = 10
N = 35000  # iterations
edges = []
path = []
# Declare a message of type Twist
velocity_msg = Twist()
# set up a tf listener to retrieve transform between the robot and the world
tf_listener = tf.TransformListener()
parent_frame = 'odom'
# child frame for the listener
child_frame = 'base_footprint'
# publish the velocity at 4 Hz (4 times per second)
rate = rospy.Rate(100)
# proportional gain values for the robot
k_h_gain = 1
k_v_gain = 1


def draw_parking_spaces():
    """ Draws parking spaces for OpenCV"""
    space_1 = cv2.rectangle(img, (50, 300 - 50 - 5), (100 - 5, 300 - 90 + 5), (255, 255, 255), 1)
    space_2 = cv2.rectangle(img, (50, 300 - 90 - 5), (100 - 5, 300 - 130 + 5), (255, 255, 255), 1)
    space_3 = cv2.rectangle(img, (50, 300 - 130 - 5), (100 - 5, 300 - 170 + 5), (255, 255, 255), 1)
    space_4 = cv2.rectangle(img, (50, 300 - 170 - 5), (100 - 5, 300 - 210 + 5), (255, 255, 255), 1)
    space_5 = cv2.rectangle(img, (50, 300 - 210 - 5), (100 - 5, 300 - 250 + 5), (255, 255, 255), 1)
    space_6 = cv2.rectangle(img, (100, 300 - 50 - 5), (150 - 5, 300 - 90 + 5), (255, 255, 255), 1)
    space_7 = cv2.rectangle(img, (100, 300 - 90 - 5), (150 - 5, 300 - 130 + 5), (255, 255, 255), 1)
    space_8 = cv2.rectangle(img, (100, 300 - 130 - 5), (150 - 5, 300 - 170 + 5), (255, 255, 255), 1)
    space_9 = cv2.rectangle(img, (100, 300 - 170 - 5), (150 - 5, 300 - 210 + 5), (255, 255, 255), 1)
    space_10 = cv2.rectangle(img, (100, 300 - 210 - 5), (150 - 5, 300 - 250 + 5), (255, 255, 255), 1)

    space_11 = cv2.rectangle(img, (200, 300 - 50 - 5), (250 - 5, 300 - 90 + 5), (255, 255, 255), 1)
    space_12 = cv2.rectangle(img, (200, 300 - 90 - 5), (250 - 5, 300 - 130 + 5), (255, 255, 255), 1)
    space_13 = cv2.rectangle(img, (200, 300 - 130 - 5), (250 - 5, 300 - 170 + 5), (255, 255, 255), 1)
    space_14 = cv2.rectangle(img, (200, 300 - 170 - 5), (250 - 5, 300 - 210 + 5), (255, 255, 255), 1)
    space_15 = cv2.rectangle(img, (200, 300 - 210 - 5), (250 - 5, 300 - 250 + 5), (255, 255, 255), 1)
    space_16 = cv2.rectangle(img, (250, 300 - 50 - 5), (300 - 5, 300 - 90 + 5), (255, 255, 255), 1)
    space_17 = cv2.rectangle(img, (250, 300 - 90 - 5), (300 - 5, 300 - 130 + 5), (255, 255, 255), 1)
    space_18 = cv2.rectangle(img, (250, 300 - 130 - 5), (300 - 5, 300 - 170 + 5), (255, 255, 255), 1)
    space_19 = cv2.rectangle(img, (250, 300 - 170 - 5), (300 - 5, 300 - 210 + 5), (255, 255, 255), 1)
    space_20 = cv2.rectangle(img, (250, 300 - 210 - 5), (300 - 5, 300 - 250 + 5), (255, 255, 255), 1)

    space_21 = cv2.rectangle(img, (350, 300 - 50 - 5), (400 - 5, 300 - 90 + 5), (255, 255, 255), 1)
    space_22 = cv2.rectangle(img, (350, 300 - 90 - 5), (400 - 5, 300 - 130 + 5), (255, 255, 255), 1)
    space_23 = cv2.rectangle(img, (350, 300 - 130 - 5), (400 - 5, 300 - 170 + 5), (255, 255, 255), 1)
    space_24 = cv2.rectangle(img, (350, 300 - 170 - 5), (400 - 5, 300 - 210 + 5), (255, 255, 255), 1)
    space_25 = cv2.rectangle(img, (350, 300 - 210 - 5), (400 - 5, 300 - 250 + 5), (255, 255, 255), 1)
    space_26 = cv2.rectangle(img, (400, 300 - 50 - 5), (450 - 5, 300 - 90 + 5), (255, 255, 255), 1)
    space_27 = cv2.rectangle(img, (400, 300 - 90 - 5), (450 - 5, 300 - 130 + 5), (255, 255, 255), 1)
    space_28 = cv2.rectangle(img, (400, 300 - 130 - 5), (450 - 5, 300 - 170 + 5), (255, 255, 255), 1)
    space_29 = cv2.rectangle(img, (400, 300 - 170 - 5), (450 - 5, 300 - 210 + 5), (255, 255, 255), 1)
    space_30 = cv2.rectangle(img, (400, 300 - 210 - 5), (450 - 5, 300 - 250 + 5), (255, 255, 255), 1)
    drawn_spaces = [space_1, space_2, space_3, space_4, space_5, space_6, space_7, space_8, space_9, space_10,
                    space_11, space_12, space_13, space_14, space_15, space_16, space_17, space_18, space_19, space_20,
                    space_21, space_22, space_23, space_24, space_25, space_26, space_27, space_28, space_29, space_30]
    spot_1 = cv2.putText(img, '1', (70, 300-65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_2 = cv2.putText(img, '2', (70, 300-105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_3 = cv2.putText(img, '3', (70, 300-145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_4 = cv2.putText(img, '4', (70, 300-185), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_5 = cv2.putText(img, '5', (70, 300-225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_6 = cv2.putText(img, '6', (120, 300-65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_7 = cv2.putText(img, '7', (120, 300-105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_8 = cv2.putText(img, '8', (120, 300-145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_9 = cv2.putText(img, '9', (120, 300-185), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_10 = cv2.putText(img, '10', (115, 300-225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    spot_11 = cv2.putText(img, '11', (215, 300-65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_12 = cv2.putText(img, '12', (215, 300-105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_13 = cv2.putText(img, '13', (215, 300-145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_14 = cv2.putText(img, '14', (215, 300-185), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_15 = cv2.putText(img, '15', (215, 300-225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_16 = cv2.putText(img, '16', (260, 300-65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_17 = cv2.putText(img, '17', (260, 300-105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_18 = cv2.putText(img, '18', (260, 300-145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_19 = cv2.putText(img, '19', (260, 300-185), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_20 = cv2.putText(img, '20', (260, 300-225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    spot_21 = cv2.putText(img, '21', (360, 300-65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_22 = cv2.putText(img, '22', (360, 300-105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_23 = cv2.putText(img, '23', (360, 300-145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_24 = cv2.putText(img, '24', (360, 300-185), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_25 = cv2.putText(img, '25', (360, 300-225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_26 = cv2.putText(img, '26', (410, 300-65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_27 = cv2.putText(img, '27', (410, 300-105), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_28 = cv2.putText(img, '28', (410, 300-145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_29 = cv2.putText(img, '29', (410, 300-185), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    spot_30 = cv2.putText(img, '30', (410, 300-225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
    return drawn_spaces


def define_parking_spaces():
    """ Define the parking spaces to check for goal and obstacles """
    space_1 = [(50, 95), (55, 85)]
    space_2 = [(50, 95), (95, 125)]
    space_3 = [(50, 95), (135, 165)]
    space_4 = [(50, 95), (175, 205)]
    space_5 = [(50, 95), (215, 245)]
    space_6 = [(100, 145), (55, 85)]
    space_7 = [(100, 145), (95, 125)]
    space_8 = [(100, 145), (135, 165)]
    space_9 = [(100, 145), (175, 205)]
    space_10 = [(100, 145), (215, 245)]

    space_11 = [(200, 245), (55, 85)]
    space_12 = [(200, 245), (95, 125)]
    space_13 = [(200, 245), (135, 165)]
    space_14 = [(200, 245), (175, 205)]
    space_15 = [(200, 245), (215, 245)]
    space_16 = [(250, 295), (55, 85)]
    space_17 = [(250, 295), (95, 125)]
    space_18 = [(250, 295), (135, 165)]
    space_19 = [(250, 295), (175, 205)]
    space_20 = [(250, 295), (215, 245)]

    space_21 = [(350, 395), (55, 85)]
    space_22 = [(350, 395), (95, 125)]
    space_23 = [(350, 395), (135, 165)]
    space_24 = [(350, 395), (175, 205)]
    space_25 = [(350, 395), (215, 245)]
    space_26 = [(400, 445), (55, 85)]
    space_27 = [(400, 445), (95, 125)]
    space_28 = [(400, 445), (135, 165)]
    space_29 = [(400, 445), (175, 205)]
    space_30 = [(400, 445), (215, 245)]

    spaces = [space_1, space_2, space_3, space_4, space_5, space_6, space_7, space_8, space_9, space_10,
              space_11, space_12, space_13, space_14, space_15, space_16, space_17, space_18, space_19, space_20,
              space_21, space_22, space_23, space_24, space_25, space_26, space_27, space_28, space_29, space_30]
    return spaces


def parking_spaces_as_obstacles(node, parking_spaces, goal_space):
    """ Function to check obstacles """
    x = node[0]
    y = node[1]
    # dist = clr + radius
    for space in parking_spaces:
        if (space[0][0] - clr) <= x <= (space[0][1] + clr) and (300 - space[1][1] - clr) <= 300 - y <= (300 - space[1][0] + clr):
            # print("Parking Space Full")
            return True
        elif (goal_space[0][0] - clr) <= x <= (goal_space[0][1]) and (300 - goal_space[1][0] - clr) <= 300 - y <= (300 - goal_space[1][0] + clr):
            # print("Found Parking Line")
            return True
        elif (goal_space[0][0] - clr) <= x <= (goal_space[0][1]) and (300 - goal_space[1][1] - clr) <= 300 - y <= (300 - goal_space[1][1] + clr):
            # print("Found Parking Line")
            return True


def heuristic(point1, point2):
    """ Calculates the heuristic between 2 points """
    h = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return h


def determine_goal_node(goal_space):
    """ Define the goal node """
    goal_node_x = (goal_space[0][0] + 10)
    goal_node_y = (goal_space[1][0] + 10)
    goal_node_x = float(goal_node_x)
    goal_node_y = float(goal_node_y)
    final_node = (goal_node_x, goal_node_y)
    return final_node


def closest_node(point, nodes):
    """ Function to find the closest nodes to the current point"""
    min_dist = 10000
    min_id = -1
    # For each index, calculate heuristic and compare distance
    for temp_id in range(len(nodes)):
        dist = heuristic(nodes[temp_id], point)
        if min_dist > dist:
            min_dist = dist
            min_id = temp_id
    if min_id == -1:
        print("Error in closest_node")
    return [min_id, min_dist]


def find_path(item):
    """ Check if theres a path available to the next node """
    global path
    for i in range(len(edges)):
        if item == edges[i][1]:
            new_item = edges[i][0]
            if new_item == 0:
                path.append(0)
                print("Path Found!")
                return
            else:
                path.append(new_item)
                find_path(new_item)


def rrt(start_node, goal_node, parking_spaces):
    """ Main RRT function """
    nodes = [start_node]
    j = 0
    # Max iterations 35000
    for i in range(N):
        # Get random point inside workspace and check closest nodes
        new_node = [np.random.rand() * ws_width, np.random.rand() * ws_height]
        near_node = closest_node(new_node, nodes)
        nearest_node = nodes[near_node[0]]

        x2 = (nearest_node[0] - new_node[0]) / near_node[1]
        y2 = (nearest_node[1] - new_node[1]) / near_node[1]
        x1 = (new_node[0] - nearest_node[0]) / near_node[1]
        y1 = (new_node[1] - nearest_node[1]) / near_node[1]
        # Check if next node is in obstacle
        next_node = [x1 + nearest_node[0], y1 + nearest_node[1]]
        if not parking_spaces_as_obstacles(next_node, parking_spaces, goal_space):
            nodes.append(next_node)
            edges.append([near_node[0], len(nodes) - 1])
        # Check if next node is within threshold for goal node
        if heuristic(next_node, goal_node) < 5:
            print("Goal Reached")
            # Print operation time
            print("--- %s seconds ---" % (time.time() - start_time))
            find_path(edges[-1][0])
            break
        j = j + 1
        # print(j)
    return nodes


def draw(nodes):
    """ Draw the parking spaces and paths on Matplotlib"""
    # First column of lanes
    plt.plot([50, 150], [50, 50], color='black', marker='.')
    plt.plot([50, 150], [90, 90], color='black', marker='.')
    plt.plot([50, 150], [130, 130], color='black', marker='.')
    plt.plot([50, 150], [170, 170], color='black', marker='.')
    plt.plot([50, 150], [210, 210], color='black', marker='.')
    plt.plot([50, 150], [250, 250], color='black', marker='.')
    plt.plot([100, 100], [50, 250], color='black', marker='.')

    # Second column of lanes
    plt.plot([200, 300], [50, 50], color='black', marker='.')
    plt.plot([200, 300], [90, 90], color='black', marker='.')
    plt.plot([200, 300], [130, 130], color='black', marker='.')
    plt.plot([200, 300], [170, 170], color='black', marker='.')
    plt.plot([200, 300], [210, 210], color='black', marker='.')
    plt.plot([200, 300], [250, 250], color='black', marker='.')
    plt.plot([250, 250], [50, 250], color='black', marker='.')

    # Third column of lanes
    plt.plot([350, 450], [50, 50], color='black', marker='.')
    plt.plot([350, 450], [90, 90], color='black', marker='.')
    plt.plot([350, 450], [130, 130], color='black', marker='.')
    plt.plot([350, 450], [170, 170], color='black', marker='.')
    plt.plot([350, 450], [210, 210], color='black', marker='.')
    plt.plot([350, 450], [250, 250], color='black', marker='.')
    plt.plot([400, 400], [50, 250], color='black', marker='.')

    xAxis = []
    yAxis = []
    short_path = []
    # Draws all the branches from RRT
    for node in nodes:
        xAxis.append(node[0])
        yAxis.append(node[1])

    plt.plot(xAxis, yAxis, '.')
    plt.axis([0, ws_width, 0, ws_height])

    # Draw Goal Point
    plt.plot([goal_node[0]], [goal_node[1]], "-ro", markersize=12)
    # Draw shortest path
    for i in path:
        point = nodes[i]
        short_path.append(nodes[i])
        plt.plot(point[0], point[1], "-go", markersize=5)
        cv2.circle(img, (int(point[0]), 300 - int(point[1])), 1, (0, 255, 0), -1)
    short_path.reverse()
    print("Shortest path", short_path)
    # print(len(short_path))
    return short_path


def get_odom_data():
    """Get the current pose of the robot from the /odom topic

    Return
    ----------
    The position (x, y, z) and the yaw of the robot.
    """
    try:
        (trans, rot) = tf_listener.lookupTransform(
            parent_frame, child_frame, rospy.Time(0))
        # rotation is a list [r, p, y]
        rotation = euler_from_quaternion(rot)
    except (tf.Exception, tf.ConnectivityException, tf.LookupException):
        rospy.loginfo("TF Exception")
        return
    # return the position (x, y, z) and the yaw
    return Point(*trans), rotation[2]


def rotate(angle_degree, angular_velocity):
    """Make the robot rotate in place

    The angular velocity is modified before publishing the message on the topic /cmd_vel.
    """

    # angular_velocity = math.radians(angular_velocity)
    velocity_msg.linear.x = 0.0
    velocity_msg.angular.z = angular_velocity

    t0 = rospy.Time.now().to_sec()
    while True:
        # rospy.loginfo("TurtleBot is rotating")
        pub.publish(velocity_msg)
        rate.sleep()
        t1 = rospy.Time.now().to_sec()
        # rospy.loginfo("t0: {t}".format(t=t0))
        # rospy.loginfo("t1: {t}".format(t=t1))
        current_angle_degree = (t1 - t0) * angular_velocity

        # rospy.loginfo("current angle: {a}".format(a=current_angle_degree))
        # rospy.loginfo("angle to reach: {a}".format(a=angle_degree))
        if abs(current_angle_degree) >= math.radians(abs(angle_degree)):
            rospy.loginfo("reached")
            break
    # finally, stop the robot when the distance is moved
    velocity_msg.angular.z = 0
    pub.publish(velocity_msg)


def go_straight(distance_to_drive, linear_velocity):
    """Move the robot in a straight line until it has driven a certain distance.

    The linear velocity is modified for a Twist message and then published on /cmd_vel.

    """

    global velocity_msg
    # update linear.x from the parameter passed
    velocity_msg.linear.x = linear_velocity
    velocity_msg.angular.z = 0.0
    # get the current time (s)
    t_0 = rospy.Time.now().to_sec()
    # keep track of the distance
    distance_moved = 0.0

    # while the amount of distance has not been reached
    while distance_moved <= distance_to_drive:
        # rospy.loginfo("TurtleBot is moving")
        pub.publish(velocity_msg)
        rate.sleep()
        # time in sec in the loop
        t_1 = rospy.Time.now().to_sec()
        distance_moved = (t_1 - t_0) * abs(linear_velocity)
        # rospy.loginfo("distance moved: {0}".format(distance_moved))

    rospy.loginfo("Distance reached")
    # finally, stop the robot when the distance is moved
    velocity_msg.linear.x = 0.0
    velocity_msg.angular.z = 0.0
    pub.publish(velocity_msg)


def drive_to_space(goal_node, next_node, space_to_drive):
    """ Function for the robot to drive to the goal """
    # Get odometry data
    (pos, rot) = get_odom_data()
    last_rotation = 0
    # Convert meters to centimeters
    next_x = next_node[0] / 100
    next_y = next_node[1] / 100
    goal_x = goal_node[0] / 100
    goal_y = goal_node[1] / 100
    count = 0
    at_goal = True
    # Calculate the distance between current node and next node
    distance_to_goal = computation.compute_distance(pos.x, pos.y, next_x, next_y)
    # while the distance to the goal is greater than the threshold, compute
    while distance_to_goal > 0.3:
        (pos, rot) = get_odom_data()
        x_start = pos.x
        y_start = pos.y
        # Calculate angle to goal
        angle_to_goal = math.atan2(next_y - y_start, next_x - x_start)
        if angle_to_goal < -math.pi / 4 or angle_to_goal > math.pi / 4:
            if 0 > next_y > y_start:
                angle_to_goal = -2 * math.pi + angle_to_goal
            elif 0 <= next_y < y_start:
                angle_to_goal = 2 * math.pi + angle_to_goal
        if last_rotation > math.pi - 0.1 and rot <= 0:
            rot = 2 * math.pi + rot
        elif last_rotation < -math.pi + 0.1 and rot > 0:
            rot = -2 * math.pi + rot
        # Multiply gain to get the angular velocity
        velocity_msg.angular.z = k_v_gain * angle_to_goal - rot
        distance_to_goal = computation.compute_distance(pos.x, pos.y, next_x, next_y)
        print(distance_to_goal)
        # proportional control to move the robot forward
        velocity_msg.linear.x = min(k_h_gain * distance_to_goal, 0.1)

        # set the z angular velocity for positive and negative rotations
        if velocity_msg.angular.z > 0:
            velocity_msg.angular.z = min(velocity_msg.angular.z, 0.1)
        else:
            velocity_msg.angular.z = max(velocity_msg.angular.z, -0.1)
        print("rotation :", rot)
        print("position :", pos.x, pos.y)
        if goal_x - 0.45 < pos.x and -0.3 < rot < 0.6 and (
                1 <= space_to_drive <= 5 or 11 <= space_to_drive <= 15 or 21 <= space_to_drive <= 25) and count == 0:
            rotate(75, 0.15)
        elif goal_x - 0.45 < pos.x and -0.3 < rot < 0.9 and (21 <= space_to_drive <= 25) and count == 0:
            rotate(65, 0.15)
        elif goal_y - 0.15 < pos.y < goal_y + 0.35 and 0.8 < rot < 1.65 and (
                3 <= space_to_drive <= 5 or 13 <= space_to_drive <= 15 or 23 <= space_to_drive <= 25) and count == 0:
            rotate(80, -0.2)
            go_straight(0.35, 0.15)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            print("At Goal")
            count = 3
        elif goal_y < pos.y < goal_y + 0.35 and 0.8 < rot < 1.65 and (
                1 <= space_to_drive <= 2 or 11 <= space_to_drive <= 12 or 21 <= space_to_drive <= 22) and count == 0:
            rotate(80, -0.2)
            go_straight(0.35, 0.15)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            print("At Goal")
            count = 3
        elif pos.y > goal_y - 0.25 and 0.8 < rot < 1.65 and (
                space_to_drive == 2 or space_to_drive == 12 or space_to_drive == 22):
            go_straight(0.32, 0.1)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            time.sleep(0.5)
        elif pos.y > goal_y - 0.25 and 0.8 < rot < 1.65 and (
                3 <= space_to_drive <= 5 or 13 <= space_to_drive <= 15 or 23 <= space_to_drive <= 25):
            go_straight(0.25, 0.1)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            time.sleep(0.5)
        elif goal_x + 0.55 < pos.x and -0.3 < rot < 0.6 and (
                6 <= space_to_drive <= 10 or 16 <= space_to_drive <= 26 or 26 <= space_to_drive <= 30):
            rotate(75, 0.15)
        elif goal_y - 0.15 < pos.y < goal_y + 0.35 and 0.8 < rot < 2.3 and (6 <= space_to_drive <= 10):
            rotate(75, 0.2)
            go_straight(0.45, 0.15)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            print("At Goal")
            count = 3
        elif goal_y - 0.15 < pos.y < goal_y + 0.35 and 0.8 < rot < 2.3 and (
                16 <= space_to_drive <= 20 or 26 <= space_to_drive <= 30):
            rotate(85, 0.2)
            go_straight(0.45, 0.15)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            print("At Goal")
            count = 3
        elif pos.y > goal_y - 0.25 and 0.8 < rot < 2.3 and (
                space_to_drive == 7 or space_to_drive == 17 or space_to_drive == 27):
            go_straight(0.32, 0.1)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            time.sleep(0.5)
        elif pos.y > goal_y - 0.25 and 0.8 < rot < 2.3 and (
                8 <= space_to_drive <= 10 or 18 <= space_to_drive <= 20 or 28 <= space_to_drive <= 30):
            go_straight(0.25, 0.1)
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            pub.publish(velocity_msg)
            time.sleep(0.5)
        # update the new rotation for the next loop
        print(count)
        last_rotation = rot
        pub.publish(velocity_msg)
        rate.sleep()

    # force the robot to stop by setting linear and angular velocities to 0
    (pos, rot) = get_odom_data()
    if distance_to_goal < 0.3 and goal_y - 0.1 < pos.y < goal_y + 0.35 and 0.8 < rot < 1.65 and (1 <= space_to_drive <= 5 or 11 <= space_to_drive <= 15 or 21 <= space_to_drive <= 25):
        # print("executing")
        rotate(80, -0.2)
        go_straight(0.45, 0.15)
        velocity_msg.linear.x = 0.0
        velocity_msg.angular.z = 0.0
        pub.publish(velocity_msg)
        print("At Goal")
        at_goal = False
        count = 3
    if count == 3:
        at_goal = False
    velocity_msg.linear.x = 0.0
    velocity_msg.angular.z = 0.0
    # publish the new message on /cmd_vel topic
    pub.publish(velocity_msg)
    return at_goal


if __name__ == "__main__":
    parking_spaces = define_parking_spaces()
    space_map = draw_parking_spaces()
    cv2.imshow('Parking Spaces', img)
    cv2.waitKey(0)
    space_to_drive = int(input("What space do you want to park in? "))
    print("Calculating...please wait")
    goal_space = parking_spaces[space_to_drive-1]
    goal_node = determine_goal_node(goal_space)

    cv2.rectangle(img, (parking_spaces[space_to_drive-1][0][0], 300 - parking_spaces[space_to_drive-1][1][0]),
                  (parking_spaces[space_to_drive-1][0][1], 300 - parking_spaces[space_to_drive-1][1][1]), (0, 255, 0), -1)
    parking_spaces.pop(space_to_drive - 1)
    parking_spaces_as_obstacles(goal_node, parking_spaces, goal_space)
    nodes = rrt(start_location, goal_node, parking_spaces)
    # in the original nodes, the start was a tuple (20,20) and caused issues
    # nodesnew is converting that tuple to a regular list like the rest of the nodes to avoid conversion problems
    nodesnew = [[start_location[0], start_location[1]]]
    for i in range(1, len(nodes)):
        nodesnew.append(nodes[i])
    fig, ax = plt.subplots()
    shortest_path = draw(nodes)
    plt.show()
    cv2.imshow('Path to Goal', img)
    cv2.waitKey(0)
    at_goal = True
    # Saves nodes to text file
    file = open("nodes.txt", "w+")
    for i in nodesnew:
        file.write(str(i) + "\n")
        file.write("-------------" + "\n")
    file.close()
    branches = []
    for i in range(len(nodesnew) - 1):
        branches.append([nodesnew[i][0], nodesnew[i][1],
                         nodesnew[i + 1][0], nodesnew[i + 1][1]])

    file = open("branches.txt", "w+")
    for i in branches:
        file.write(str(i) + "\n")
    file.close()

    # Save branches array to file with numpy
    branches_arr = np.array(branches)
    file = open("branches_arr", "wb")
    np.save(file, branches_arr)
    file.close()

    # Creates shortest_path text file to store the path to the goal
    for i in range(len(shortest_path) - 1):
        at_goal = drive_to_space(goal_node, shortest_path[i+1], space_to_drive)
        # print(at_goal)
        if not at_goal:
            break
        # print(i)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

