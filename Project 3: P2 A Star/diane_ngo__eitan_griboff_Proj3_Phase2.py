#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
from queue import PriorityQueue
import time
# import matplotlib.pyplot as plt
start_time = time.time()
# Define workspace
ws_width = 400
ws_height = 300
# Clearance
# global clr
# Euclidean distance
euc = 0.5
# Theta threshold
theta_thresh = 30
img = np.zeros([ws_height, ws_width, 3], dtype=np.uint8)


# Function to create obstacles in the workspace using OpenCV
def create_obstacles():
    # Circle
    cv2.circle(img, (90, 230), 35, (255, 255, 255), -1)
    # Ellipse
    cv2.ellipse(img, (246, 155), (60, 30), 0, 0, 360, (255, 255, 255), -1)
    # Rectangle
    rect_cnrs = np.array([[48, 192], [171, 106], [160, 90],
                          [37, 176]], np.int32)
    cv2.fillPoly(img, [rect_cnrs], (255, 255, 255), 1)
    # C shape
    c_cnrs = np.array([[200, 20], [230, 20], [230, 30], [210, 30], [210, 60],
                       [230, 60], [230, 70], [200, 70]], np.int32)
    cv2.fillPoly(img, [c_cnrs], (255, 255, 255), 1)
    return img


# Function to check if the node is in an obstacle
# Defaultclearance is 5 but user defined
def check_in_obstacle(node,clr=5):
    # print(clr)
    x, y = node[0], node[1]
    # y = 300 - y
     # Inside circle
    if (x - 90) ** 2 + (y - 70) ** 2 <= (35 + clr) ** 2:
        # print("Coordinate inside circle obstacle")
        return True
    # Ellipse
    elif((x-246) ** 2) / (60+clr) ** 2 + (((y-145) ** 2)/(30+clr) ** 2) <= 1:
        # print("Coordinate inside ellipse obstacle")
        return True
    # C shape
    elif (200 - clr <= x <= 210 + clr and 230 - clr <= y <= 280 + clr) or \
            (210 - clr <= x <= 230 + clr and 270 - clr <= y <= 280 + clr) \
            or (210 - clr <= x <= 230 + clr and 230 - clr <= y <= 240 + clr):
        # print("Coordinate inside C shaped obstacle")
        return True
    # Angled rectangle
    elif (y - 0.7 * x - clr >= 0) and (y - 0.7 * x - 100 - clr <= 0) and \
            (y + 1.4*x-176.64 + clr >= 0) and (y + 1.4*x-438.64 - clr <= 0):
        # print("Coordinate inside rectangle")
        return True
    else:
        # print("Coordinates are good to go")
        return False
    
# def create_map(clr):
#     obsx = []
#     obsy = []
#     for i in range(300):
#         for j in range(400):
#             node = [i, j]
#             if check_in_obstacle(node, clr):
#                 obsx.append(i)
#                 obsy.append(j)
#     return obsx, obsy

# Function to check if current node is at goal node
def check_solution(curr_node, goal_node):
    if curr_node == goal_node:
        return True
    else:
        return False


# Function for start and goal coordinates by user input
# Note that the origin of the workspace is bottom lef2
def get_coords():
    # Start positions
    start_x, start_y = [int(x) for x in input(
        "Enter starting x, y position: ").split()]
    # Default theta: 30
    start_th = int(input("Enter starting theta position: "))
    # start_th = 30  # Default
    while start_x > ws_width or start_y > ws_height:
        print("Start cannot be outside of workspace")
        start_x, start_y = [int(x) for x in input(
            "Enter starting x, y position: ").split()]
        start_y = img.shape[0] - start_y
    print("Start coordinates are not outside of workspace")

    start_node = (start_x, start_y, start_th)

    # While start positions are inside an obstacle ask for new input
    while check_in_obstacle(start_node):
        print("Start cannot be inside an obstacle")
        start_x, start_y = [int(x) for x in input(
            "Enter starting x, y position: ").split()]
        start_y = img.shape[0] - start_y
        start_node = (start_x, start_y)

    start_node = (start_x, start_y, start_th)

    # Inputs for goal
    # Note ignore theta g atm
    goal_x, goal_y = [int(x) for x in input(
        "Enter end goal x, y position: ").split()]
    # goal_th = int(input("Enter goal theta position: "))
    goal_th = 30
    while goal_x > ws_width or goal_y > ws_height:
        print("Goal cannot be outside of workspace")
        goal_x, goal_y = [int(x) for x in input(
            "Enter end goal x, y position: ").split()]
        goal_y = img.shape[0] - goal_y
    print("Goal coordinates are not outside of workspace")

    goal_node = (goal_x, goal_y, goal_th)

    # While goal positions are inside obstacles get new coordinates
    while check_in_obstacle(goal_node):
        print("Goal cannot be inside an obstacle")
        goal_x, goal_y = [int(x) for x in input(
            "Enter end goal x, y position: ").split()]
        # goal_node = (goal_x, goal_y)
        goal_y = img.shape[0] - goal_y
        goal_node = (goal_x, goal_y, goal_th)
    goal_node = (goal_x, goal_y, goal_th)
    return start_x, start_y, start_th, goal_x, goal_y, goal_th, start_node, goal_node


# Defining movements for the robot
def action_step(current_node, step_size, th_tresh):
    th_thresh_rad = math.radians(th_tresh)
    angle = math.radians(current_node[1][2])
    actions = [[step_size * math.cos(angle), step_size * math.sin(angle), 0, step_size],
               [step_size * math.cos(angle + th_thresh_rad), step_size * math.sin(angle + th_thresh_rad), th_tresh,
                step_size],
               [step_size * math.cos(angle + (2 * th_thresh_rad)), step_size * math.sin(angle + (2 * th_thresh_rad)),
                2 * th_tresh, step_size],
               [step_size * math.cos(angle - th_thresh_rad), step_size * math.sin(angle - th_thresh_rad), - th_tresh,
                step_size],
               [step_size * math.cos(angle - (2 * th_thresh_rad)), step_size * math.sin(angle - (2 * th_thresh_rad)),
                -2 * th_tresh, step_size]]
    return actions, step_size


# Function to calculate distance between points
def heuristic(point1, point2):
    h = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return h

# Function to convert x, y, and theta with the threshold
def discretize(x, y, theta, th_thresh, euc_thresh):
    x = (round(x / euc_thresh) * euc_thresh)
    y = (round(y / euc_thresh) * euc_thresh)
    theta = (round(theta / th_thresh) * th_thresh)
    return x, y, theta


# A* algorithm function
def a_star(start_node, goal_node):
    print("======== A* ========")
    # Creates initial empty variables
    solvable = True
    path = {}
    child_node = []
    current_node = []
    parent_node = []
    visited = np.zeros([int(400 / euc), int(300 / euc), int(360 / theta_thresh)])
    q = PriorityQueue()
    # Start and goal vectors
    start = (0, start_node, None)
    goal = (0, goal_node, None)
    start_node = discretize(start_node[0], start_node[1], start_node[2], theta_thresh, euc)
    goal_node = discretize(goal_node[0], goal_node[1], goal_node[2], theta_thresh, euc)
    q.put(start)  # adding the cost values in the queue
    if solvable:
        while q:
            curr_node = list(q.get())
            curr_node[0] = curr_node[0] - heuristic(curr_node[1], goal[1])

            if curr_node[2] is not None:
                node1 = str(curr_node[1][0])
                node2 = str(curr_node[1][1])
                node3 = str(curr_node[1][2])

                parent1 = str(curr_node[2][0])
                parent2 = str(curr_node[2][1])
                parent3 = str(curr_node[2][2])

                node_str = node1 + ',' + node2 + ',' + node3
                parent_str = parent1 + ',' + parent2 + ',' + parent3
                path[node_str] = parent_str
            else:
                node1 = str(curr_node[1][0])
                node2 = str(curr_node[1][1])
                node3 = str(curr_node[1][2])
                parent1 = str(curr_node[2])

                node_str = node1 + ',' + node2 + ',' + node3
                parent_str = parent1
                path[node_str] = parent_str

            moves, step_size = action_step(curr_node, 1, theta_thresh)

            for next_node in moves:
                angle = next_node[2] + curr_node[1][2]
                if angle < 0:
                    angle = angle + 360
                elif angle > 360:
                    angle = angle - 360
                elif angle == 360:
                    angle = 0

                node = (curr_node[1][0] + next_node[0], curr_node[1][1] + next_node[1], angle)
                node = discretize(node[0], node[1], node[2], theta_thresh, euc)
                node_cost = curr_node[0] + next_node[3] + heuristic(node, goal[1])
                node_parent = curr_node[1]

                if not check_in_obstacle(node):
                    if (ws_width - clr) > node[0] > 0 and (ws_height - clr) > node[1] > 0:
                        if visited[int(node[0] / euc)][int(node[1] / euc)][int(node[2] / theta_thresh)] == 0:
                            visited[int(node[0] / euc)][int(node[1] / euc)][int(node[2] / theta_thresh)] = 1
                            parent_node1 = (node_parent[0], node_parent[1])
                            parent_node.append(parent_node1)
                            child_node = (node[0], node[1])
                            current_node.append(child_node)
                            # print(child_node)
                            next_node = (node_cost, node, node_parent)
                            q.put(next_node)

            if check_solution(curr_node[1], goal_node):
                print(">>At goal<<")
                # Print operation time
                print("--- %s seconds ---" % (time.time() - start_time))
                backtracked = []
                # Set parent points 
                p1 = str(curr_node[2][0])
                p2 = str(curr_node[2][1])
                p3 = str(curr_node[2][2])
                parent = p1 + ',' + p2 + ',' + p3
                # While parent is not none
                while parent != "None":
                    # Set temp to get the parent
                    temp = path.get(parent)
                    # Loop through to find backtrack nodes
                    if parent[1] == '.' and parent[5] == '.':
                        param1 = float(parent[0]) + float(parent[2]) / 10
                        param2 = float(parent[4]) + float(parent[6]) / 10
                    if parent[2] == '.' and parent[7] == '.':
                        param1 = float(parent[0] + parent[1]) + float(parent[3]) / 10
                        param2 = float(parent[5] + parent[6]) + float(parent[8]) / 10
                    if parent[1] == '.' and parent[6] == '.':
                        param1 = float(parent[0]) + float(parent[2]) / 10
                        param2 = float(parent[4] + parent[5]) + float(parent[7]) / 10
                    if parent[2] == '.' and parent[6] == '.':
                        param1 = float(parent[0] + parent[1]) + float(parent[3]) / 10
                        param2 = float(parent[5]) + float(parent[7]) / 10
                    if parent[3] == '.' and parent[9] == '.':
                        param1 = float(parent[0] + parent[1] + parent[2]) + float(parent[4]) / 10
                        param2 = float(parent[6] + parent[7] + parent[8]) + float(parent[10]) / 10
                    if parent[3] == '.' and parent[7] == '.':
                        param1 = float(parent[0] + parent[1] + parent[2]) + float(parent[4]) / 10
                        param2 = float(parent[6]) + float(parent[8]) / 10
                    if parent[3] == '.' and parent[8] == '.':
                        param1 = float(parent[0] + parent[1] + parent[2]) + float(parent[4]) / 10
                        param2 = float(parent[6] + parent[7]) + float(parent[9]) / 10
                    if parent[1] == '.' and parent[7] == '.':
                        param1 = float(parent[0]) + float(parent[2]) / 10
                        param2 = float(parent[4] + parent[5] + parent[6]) + float(parent[8]) / 10
                    if parent[2] == '.' and parent[8] == '.':
                        param1 = float(parent[0] + parent[1]) + float(parent[3]) / 10
                        param2 = float(parent[5] + parent[6] + parent[7]) + float(parent[9]) / 10
                    backtracked.append((param1, param2))
                    parent = temp
                    # If the parameters are the start node, can terrminate loop
                    if (param1, param2) == (start_node[0], start_node[1]):
                        break
                backtracked.append((start_node[0], start_node[1]))
                return backtracked, current_node, parent_node, visited
    
    
# Function for visualizing the path and search
def visual(path, img, current_node, parent, out):
    imgcopy = img.copy()
    # Set search color to blue
    for i in range(len(current_node)-1):
        imgcopy[299 - int(current_node[i][1]), int(current_node[i][0]) - 1] = [255, 0, 0]
        # cv2.circle(imgcopy, (299 - int(current_node[i][1]), int(current_node[i][0]) - 1), 3, (255, 255, 0))
        out.write(imgcopy)
    # Draws the optimal path
    for j in range(len(path) - 1):
        imgcopy = cv2.line(imgcopy, (int(path[j][0]) - 1, 299 - int(path[j][1])),
                           (int(path[j+1][0]) - 1, 299 - int(path[j+1][1])), (255, 255, 255), 2)
        out.write(imgcopy)
    # Saves to video and returns the frame
    return imgcopy


# Main function
def main():
    # Define global variables and user inputs
    global clr  # clearance
    clr = int(input("Define the clearance distance between objects:"))
    # global step_size # step size
    # step_size = int(input("Define the step size, between 1 and 10: "))
    # Create obstacles in map
    ws_map = create_obstacles()
    # Ask user for start and goal position
    start_x, start_y, start_th, goal_x, goal_y, goal_th, start_node, goal_node = get_coords()
    # Call A star function and return the shortest path
    shortest_path, current_node, parent_node, visited = a_star(start_node, goal_node)
    # print(shortest_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Astar.mp4', fourcc, 4000, (400, 300), isColor=True)
    # Visualize the search and optimal path
    imgcopy = visual(shortest_path, ws_map, current_node, parent_node, out)
    cv2.circle(imgcopy, (start_x, 300 - start_y), radius=1,
               color=(0, 255, 0), thickness=3)
    cv2.circle(imgcopy, (goal_x, 300 - goal_y), radius=1,
               color=(0, 0, 255), thickness=3)
    out.write(imgcopy)
    # X = []
    # Y = []
    # U = []
    # V = []
    # # plt.label
    # # plt.figure(figsize=(400,300), dpi=90)

    
    # fig, ax = plt.subplots() 
    # plt.plot(x_obs, y_obs, ".k")
    # plt.xlim(0,400)
    # plt.ylim(0,300)
    # for i in range(len(current_node)):
    #     X.append(current_node[i][0])
    #     Y.append(current_node[i][1])
    #     U.append(parent_node[i][0])
    #     V.append(parent_node[i][1])
    
    # for i in range(len(X)):        
    #     q = ax.quiver(X[i], Y[i], U[i], V[i], units='xy', 
    #             scale=1, color='b')
    #     #  headwidth = 0.1, 
    #             # headlength=0, width = 0.5
    # for i in range(len(shortest_path)):
    #     plt.plot(shortest_path[i][0], shortest_path[i][1], ".r")
    #     # q = 
        
    # plt.show()
    cv2.imwrite("path.png", imgcopy)
    cv2.imshow("Visualization", imgcopy)

    # Creates shortest_path text file to store the optimal path
    file = open("shortest_path.txt", "w+")
    for i in shortest_path:
        file.write(str(i) + "\n")
        file.write("-------------" + "\n")
    file.close()
    # Creates backtrack_path text file to store the backtrack path taken
    file = open("backtrack_path.txt", "w+")
    for i in current_node:
        file.write(str(i) + "\n")
        file.write("-------------" + "\n")
    file.close()
    # End of main
    if cv2.waitKey(0) & 0xFF == ord('q'):
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
