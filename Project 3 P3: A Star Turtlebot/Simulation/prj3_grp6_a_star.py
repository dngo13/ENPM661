#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
from queue import PriorityQueue
import time
import matplotlib.pyplot as plt
import itertools


start_time = time.time()
# Define workspace
ws_width = 1000
ws_height = 1000
# Euclidean distance
euc = 0.5
# Theta threshold
theta_thresh = 30
img = np.zeros([ws_height, ws_width, 3], dtype=np.uint8)
radius = 17.7
global RPM_L, RPM_R


# i being parent coordinate and UL/UR children
# Uses differential drive constraints
# Function to plot the curves on matplot

def plot_curve(Xi, Yi, Thetai, UL, UR):
    t = 0
    r = 3.3  # 17.7
    L = 8.9  # 35.4
    dt = 0.1
    Xn = Xi
    Yn = Yi
    Thetan = np.pi * Thetai / 180
    # Xi, Yi,Thetai: Input point's coordinates
    # Xs, Ys: Start point coordinates for plot function
    # Xn, Yn, Thetan: End point coordintes
    D = 0
    while t < 1:
        t = t + dt
        Xs = Xn
        Ys = Yn
        Xn += 0.5*r * (UL + UR) * math.cos(Thetan) * dt
        Yn += 0.5*r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        plt.plot([Xs, Xn], [Ys, Yn], color="blue")
        # plt.pause(0.05)

    # Xn = round(Xn)
    Thetan = 180 * (Thetan) / 3.14
    return Xn, Yn, Thetan, D


# Function to create obstacles in the workspace using OpenCV
def create_obstacles():
    # Bottom Left Circle
    cv2.circle(img, (200, 1000 - 200), 100, (255, 255, 255), -1)
    # Top Left Circle
    cv2.circle(img, (200, 1000 - 800), 100, (255, 255, 255), -1)
    # Bottom left Rectangle
    cv2.rectangle(img, (25, 1000 - 575),
                  (175, 1000 - 425), (255, 255, 255), -1)
    # Middle Rectangle
    cv2.rectangle(img, (375, 1000 - 575),
                  (625, 1000 - 425), (255, 255, 255), -1)
    # Bottom Right Rectangle
    cv2.rectangle(img, (725, 1000-400), (875, 1000-200), (255, 255, 255), -1)
    return img


# Function to check if the node is in an obstacle
# Defaultclearance is 5 but user defined
def check_in_obstacle(node, clr=5):
    # print(clr)
    x, y = node[0], node[1]
    # y = 300 - y
    # Inside circle
    dist = clr + radius
    # Bottom Left Circle
    if (x - 200) ** 2 + (y - 200) ** 2 <= (100 + dist) ** 2:
        # print("Coordinate inside circle obstacle")
        return True
    # Bottom Right Circle
    elif ((x - 200) ** 2) + (y - 800) ** 2 <= (100 + dist) ** 2:
        # print("Coordinate inside ellipse obstacle")
        return True
    # Left Rectangle
    elif 25-dist <= x <= 175+dist and 425-dist <= y <= 575+dist:
        # print("Coordinate inside C shaped obstacle")
        return True
    # Middle rectangle
    elif 375-dist <= x <= 625+dist and 425-dist <= y <= 575+dist:
        # print("Coordinate inside C shaped obstacle")
        return True
    # Right Rectangle
    elif 725-dist <= x <= 875+dist and 200-dist <= y <= 400+dist:
        # print("Coordinate inside C shaped obstacle")
        return True
    else:
        # print("Coordinates are good to go")
        return False


# Function to check if current node is at goal node
def check_solution(curr_node, goal_node):
    if curr_node[0] == goal_node[0] and curr_node[1] == goal_node[1]:
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
        goal_y = img.shape[0] - goal_y
        goal_node = (goal_x, goal_y, goal_th)
    # goal_node = (goal_x, goal_y, goal_th)
    return start_x, start_y, start_th, goal_x, goal_y, goal_th, start_node, goal_node


# Function for action steps
def action_step(RPM_L, RPM_R):
    actions = [[0, RPM_L], [RPM_R, 0], [RPM_L, RPM_L],  [RPM_L, RPM_R],
               [RPM_R, RPM_L], [RPM_R, RPM_R], [0, RPM_R], [RPM_L, 0],
               [0, -RPM_L], [-RPM_R, 0], [-RPM_L, -RPM_L], [0, -RPM_R],
               [-RPM_L, 0], [-RPM_R, -RPM_R], [-RPM_L, -RPM_R], [-RPM_R, -RPM_L]]

    return actions


# Function for determining next position
def find_next(RPM_L, RPM_R, curr_node):
    # print("Current node", curr_node)
    t = 0
    r = 3.3  # 0.038
    L = 8.9  # 0.354
    dt = 0.1
    Xn = float(curr_node[1][0])
    Yn = float(curr_node[1][1])
    Thetan = 3.14 * curr_node[1][2] / 180

    UL = RPM_L
    UR = RPM_R
    while t < 1:
        t = t + dt
        Xn += 0.5 * r * (UL + UR) * math.cos(Thetan) * dt
        Yn += 0.5 * r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        # D = D+ math.sqrt(math.pow((0.5*r * (UL + UR) * math.cos(Thetan) * dt), 2)+math.pow((0.5*r * (UL + UR) * math.sin(Thetan) * dt),2))
    Thetan = 180 * (Thetan) / 3.14
    next_node = [Xn, Yn, Thetan]
    return next_node


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
def a_star(start_node, goal_node, actions, RPM_L, RPM_R):
    print("======== A* ========")
    # Creates initial empty variables
    solvable = True
    path = {}
    child_node = []
    current_node = []
    parent_node = []
    visited = np.zeros(
        [int(1000 / euc), int(1000 / euc), int(360 / theta_thresh)])
    q = PriorityQueue()
    # Start and goal vectors
    start = (0, start_node, None)
    goal = (0, goal_node, None)
    start_node = discretize(
        start_node[0], start_node[1], start_node[2], theta_thresh, euc)
    goal_node = discretize(
        goal_node[0], goal_node[1], goal_node[2], theta_thresh, euc)

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

            # moves, step_size = action_step(curr_node, 1, theta_thresh)

            for action in actions:
                moves = find_next(action[0], action[1], curr_node)
                angle = moves[2] + curr_node[1][2]
                if angle < 0:
                    angle = angle + 360
                elif angle > 360:
                    angle = angle - 360
                elif angle == 360:
                    angle = 0

                node = (curr_node[1][0] + action[0],
                        curr_node[1][1] + action[1], angle)
                node = discretize(node[0], node[1], node[2], theta_thresh, euc)
                node_cost = curr_node[0] + \
                    moves[2] + heuristic(node, goal[1])
                node_parent = curr_node[1]

                if not check_in_obstacle(node):
                    # print("Checking if visited and obstacle")
                    if (ws_width - clr) > node[0] > 0 and (ws_height - clr) > node[1] > 0:
                        if visited[int(node[0] / euc)][int(node[1] / euc)][int(node[2] / theta_thresh)] == 0:
                            visited[int(node[0] / euc)][int(node[1] / euc)
                                                        ][int(node[2] / theta_thresh)] = 1
                            parent_node1 = (
                                node_parent[0], node_parent[1], angle)
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
                        print("Looping for parent")
                        # Set temp to get the parent
                        temp = path.get(parent)
                        # Loop through to find backtrack nodes
                        if parent[1] == '.' and parent[5] == '.':
                            param1 = float(parent[0]) + float(parent[2]) / 10
                            param2 = float(parent[4]) + float(parent[6]) / 10
                        if parent[2] == '.' and parent[7] == '.':
                            param1 = float(
                                parent[0] + parent[1]) + float(parent[3]) / 10
                            param2 = float(
                                parent[5] + parent[6]) + float(parent[8]) / 10
                        if parent[1] == '.' and parent[6] == '.':
                            param1 = float(parent[0]) + float(parent[2]) / 10
                            param2 = float(
                                parent[4] + parent[5]) + float(parent[7]) / 10
                        if parent[2] == '.' and parent[6] == '.':
                            param1 = float(
                                parent[0] + parent[1]) + float(parent[3]) / 10
                            param2 = float(parent[5]) + float(parent[7]) / 10
                        if parent[3] == '.' and parent[9] == '.':
                            param1 = float(
                                parent[0] + parent[1] + parent[2]) + float(parent[4]) / 10
                            param2 = float(
                                parent[6] + parent[7] + parent[8]) + float(parent[10]) / 10
                        if parent[3] == '.' and parent[7] == '.':
                            param1 = float(
                                parent[0] + parent[1] + parent[2]) + float(parent[4]) / 10
                            param2 = float(parent[6]) + float(parent[8]) / 10
                        if parent[3] == '.' and parent[8] == '.':
                            param1 = float(
                                parent[0] + parent[1] + parent[2]) + float(parent[4]) / 10
                            param2 = float(
                                parent[6] + parent[7]) + float(parent[9]) / 10
                        if parent[1] == '.' and parent[7] == '.':
                            param1 = float(parent[0]) + float(parent[2]) / 10
                            param2 = float(
                                parent[4] + parent[5] + parent[6]) + float(parent[8]) / 10
                        if parent[2] == '.' and parent[8] == '.':
                            param1 = float(
                                parent[0] + parent[1]) + float(parent[3]) / 10
                            param2 = float(
                                parent[5] + parent[6] + parent[7]) + float(parent[9]) / 10
                        backtracked.append((param1, param2, curr_node[2][2]))
                        parent = temp
                        # If the parameters are the start node, can terrminate loop
                        if (param1, param2) == (start_node[0], start_node[1]):
                            break
                    backtracked.append(
                        (start_node[0], start_node[1], start_node[2]))
                    return backtracked, current_node, parent_node, visited


# Function for visualizing the path and search
def visual(path, img, current_node, parent, out):
    imgcopy = img.copy()
    # Set search color to blue
    for i in range(len(current_node) - 1):
        imgcopy[1000 - int(current_node[i][1]),
                int(current_node[i][0]) - 1] = [255, 0, 0]
        # cv2.circle(imgcopy, (299 - int(current_node[i][1]), int(current_node[i][0]) - 1), 3, (255, 255, 0))
        out.write(imgcopy)
    # Draws the optimal path
    for j in range(len(path) - 1):
        imgcopy = cv2.line(imgcopy, (int(path[j][0]) - 1, 1000 - int(path[j][1])),
                           (int(path[j + 1][0]) - 1, 1000 - int(path[j + 1][1])), (255, 255, 255), 2)
        out.write(imgcopy)
    # Saves to video and returns the frame
    return imgcopy


# Main function
def main():
    # Define global variables and user inputs
    global clr  # clearance
    clr = int(input("Define the clearance distance between objects:"))
    # Create obstacles in map
    ws_map = create_obstacles()
    # Ask user for start and goal position
    start_x, start_y, start_th, goal_x, goal_y, goal_th, start_node, goal_node = get_coords()
    # RPM_L = 1
    # RPM_R = 1
    RPM_L, RPM_R = [int(x) for x in input(
        "Enter RPM for left and right wheels: ").split()]
    actions = action_step(RPM_L, RPM_R)
    # print(actions[-16])
    # Call A star function and return the shortest path
    shortest_path, current_node, parent_node, visited = a_star(
        start_node, goal_node, actions, RPM_L, RPM_R)
    # print(current_node)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Astar.mp4', fourcc, 4000,
                          (1000, 1000), isColor=True)
    # Visualize the search and optimal path
    imgcopy = visual(shortest_path, ws_map, current_node, parent_node, out)

    cv2.circle(imgcopy, (start_x, 1000 - start_y), radius=1,
               color=(0, 255, 0), thickness=3)
    cv2.circle(imgcopy, (goal_x, 1000 - goal_y), radius=1,
               color=(0, 0, 255), thickness=3)
    for i in range(1000):
        out.write(imgcopy)
    # Plot
    fig, ax = plt.subplots(dpi=500)
    # plt.figure(figsize=(1000, 1000))
    plt.grid()
    ax.set_aspect('equal')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.title('Map', fontsize=10)
    # print(shortest_path)

    for i in range(len(shortest_path)):
        if i % 16 == 0:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[0][0], actions[0][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[0][0], actions[0][1])
        elif i % 16 == 1:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[1][0], actions[1][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[1][0], actions[1][1])
        elif i % 16 == 2:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[2][0], actions[2][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[2][0], actions[2][1])
        elif i % 16 == 3:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[3][0], actions[3][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[3][0], actions[3][1])
        elif i % 16 == 4:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[4][0], actions[4][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[4][0], actions[4][1])
        elif i % 16 == 5:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[5][0], actions[5][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[5][0], actions[5][1])
        elif i % 16 == 6:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[6][0], actions[6][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[6][0], actions[6][1])
        elif i % 16 == 7:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[7][0], actions[7][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[7][0], actions[7][1])
        elif i % 16 == 8:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[8][0], actions[8][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[8][0], actions[8][1])
        elif i % 16 == 9:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[9][0], actions[9][1])
            X2 = plot_curve(X1[0], X1[1], X1[2], actions[9][0], actions[9][1])
        elif i % 16 == 10:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[10][0], actions[10][1])
            X2 = plot_curve(X1[0], X1[1], X1[2],
                            actions[10][0], actions[10][1])
        elif i % 16 == 11:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[11][0], actions[11][1])
            X2 = plot_curve(X1[0], X1[1], X1[2],
                            actions[11][0], actions[11][1])
        elif i % 16 == 12:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[12][0], actions[12][1])
            X2 = plot_curve(X1[0], X1[1], X1[2],
                            actions[12][0], actions[12][1])
        elif i % 16 == 13:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[13][0], actions[13][1])
            X2 = plot_curve(X1[0], X1[1], X1[2],
                            actions[13][0], actions[13][1])
        elif i % 16 == 14:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[14][0], actions[14][1])
            X2 = plot_curve(X1[0], X1[1], X1[2],
                            actions[14][0], actions[14][1])
        elif i % 16 == 15:
            X1 = plot_curve(shortest_path[i][0], shortest_path[i][1],
                            shortest_path[1][2], actions[15][0], actions[15][1])
            X2 = plot_curve(X1[0], X1[1], X1[2],
                            actions[15][0], actions[15][1])
        # plt.pause(0.05)
    for i in range(len(shortest_path)-1):
        plt.plot([shortest_path[i][0], shortest_path[i+1][0]],
                 [shortest_path[i][1], shortest_path[i+1][1]], color='red')

    plt.show()
    plt.close()

    cv2.imwrite("path.png", imgcopy)
    cv2.imshow("Visualization", imgcopy)

    # # Creates shortest_path text file to store the optimal path
    file = open("shortest_path.txt", "w+")
    for i in shortest_path:
        file.write(str(i) + "\n")
        file.write("-------------" + "\n")
    file.close()
    # # Creates backtrack_path text file to store the backtrack path taken
    # file = open("backtrack_path.txt", "w+")
    # for i in current_node:
    #     file.write(str(i) + "\n")
    #     file.write("-------------" + "\n")
    # file.close()
    # End of main
    if cv2.waitKey(0) & 0xFF == ord('q'):
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
