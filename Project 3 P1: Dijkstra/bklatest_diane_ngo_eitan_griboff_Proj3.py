#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:29:50 2021

@author: diane
"""

import numpy as np
import cv2
import math
import ast
from queue import PriorityQueue
from numpy import inf
# import copy
import time

start_time = time.time()
# Define workspace
ws_width = 400
ws_height = 300
clr = 5
img = np.zeros([ws_height, ws_width, 3], dtype=np.uint8)


# ===========TO DO ============= #
# Radius 10; clearance 5
# Dijkstra

class Node:
    # creating objects for position, cost and parent information
    def __init__(self, pos, cost, parent):
        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]
        self.cost = cost
        self.parent = parent


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


# elif (200- clr <= x <= 210 + clr and 230 - clr  <= y <= 280+ clr ) or \
    # (210 - clr <= x <= 230 + clr and 270 - clr <= y <= 280 + clr) \
    # or (210 - clr <= x <= 230 + clr and 230 - clr <= y <= 240 + clr):

# Function to check if the node is in an obstacle
def check_in_obstacle(node):
    x, y = node[0], node[1]
    # y = 300 - y
    # Inside circle
    if (x - 90) ** 2 + (y - 70) ** 2 <= (35+clr)**2:
        # print("Coordinate inside circle obstacle")
        return True
    # Ellipse
    elif ((x - 246-clr) ** 2) / 60 ** 2 + (((y - 145-clr) ** 2) / 30 ** 2) <= 1:
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
            (y + 1.4 * x - 176.64 - clr >= 0) and (y + 1.4 * x - 438.64 - clr <= 0):
        # print("Coordinate inside rectangle")
        return True
    else:
        # print("Coordinates are good to go")
        return False


# Function to check if current node is at goal node
def check_solution(curr_node, goal_node):
    if curr_node == goal_node:
        return True
    else:
        return False


# Function for start and goal coordinates by user input
# Note that the origin of the workspace is bottom lef2
def get_coords():
    start_x, start_y = [int(x) for x in input(
        "Enter starting x, y position: ").split()]
    while start_x > ws_width or start_y > ws_height:
        print("Start cannot be outside of workspace")
        start_x, start_y = [int(x) for x in input(
            "Enter starting x, y position: ").split()]
        start_y = img.shape[0] - start_y
    print("Start coordinates are not outside of workspace")

    start_node = (start_x, start_y)

    # While start positions are inside an obstacle ask for new input
    while check_in_obstacle(start_node):
        print("Start cannot be inside an obstacle")
        start_x, start_y = [int(x) for x in input(
            "Enter starting x, y position: ").split()]
        start_y = img.shape[0] - start_y
        start_node = (start_x, start_y)

    start_node = (start_x, start_y)

    # Inputs for goal
    goal_x, goal_y = [int(x) for x in input(
        "Enter end goal x, y position: ").split()]
    while goal_x > ws_width or goal_y > ws_height:
        print("Goal cannot be outside of workspace")
        goal_x, goal_y = [int(x) for x in input(
            "Enter end goal x, y position: ").split()]
        goal_y = img.shape[0] - goal_y
    print("Goal coordinates are not outside of workspace")

    goal_node = (goal_x, goal_y)

    # While goal positions are inside obstacles get new coordinates
    while check_in_obstacle(goal_node):
        print("Goal cannot be inside an obstacle")
        goal_x, goal_y = [int(x) for x in input(
            "Enter end goal x, y position: ").split()]
        # goal_node = (goal_x, goal_y)
        goal_y = img.shape[0] - goal_y
        goal_node = (goal_x, goal_y)
    return start_x, start_y, goal_x, goal_y, start_node, goal_node


# Function to determine action set of movements and cost
def move(node):
    i = node.x
    j = node.y

    possible_moves = [(i, j + 1), (i + 1, j), (i - 1, j), (i, j - 1),
                      (i + 1, j + 1), (i - 1, j - 1), (i - 1, j + 1),
                      (i + 1, j - 1)]
    possible_paths = []

    for pos, path in enumerate(possible_moves):
        if not (path[0] >= ws_height or path[0] < 0 or path[1] >= ws_width or path[1] < 0):
            # if check_in_obstacle(node):
            cost = math.sqrt(2) if pos > 3 else 1
            possible_paths.append([path, cost])
    return possible_paths


# Dijkstra algorithm function
def dijkstra(start_node, goal_node):
    print("======== DIJKSTRA ========")
    # Creates initial empty variables
    solvable = True
    parent = {}
    total_cost = {}
    visited = []
    q = PriorityQueue()
    for i in range(0, ws_width):
        for j in range(0, ws_height):
            # making the value of all the unvisited nodes as infinity
            total_cost[str([i, j])] = inf
    # Sets total cost of starting node to 0
    total_cost[str(start_node)] = 0
    # Adds start node to visited list
    visited.append(str(start_node))
    # Takes start node and sets it to parent node
    node = Node(start_node, 0, None)
    parent[str(node.pos)] = node
    q.put([node.cost, node.pos])  # adding the cost values in the queue
    if solvable:
        while q:
            curr_node = q.get()
            node = parent[str(curr_node[1])]
            if check_solution(curr_node[1], goal_node):
                print(">>At goal<<")
                # Print operation time
                print("--- %s seconds ---" % (time.time() - start_time))
                parent[str(goal_node)] = Node(goal_node, curr_node[0], node)
                break

            for next_node, cost in move(node):
                if not check_in_obstacle(next_node):
                    if next_node[0] < (ws_width-clr) and next_node[1] < (ws_height-clr):
                        if str(next_node) in visited:
                            curr_cost = cost + total_cost[str(node.pos)]
                            if curr_cost < total_cost[str(next_node)]:
                                total_cost[str(next_node)] = curr_cost
                                parent[str(next_node)].parent = node
                        else:
                            visited.append(str(next_node))
                            absolute_cost = cost + total_cost[str(node.pos)]
                            total_cost[str(next_node)] = absolute_cost
                            new_node = Node(next_node, absolute_cost,
                                            parent[str(node.pos)])
                            parent[str(next_node)] = new_node
                            q.put([absolute_cost, new_node.pos])

        goal_node = parent[str(goal_node)]
        parent_node = goal_node.parent
        backtracked = []
        while parent_node:  # This should work for backtracking purposes
            backtracked.append(parent_node.pos)
            print("Position:", parent_node.pos, " | Cost:", parent_node.cost)
            parent_node = parent_node.parent

        return backtracked, visited


def visual(path, img, visited, out):
    imgcopy = img.copy()
    visited = [ast.literal_eval(x.strip()) for x in visited]
    # Draws visited nodes
    for i in visited:
        # 300-
        if 300-i[1] > 299:
            val = i[1]
        else:
            val = 300-i[1]
        imgcopy[val, i[0]] = (255, 10, 0)
        out.write(imgcopy)
    # Draws the shortest path
    for i in path:
        # doesnt work with just i[1], i[0], path is flipped horizontally
        imgcopy[300-i[1], i[0]] = (200, 255, 0)
        out.write(imgcopy)

    # cv2.waitKey(1)
    # out.release()
    # cv2.destroyAllWindows()
    return imgcopy


# Main function
def main():
    # Create obstacles in map
    ws_map = create_obstacles()
    # Ask user for start and goal position
    start_x, start_y, goal_x, goal_y, start_node, goal_node = get_coords()
    # For test, show start and goal as green and red on the workspace

    # cv2.imshow("Map", ws_map)

    # Call dijkstra function and return the shortest path
    shortest_path, visited = dijkstra(start_node, goal_node)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Dijkstra.mp4', fourcc, 30, (400, 300), isColor=True)
    # Should now be able to use the shortest path to plot the path on the graph
    imgcopy = visual(shortest_path, ws_map, visited, out)
    cv2.circle(imgcopy, (start_x, 300 - start_y), radius=1,
               color=(0, 255, 0), thickness=3)
    cv2.circle(imgcopy, (goal_x, 300 - goal_y), radius=1,
               color=(0, 0, 255), thickness=3)
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
    for i in visited:
        file.write(str(i) + "\n")
        file.write("-------------" + "\n")
    file.close()
    # End of main
    if cv2.waitKey(0) & 0xFF == ord('q'):
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
