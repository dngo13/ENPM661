#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:33:15 2021

@author: diane
"""

import numpy as np
import cv2
import copy
import time


start_time = time.time()
# Define workspace
ws_width = 400
ws_height = 300
img = np.zeros([ws_height, ws_width, 3], dtype=np.uint8)


#  Ref: https://www.life2coding.com/draw-polygon-on-image-using-python-opencv/
def create_obstacles():
    # Circle
    cv2.circle(img, (90, 230), 35, (255, 255, 255), -1)
    # Ellipse
    cv2.ellipse(img, (246, 155), (60, 30), 0, 0, 360, (255, 255, 255), -1)
    # BL [48, 192], BR [171, 106], TL [37, 176], TR [160.90]
    # Rectangle
    rect_cnrs = np.array([[48, 192], [171, 106], [160, 90], [37, 176]], np.int32)
    cv2.fillPoly(img, [rect_cnrs], (255, 255, 255), 1)

    # C shape
    c_cnrs = np.array([[200, 20], [230, 20], [230, 30], [210, 30], [210, 60],
                       [230, 60], [230, 70], [200, 70]], np.int32)
    cv2.fillPoly(img, [c_cnrs], (255, 255, 255), 1)
    # Polygon
    poly_cnrs = np.array([[327, 236], [380, 183], [380, 128], [354, 161], [327, 156],
                          [285, 196]], np.int32)
    cv2.fillPoly(img, [poly_cnrs], (255, 255, 255), 1)
    return img

# Function to check if the node is in an obstacle
def check_in_obstacle(node):
    x, y = node[0], node[1]
    y = 300-y
    # Inside circle
    if (x-90)**2 + (y - 70)**2 < 35**2:
        print("Coordinate inside circle obstacle")
        return True
    # Ellipse
    elif ((x-246)**2)/60**2 + (((y-145)**2)/30**2) <=1:
        print("Coordinate inside ellipse obstacle")
        return True
    # C shape
    elif (x >= 200 and x <= 210 and y >=220 and y <= 280) or (x >= 210 and x<=230 and y>=270 and y<=280) \
        or (x >=210 and x<=230 and  y>=230 and y<=240):
        print("Coordinate inside C shaped obstacle")
        return True 
    # Angled rectangle
    elif (y - 0.7*x  >= 0) and (y - 0.7*x - 100 <= 0) and (y + 1.4*x - 176.64 >= 0) and (y + 1.4*x - 438.64 <= 0):
        print("Coordinate inside rectangle")
        return True
    # Polygon
    elif (((x-y-265<=0) and (x+y-391>=0) and (5*x+5*y-2351<=0) and (50*x-50*y-9007>=0)) 
          or ((5*x+5*y-2351>=0) and (703*x+2883*y-646716<0) and (x+y-492<=0) and (x-y-265<=0)) or 
          ((x+y-492>=0) and (x-y-265<=0) and (x<=381.03) and (1101*x-901*y-265416>0))):
        print("Coordinate inside polygon")
        return True
    else:
        print("Coordinates are good to go")
        return False

# Function to move left
def move_left(node):
    row, col = (node[0], node[1])
    # check_in_obstacle(node[0]-1, node[1], img)
    print("Checking if moving left is possible")
    if not check_in_obstacle(node):
        if row > 0:
            return (row-1, col)
    else:
        print("---Can't move left")
        return None

# Function to move up + left
def move_up_left(node):
    row, col = [node[0], node[1]]
    # check_in_obstacle(node[0]-1, node[1]+1, img)
    print("Checking if moving up and left is possible")
    if not check_in_obstacle(node):
        if col < 400 and row > 0:
            return (row-1, col+1)
    else:
        print("---Can't move up and left")
        return None


# Function to move up + left
def move_up_right(node):
    row, col = [node[0], node[1]]
    # and check_in_obstacle(node[0]+1, node[1]+1, img)
    print("Checking if moving up and right is possible")
    if not check_in_obstacle(node):
        if col < 300 and row < 400:
            return (row+1, col+1)
    else:
        print("---Can't move up and right")
        return None


# Function to move right
def move_right(node):
    row, col = [node[0], node[1]]
    # check_in_obstacle(node[0]+1, node[1], img)
    print("Checking if moving right is possible")
    if not check_in_obstacle(node):
        if row < 400:
            return (row+1, col)
    else:
        print("---Can't move right")
        return None


# Function to move up
def move_up(node):
    row, col = [node[0], node[1]]
    #  check_in_obstacle(row, col+1, img)
    print("Checking if moving up is possible")
    if not check_in_obstacle(node):
        if col < 300:
            return (row, col+1)
    else:
        print("---Can't move up")
        return None

# Function to move up + left
def move_down_left(node):
    row, col = [node[0], node[1]]
    # and check_in_obstacle(node[0]-1, node[1]-1, img)
    print("Checking if moving down and left is possible")
    if not check_in_obstacle(node):
        if col > 0 and row > 0:
            return (row-1, col-1)
    else:
        print("---Can't move down and left")
        return None


# Function to move up + left
def move_down_right(node):
    row, col = [node[0], node[1]]
    # and check_in_obstacle(node[0]+1, node[1]-1, img)
    print("Checking if moving down and right is possible")
    if not check_in_obstacle(node):
        if col > 0 and row < 400:
            return ( row+1, col-1)
    else:
        print("---Can't move down and right")
        return None

# Function to move down
def move_down(node):
    row, col = [node[0], node[1]]
    # and check_in_obstacle(node[0], node[1]-1, img)
    print("Checking if moving down is possible")
    if not check_in_obstacle(node):
        if col > 0:
            return (row, col-1)
    else:
        print("---Can't move down")
        return None


# Function to find children and action set of next nodes
def find_children(curr_node):
    print("--Find Children")
    children_list = []
    
    check_down_right = move_down_right(curr_node)
    if check_down_right is not None:
        children_list.append(check_down_right)
    else:
        print("----Next node can't be down and right")
    
    check_down_left = move_down_left(curr_node)
    if check_down_left is not None:
        children_list.append(check_down_left)
    else:
        print("----Next node can't be down and left")
    
    check_up_right = move_up_right(curr_node)
    if check_up_right is not None:
        children_list.append(check_up_right)
    else:
        print("----Next node can't be up and right")
    
    check_up_left = move_up_left(curr_node)
    if check_up_left is not None:
        children_list.append(check_up_left)
    else:
        print("----Next node can't be up and left")
        
    check_up = move_up(curr_node)
    if check_up is not None:
        children_list.append(check_up)
    else:
        print("----Next node can't be up")


    check_down = move_down(curr_node)
    if check_down is not None:
        children_list.append(check_down)
    else:
        print("----Next node can't be down")

    check_left = move_left(curr_node)
    if check_left is not None:
        children_list.append(check_left)
    else:
        print("----Next node can't be left")

    check_right = move_right(curr_node)
    if check_right is not None:
        children_list.append(check_right)
    else:
        print("----Next node can't be right")
    return children_list


# Function to see if the node has been visited
def check_visited(temp_node, visited):
    # Loop to see if the temp node from the children is in the visited array
    if temp_node in visited:
        vis_bool = True
        # print("Visited?", vis_bool)
    else:
        vis_bool = False
        # print("Visited?", vis_bool)
    return vis_bool


def check_solution(curr_node, goal_node):
    if curr_node == goal_node:
        return True
    else:
        return False

    
# BFS Algorithm
def main():
    
    start_x, start_y = [int(x) for x in input("Enter starting x, y position: ").split()]
    while start_x > ws_width or start_y > ws_height:
        print("Start cannot be outside of workspace")
        start_x, start_y = [int(x) for x in input("Enter starting x, y position: ").split()]
    print("start coordinates are not outside of workspace")
    start_y = img.shape[0] - start_y
   
    curr_node = (start_x, start_y)

    while check_in_obstacle(curr_node):
        print("Start cannot be inside an obstacle")
        start_x, start_y = [int(x) for x in input("Enter starting x, y position: ").split()]
        start_y = img.shape[0] - start_y
        curr_node = (start_x, start_y)
    start_node = (start_x, start_y)

    goal_x, goal_y = [int(x) for x in input("Enter end goal x, y position: ").split()]

    while goal_x > ws_width or goal_y > ws_height:
        print("Goal cannot be outside of workspace")
        goal_x, goal_y = [int(x) for x in input("Enter end goal x, y position: ").split()]
    goal_y = img.shape[0] - goal_y
    goal_node = (goal_x, goal_y)
    
    while check_in_obstacle(goal_node):
        print("Goal cannot be inside an obstacle")
        goal_x, goal_y = [int(x) for x in input("Enter end goal x, y position: ").split()]
        goal_y = img.shape[0] - goal_y
        goal_node = (goal_x, goal_y)
  
# =============================================================================a
    vis = create_obstacles()  
    
    visited = []
    queue = []
    parent_map = {}
    parent_map[curr_node] = None
    print("========== BFS ============")
    # Adds current node to the queue
    queue.append(curr_node)
    count = 0
    # While the queue is not empty
    flag = True
    final = None
    while len(queue) != 0 and flag:
        count += 1
        print("*" * 50)
        print("Taking first node of queue")
        node = queue.pop(0)
        print("Current Node\n", node)
        print("Place in queue")

        # Check if the state is at the goal
        print("Checking if at goal")
        solution = check_solution(node, goal_node)
        if solution:
            final = node
            print("Made it to goal")
            print(count, "steps")  # Print amount of steps taken
            print("--- %s seconds ---" % (time.time() - start_time))  # Print operation time
            print("Start was: ", start_x, ",", start_y)
            print("Goal was: ", goal_x, ",", goal_y)
            backtrack(parent_map, final)
            visual(visited, vis, parent_map, start_node)
            break
        else:
            print("Not at goal")
        # Copy node information to temporary node variable for actions
        temp_node = copy.deepcopy(node)
        children_list = find_children(temp_node)
        # Checks for children and appends to queue
        for child in children_list:        
            if check_visited(child, visited):
                continue
            if check_visited(child,queue):
                continue
            else:
                queue.append(child)
                parent_map[child] = temp_node
                # print(child[1], child[0])
            # img[child[1], child[0]] = [255, 20, 0]
        
        
        print("Checking if visited")
        visited.append(temp_node)
        vis = cv2.circle(vis, (start_x, start_y), radius=1, color=(0, 255, 0), thickness=3)
        vis = cv2.circle(vis, (goal_x, goal_y), radius=1, color=(0, 0, 255), thickness=2) 
# =============================================================================
    
def backtrack(parent_map, final):
    ###### BACKTRACKING ######
    parent = parent_map[final]
    path_list = []
    while parent is not None:
        parent = parent_map[parent]
        path_list.append(parent)
        print("Parent")
        print(parent)
    print("Path List")
    print(path_list)
    # print("Queue")
    # print(queue)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('BFS.avi', fourcc, float(120), (img.shape[1], img.shape[0]))
def visual(visited, vis, parent_map, start_node):
    try:
        for item in visited:
            # print(item[1], item[0])
            vis[item[1], item[0]] = (255, 200, 0)
            out.write(vis)
    except:
        pass
    cv2.circle(vis, (start_node[0], start_node[1]), radius=1, color=(0, 255, 0), thickness=2)
    cv2.imshow("Image", vis)
    cv2.waitKey(0)
    out.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
