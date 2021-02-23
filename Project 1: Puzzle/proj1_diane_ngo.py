#!/usr/bin/env python
# coding: utf-8

#  Diane Ngo
# ENPM 661 Project 1

import numpy as np
import copy
import time

start_time = time.time()
# Defining the initial and goal state of the puzzle
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
goal_node = np.reshape(goal_state, (4, 4))
# The first node_state_i is case 5
node_state_i = [1, 6, 2, 3, 9, 5, 7, 4, 0, 10, 11, 8, 13, 14, 15, 12]
# The second node_state_i is for manually copying the test cases 1-5, must comment the first
# node_state_i = [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 7, 12, 13, 14, 11, 15]
curr_node = np.reshape(node_state_i, (4, 4))
# Test cases from rubric
# Test Case 1: [1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 7, 12, 13, 14, 11, 15] works 0.09s
# Test Case 2: [1, 0, 3, 4, 5, 2, 7, 8, 9, 6, 10, 11, 13, 14, 15, 12] works 2.124s
# Test Case 3: [0, 2, 3, 4, 1, 5, 7, 8, 9, 6, 11, 12, 13, 10, 14, 15] works 8.42s
# Test Case 4: [5, 1, 2, 3, 0, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12] works 148.75s
# Test Case 5: [1, 6, 2, 3, 9, 5, 7, 4, 0, 10, 11, 8, 13, 14, 15, 12] works, 3049s - 50minutes

print("Goal\n", goal_node)
print("****")
print("Current\n", curr_node)


# Function to find the blank tile in the puzzle
def find_blank_tile(temp_node):
    for i in range(4):
        for j in range(4):
            if temp_node[i][j] == 0:
                # print("Value:", curr_node[i][j])
                r = i
                c = j
                print("Blank tile found at location:")
                print('(',r, ',', c,")")
                # print("*"*50)
                return r, c


def print_tile(new_node):
    for i in range(4):
        for j in range(4):
            if new_node[i][j] == 0:
                # print("Value:", curr_node[i][j])
                r = i
                c = j
                print("New tile location: (", r, ',', c, ")")


# Function to convert list to string
def list2str(list_s):
    str_list = ''.join(str(e) for e in list_s)
    # print(str_list)
    return str_list


# Function to move the blank tile left
def move_left(node):
    row, col = find_blank_tile(node)
    print("Checking if moving left is possible")
    if col > 0:
        print("---Moving left: ")
        new_node = copy.deepcopy(node)  # Copies node to new variable for swapping location
        new_node[row][col], new_node[row][col-1] = new_node[row][col-1], new_node[row][col]  # swap location of 0
        print("Current node\n", new_node)
        print_tile(new_node)
        return new_node
    else:
        print("---Can't move left")
        return None


# Function to move the blank tile right
def move_right(node):
    row, col = find_blank_tile(node)
    print("Checking if moving right is possible")
    if col < 3:
        print("---Moving right: ")
        new_node = copy.deepcopy(node)   # Copies node to new variable for swapping location
        new_node[row][col], new_node[row][col+1] = new_node[row][col+1], new_node[row][col]  # swap location of 0
        print("Current node\n", new_node)
        print_tile(new_node)
        return new_node
    else:
        print("---Can't move right")
        return None


# Function to move the blank tile up
def move_up(node):
    row, col = find_blank_tile(node)
    print("Checking if moving up is possible")
    if row > 0:
        print("---Moving up: ")
        new_node = copy.deepcopy(node)  # Copies node to new variable for swapping location
        new_node[row][col], new_node[row-1][col] = new_node[row-1][col], new_node[row][col]  # swap location of 0
        print("Current node\n", new_node)
        print_tile(new_node)
        return new_node
    else:
        print("---Can't move up")
        return None


# Function to move the blank tile down
def move_down(node):
    row, col = find_blank_tile(node)
    print("Checking if moving down is possible")
    if row < 3:
        print("---Moving down: ")
        new_node = copy.deepcopy(node)   # Copies node to new variable for swapping location
        new_node[row][col], new_node[row+1][col] = new_node[row+1][col], new_node[row][col]  # swap location of 0
        print("Current node\n", new_node)
        print_tile(new_node)
        return new_node
    else:
        print("---Can't move down")
        return None


# Function to find children and action set of next nodes
def find_children(curr_node):
    print("--Find Children")
    children_list = []
    # print(child_node)
    check_up = move_up(curr_node)
    if check_up is not None:
        children_list.append(check_up)
    else:
        print("----Next node can't be up")
    check_down = move_down(curr_node)
    if check_down is not None:
        children_list.append(check_down)
    else:
        print("Next node can't be down")
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


# Function to convert the array to a list
def convert(mat):
    listname = list(np.array(mat).reshape(-1))
    # print(listname)
    return listname


# BFS Algorithm
def bfs_path(curr_node):
    # Creates empty lists
    visited = []
    visited_str = []
    queue = []
    print("========== BFS ============")
    # Adds current node to the queue
    queue.append(curr_node)
    count = 0
    # While the queue is not empty
    flag = True
    while len(queue) != 0 and flag:
        count += 1
        print("*" * 50)
        print("Taking first node of queue")
        node = queue.pop(0)
        print("Current Node\n", node)
        print("Place in queue")
        # Converts the array state to a list
        curr_list = convert(node)
        # Check if the state is at the goal
        print("Checking if at goal")
        solution = check_solution(curr_list, goal_state)
        if solution:
            print("Made it to goal")
            print(count, "steps")  # Print amount of steps taken
            print("--- %s seconds ---" % (time.time() - start_time))  # Print operation time
            break
        else:
            print("Not at goal")
        # Copy node information to temporary node variable for actions
        temp_node = copy.deepcopy(node)
        children_list = find_children(temp_node)
        # Checks for children and appends to queue
        for child in children_list:
            queue.append(child)
        print("Checking if visited")
        visited.append(temp_node)

        for vis in visited:
            vis_2 = list2str(convert(vis))
            # print("Converted:", vis_2)
            visited_str += [vis_2]
        # Commented out the print visited array since as the visited gets larger the program freezes
        # print("Visited array:", visited_str)
        check_visited(temp_node, visited_str)
            # break


        # Creates nodePath text file to store the path taken
        file = open("nodePath.txt", "w+")
        for node in visited:
            file.write(str(node) + "\n")
            file.write("-------------" + "\n")
        file.write(str(goal_node))
        file.close()


# Function to see if the current state is the goal state
def check_solution(curr, goal):
    # Converts the lists to a string
    curr = list2str(curr)
    goal = list2str(goal)
    if curr == goal:
        return True
    return False


# Function to see if the node has been visited
def check_visited(temp_node, visited_str):
    # print("Temp node", temp_node)
    # print("Visited\n", visited_str)
    # Converts the node to a list and then a string
    temp = convert(temp_node)
    temp = list2str(temp)
    # Loop to see if the temp node from the children is in the visited array
    if temp in visited_str:
        vis_bool = True
        print("Visited?", vis_bool)
    else:
        vis_bool = False
        print("Visited?", vis_bool)
    return vis_bool


bfs_path(curr_node)
