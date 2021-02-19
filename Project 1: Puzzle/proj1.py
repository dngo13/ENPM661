#!/usr/bin/env python
# coding: utf-8

#  Diane Ngo
# ENPM 661 Project 1

import numpy as np

goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
# goal_state = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])
# node_state_i = np.array([[5, 2, 3], [1, 4, 6], [7, 0, 8]])
node_state_i = [5, 2, 3, 1, 4, 6, 7, 0, 8]
# Convert the matrix list to a 3x3 matrix
curr_node = np.reshape(node_state_i, (3, 3))
print("Goal\n", goal_state)
print("****")
print("Current\n", curr_node)


# Function to find the blank tile in the puzzle
def find_blank_tile(node):
    for i in range(0, 3):
        for j in range(0, 3):
            if curr_node[i][j] == 0:
                print("Value:", curr_node[i][j])
                r = i
                c = j
                print("Blank tile found at location:")
                print('(', r, ',', c, ")")
                print("****")
                return r, c


# Function to move the blank tile left
def move_left(node):
    row, col = find_blank_tile(node)
    print("Checking if moving left is possible")
    if col != 0:
        print("Moving left: ")
        old_node = np.copy(curr_node)
        curr_node[row][col] = old_node[row][col-1]
        curr_node[row][col-1] = 0
        print(curr_node)
        return curr_node
    else:
        print("Can't move left")
        curr_node[row][col] = curr_node[row][col]
        print(curr_node)
        return None


# Function to move the blank tile right
def move_right(node):
    row, col = find_blank_tile(node)
    print("Checking if moving right is possible")
    if col < 3:
        print("Moving right: ")
        old_node = np.copy(curr_node)
        curr_node[row][col] = old_node[row][col+1]
        curr_node[row][col+1] = 0
        print(curr_node)
        return curr_node
    else:
        print("Can't move right")
        curr_node[row][col] = curr_node[row][col]
        print(curr_node)
        return None


# Function to move the blank tile up
def move_up(node):
    row, col = find_blank_tile(node)
    print("Checking if moving up is possible")
    if row > 0:
        print("Moving up: ")
        old_node = np.copy(curr_node)
        curr_node[row][col] = old_node[row-1][col]
        curr_node[row-1][col] = 0
        print(curr_node)
        return curr_node
    else:
        print("Can't move up")
        curr_node[row][col] = curr_node[row][col]
        print(curr_node)
        return None


# Function to move the blank tile down
def move_down(node):
    row, col = find_blank_tile(node)
    print("Checking if moving down is possible")
    if row < 3:
        print("Moving down: ")
        old_node = np.copy(curr_node)
        curr_node[row][col] = old_node[row+1][col]
        curr_node[row+1][col] = 0
        print(curr_node)
        return curr_node
    else:
        print("Can't move down")
        curr_node[row][col] = curr_node[row][col]
        print(curr_node)
        return None


print("Blank tile", find_blank_tile(curr_node))
print("Move up?")
print(move_up(curr_node))
print("Move left?")
print(move_left(curr_node))
print("Move right?")
print(move_right(curr_node))
print("Move right?")
print("Move down?")
print(move_down(curr_node))

# print(move_right(curr_node))
# print("Move up?")
# print(move_up(curr_node))
# print("Move up?")
# print(move_up(curr_node))
# print("Move down?")
# print(move_down(curr_node))
# def at_goal():
