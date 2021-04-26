Diane Ngo, Eitan Griboff

ENPM 661 Project 3 Phase 3 TurtleBot with A*
Group 6

_________________Part 1_________________ \
This project is split into 2 parts, part 1 with OpenCV/Matplotlib showing the optimal path, and part 2 with ROS simulation in Gazebo.
The following are required for running the 1st part of the program: Python 3.7, OpenCV 4.14, and Matplotlib.


Run python prj3_grp6_a_star.py in terminal.


The program will prompt the user for clearance (default 5), starting x y position, starting theta, goal x y position, and RPM for both wheels.
Note that for this program, multiply start and goal positions by 100 because it is in centimeters.


Test case 1 uses clearance 5, starting position: 100 100, starting theta: 30, goal position: 900 900, 1 for both wheels. \
Note that the program takes very long to run, as it has to path plan, create a video in OpenCV, and draw the search in Matplotlib. \
In the directory where the file is ran, a video named Astar.mp4 shows the optimal path video. The blue areas are just examples of what the path has explored. \
The program will pop up a figure from matplotlib with the curves from A star and the optimal path in red.

Test case 1 video: https://drive.google.com/file/d/1Pdx3iKQsqN0YY06vfCU54j5zerm5UysD/view?usp=sharing

_________________Part 2_________________ \
For part 2 of the project, you need to have ROS Melodic, Gazebo, and matplotlib. \
Take astar_prj.zip and extract it to your catkin workspace. \
In terminal do cd ~/catkin_ws (or your workspace name), catkin build, and source devel/setup.bash \
Go to the scripts folder with terminal and type chmod +x a_star_ros.py, this makes the file executable. \
Open a new terminal and run "roslaunch astar_prj astar_world.launch"
Wait for Gazebo to launch with the TurtleBot, and open a 2nd terminal.
Type "rosrun astar_prj a_star_ros.py"
Case 1 the start position is (8, 5) to (7, 7)
Case 1 video: https://drive.google.com/file/d/1lNK65Vv3AeSqXwpeEdRFdRYJuuJAOUl0/view
