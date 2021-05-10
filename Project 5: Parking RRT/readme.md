Diane Ngo, Eitan Griboff 

ENPM 661 Project 5 RRT in a Parking Lot\
Group 6

Project 5 focuses on simulating autonomous parking using Turtlebot with Gazebo.
The program is written in Python 2.7. You will need OpenCV 4.2, Matplotlib, and ROS installed with its required packages in order to run this program.

Extract parking_search.zip to ~/catkin_ws/src/ \
In terminal: \
cd ~/catkin_ws \
catkin build \
source devel/setup.bash \
cd src/parking_search/scripts  \
chmod +x RRT_ros.py \
cd back to catkin_ws, cd ../../.. \
roslaunch parking_search navigation.launch \
New terminal: \
rosrun RRT_ros.py \\

Program will display an image for parking spaces. Press Q to move on. \
Program will then ask user for input of desired parking space. \
Input a number and press enter.\
Wait for the program to finish running. After seeing the images that are displayed, press Q until the image windows close. \
ROS will then start moving the robot in Gazebo.
