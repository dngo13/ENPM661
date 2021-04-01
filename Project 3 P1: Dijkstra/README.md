Diane Ngo, Eitan Griboff
ENPM 661 Project 3 - Phase 1
Dijskstra

Requires:  
Python 3.7, OpenCV 4.1.0, NumPy
To run the file, open terminal or Spyder.
In terminal, run py diane_ngo_eitan_griboff_Proj3, or open the file in Spyder.

The program will prompt the user to input start and goal coordinates. The map dimensions is (400x300).
Note that the origin of the map is the bottom left corrner.
Input desired coordinates and the program will keep prompting if the points are inside  an obstacle.
If the points are far apart, the program will take a few minutes to run.
Once the program is done, it will show the final map with its exploration, optimal path, and start/goal points.
Press q to close it. This will then save the video, output files, and end the program.

In the directory:
Dijkstra.mp4        is the video recording of the exploration with optimal path.
path.png            is an image of the map after it has completed exploration with optimal path drawn on top.
exploration.txt     is the list of visited nodes during exploration.
shortest_path.txt   is the optimal nodes path.
