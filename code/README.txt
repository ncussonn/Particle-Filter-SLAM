PROJECT 2 - CODE README - Nathan Cusson-Nadeau

############################
####	REQUIREMENTS	####
############################

numpy
pandas
matplotlib
cv2
os

###########
# SCRIPTS #
###########

Mapping.py 	- main script, running this file independently completes entire project
pr2_utils.py 	- provided file that defines multiple useful functions for project
movies.py	- contains a series of defined functions to help visualize data

####################################################################################
####################################################################################

movies.py	- series of functions used to create videos

##### movies.py functions #####

read_data_from_csv(filename)	- provided function which reads meaurement data from a csv file into numpy array

INPUT 
filename        - file address

OUTPUT 
timestamp       - timestamp of each observation
data            - a numpy array containing a sensor measurement in each row

####################################################################################

show_lidar(t)	- plots one full scan of LiDAR sensor sweep at time t
		- used to visualize lidar scan data

INPUT 
t       - valid timestamp index (references row number in numpy array)
