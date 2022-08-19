import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
import random as rand
import cv2 as cv2

############################################
###         DEFINING FUNCTIONS           ###
############################################

# TIK-TOK Functions used to time certain parts
def tic():
	return time.time()
def toc(tstart, name="Operation"):
	print('%s took: %s sec.\n' % (name,(time.time() - tstart)))
# Used to get data from csv files
def read_data_from_csv(filename):
	'''
	INPUT 
	filename        file address

	OUTPUT 
	timestamp       timestamp of each observation
	data            a numpy array containing a sensor measurement in each row
	'''
	data_csv = pd.read_csv(filename, header=None)
	data = data_csv.values[:, 1:]
	timestamp = data_csv.values[:, 0]
	return timestamp, data
# Ray Tracing for Lidar Scan
def bresenham2D(sx, sy, ex, ey):
	'''
	Bresenham's ray tracing algorithm in 2D.
	Inputs:
		(sx, sy)	start point of ray
		(ex, ey)	end point of ray
	'''
	sx = int(round(sx))
	sy = int(round(sy))
	ex = int(round(ex))
	ey = int(round(ey))
	dx = abs(ex-sx)
	dy = abs(ey-sy)
	steep = abs(dy)>abs(dx)
	if steep:
		dx,dy = dy,dx # swap 

	if dy == 0:
		q = np.zeros((dx+1,1))
	else:
		q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
	if steep:
		if sy <= ey:
			y = np.arange(sy,ey+1)
		else:
			y = np.arange(sy,ey-1,-1)
		if sx <= ex:
			x = sx + np.cumsum(q)
		else:
			x = sx - np.cumsum(q)
	else:
		if sx <= ex:
			x = np.arange(sx,ex+1)
		else:
			x = np.arange(sx,ex-1,-1)
		if sy <= ey:
			y = sy + np.cumsum(q)
		else:
			y = sy - np.cumsum(q)
	return np.vstack((x,y))
# Correlation for current particle
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
	'''
	INPUT 
	im              the map 
	x_im,y_im       physical x,y positions of the grid map cells
	vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
	xs,ys           physical x,y,positions you want to evaluate "correlation" 

	OUTPUT 
	c               sum of the cell values of all the positions hit by range sensor
	'''
	nx = im.shape[0]
	ny = im.shape[1]
	xmin = x_im[0]
	xmax = x_im[-1]
	xresolution = (xmax-xmin)/(nx-1)
	ymin = y_im[0]
	ymax = y_im[-1]
	yresolution = (ymax-ymin)/(ny-1)
	nxs = xs.size
	nys = ys.size
	cpr = np.zeros((nxs, nys))
	for jy in range(0,nys):
		y1 = vp[1,:] + ys[jy] # 1 x 1076
		iy = np.int16(np.round((y1-ymin)/yresolution))
		for jx in range(0,nxs):
			x1 = vp[0,:] + xs[jx] # 1 x 1076
			ix = np.int16(np.round((x1-xmin)/xresolution))
			valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
															np.logical_and((ix >=0), (ix < nx)))
			cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
	return cpr
# Returns disparity and RGB image from left camera at current timeStamp
def compute_stereo(timeStamp):
	'''
	INPUT 
	timeStamp		current time stamp of the system

	OUTPUT 
	disparity       distance between cameras
	image_l			the left stereo cameras image in RGB pixel values
	'''

	path_l = '../Data/data/stereo_images/stereo_left/'+ str(timeStamp) + '.png'
	path_r = '../Data/data/stereo_images/stereo_right/'+ str(timeStamp) + '.png'

	image_l = cv2.imread(path_l, 0)
	image_r = cv2.imread(path_r, 0)

	# RGB Pixels
	image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
	image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

	image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
	image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

	stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) # may need to fine tune the disparities and blocksize for more accuracy
	disparity = stereo.compute(image_l_gray, image_r_gray)

	'''
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
	ax1.imshow(image_l)
	ax1.set_title('Left Image')
	ax2.imshow(image_r)
	ax2.set_title('Right Image')
	ax3.imshow(disparity, cmap='gray')
	ax3.set_title('Disparity Map')
	plt.show()
	'''
	return disparity, image_l

tsetup = tic()

############################################
############################################
#####            Parameters            #####
############################################
############################################

############################################
### Initializing particle set parameters ###
############################################

# Particle Count
N = 3 # Make 300 for final implementation
Nthresh = round(0.6*N) # threshold for resampling, 60% rounded to nearest integer value
alpha_thresh = 0.05 # threshold weight for when particle should be deleted
print("Number of particles:\n", (N))

# Initializing Particles - (x,y,theta)' each column corresponds to different particle
''' assumes particle count in static, in final implementation this would need to change '''
mu = np.zeros((3,N))
# Weights
alpha = np.full(N,1/N)
# Correlation Vector for Update Step
correlation = np.zeros(N)

# Assign Random Initial Locations and angles to partcle set with a 2D gaussian distribution
for i in range(0,N):
	mu[:,i] = [rand.gauss(0, 150),rand.gauss(0,150),rand.randrange(0,np.ceil(np.pi))]

# Time steps skipped until updating models, used to improve runtime
stepSize = 30
laserSkip = 2
# Loop Iteration Counter
n = 0

# Initial Occupancy Map 
MAP = {}

MAP['res']   = 1 #meters
MAP['xmin']  = -300  #meters
MAP['ymin']  = -1300
MAP['xmax']  =  1600
MAP['ymax']  =  500

MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

# Initialize an occupancy map all occupied
MAP['map'] = np.full((MAP['sizex'],MAP['sizey']),1,dtype=np.int8) #DATA TYPE: char or int8

# redefine dictionary terms as variables for code simplicity
m = MAP['map']
r = MAP['res']

x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

# 9 x 9 resoloution range around particle
x_range = np.arange(-4*r,4*r+r,r)
y_range = np.arange(-4*r,4*r+r,r)

tt = tic()

################################
### Log-Odds Hyperparameters ###
################################

# log-odds constant
logOdd = np.log(9)
# Overconfidence constraints
logOddsMin = -3*logOdd
logOddsMax = 3*logOdd

############################
#### Encoder Parameters ####
############################

# Encoder parameters taken from parameter file
ticks = 4096
d_l = 0.623479
d_r = 0.622806

# Meters traveled per tick for left and right rear-wheel
mL_tick = (np.pi * d_l) / ticks
mR_tick = (np.pi * d_r) / ticks

########################
### Lidar Parameters ###
########################

#Transformations from Lidar to body frame

# Rotation
bRl = np.array([[0.00130201, 0.796097, 0.605167], [0.999999, -0.000419027, -0.00160026], [-0.00102038, 0.605169, -0.796097]])
# Translation
pl = np.array([[0.8349], [-0.0126869], [1.76416]])

#####################################
### Left Stereo Camera Parameters ###
#####################################

# Rotation
bRs = np.array([[-0.00680499, -0.0153215, 0.99985], [-0.999977, 0.000334627, -0.00680066], [-0.000230383, -0.999883, -0.0153234]])      
# Translation
ps = np.array([1.64239, 0.247401, 1.58411])

print("Loading Data")
# Obtaining required data from FOG and encoders
ts = tic()
timeStampLidar, lidarData = read_data_from_csv("../Data/sensor_data/lidar.csv")
timeStampEncoder, encoderData = read_data_from_csv("../Data/sensor_data/encoder.csv")
timeStampFOG, fogData = read_data_from_csv("../Data/sensor_data/fog.csv")
toc(ts,"Data Load")

##################################
####  Motion Model Parameters ####
##################################

predictionSteps = len(timeStampLidar)
state = np.zeros((predictionSteps,3))
theta = 0
mean = 0

toc(tsetup,"Initializing Variables")

########################################
########################################
#######      PARTICLE FILTER     #######
########################################
########################################

print("Starting Particle Filter")
ts = tic()
tloop = tic()

''' delete if able to get rid of y'''
y = np.full((N,MAP['sizex'],MAP['sizey']),1)

for t in range(0,int(predictionSteps*3/4),stepSize):
	########################################
	####       CREATING RANGE SCAN      ####
	########################################

	# Initialize observations
	ranges = lidarData[t]
	angles = np.linspace(-5, 185, 286) / 180 * np.pi # converted to radians

	# Only take valid indices (omit too far or too close)
	# > 80 means no information, < 0.1 risks including scanning part of the vehicle
	indValid = np.logical_and((ranges < 80),(ranges > 0.1))
	ranges = ranges[indValid]
	angles = angles[indValid]

	###################################################
	####  GENERATING ROBOT'S POSE IN WORLD FRAME   ####         
	###################################################

	# Convert ranges to cartesian coordinates in lidar frame
	xs0 = ranges*np.cos(angles)
	ys0 = ranges*np.sin(angles)
	zl = np.zeros(len(ranges))
	rl = np.array([xs0,ys0,zl])

	# convert from meters to cells - used later to create map
	xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
	yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

	# Transformation from lidar to body frame
	bRl_mult = np.dot(bRl,rl)
	lidar2body = bRl_mult+pl

	###################################
	####  OBSERVATIONS AT STEP K   ####         
	###################################
	
	if t == 0:
		'''Skip, cannot predict without a prior'''
		# Body to world is identity
		wRb = np.zeros((3,3))
		np.fill_diagonal(wRb,1)

	else:
		#################
		#### ENCODER ####
		#################

		# Tick Count since last observation
		zL = encoderData[t,0] - encoderData[t-stepSize,0]
		zR = encoderData[t,1] - encoderData[t-stepSize,1]

		# Time Elapsed since last encoder observation (converted from nanoseconds -> seconds)
		tau = (timeStampEncoder[t]-timeStampEncoder[t-stepSize])*10**-9

		# Meters Traveled Since Last Encoder Observation
		mL = mL_tick*zL
		mR = mR_tick*zR

		# Instantaneous Velocity
		vL = mL / tau
		vR = mR / tau
		v = (vR + vL)/2 # Average Velocity

		#############
		#### FOG ####
		#############
		'''
		Nested for loop here is pretty bad for computation time, but cannot think of other way
		to accurately predict angle. Could consider skipping steps and using prior measurement as
		approximation?
		'''
		# Use every 10th update of FOG in prediction - *need every measurement for accurate angle
		for j in range((t-stepSize)*10,t*10):
			
			# Time Elapsed since Last FOG measurement (converted from nanoseconds -> seconds)
			tauFOG = (timeStampFOG[j+1]-timeStampFOG[j])*10**-9

			# Angular Velocity (yaw)
			delYaw = fogData[j,2] # angle moved about z since last fog measurement
			omegaDot = delYaw/tauFOG # rad/sec

			# Current Angle of Robot in radians
			theta = theta + delYaw

		# Rotation matrix from body to world frame
		wRb = np.array([[np.cos(theta),-np.sin(theta),0],
						[np.sin(theta),np.cos(theta),0],
						[0,0,1]]) # purely about z (yaw) rotation	

	intScanCoord = []
	scanKeyParticle = []

	for k in range(0,N):

		########################################
		########################################
		#######       UPDATE STEP        #######
		####### USING LASER CORRELATION  #######
		#######       	 MODEL		     #######
		########################################
		########################################
		''' 
		Use laser scan from each particle to compute map
		correlation then update particle weights alpha.
		'''
		# Updating Translation Using Observation from Lidar
		# translation from body to world is particle's current estimated body frame position
		wpb = np.array([[mu[0,k]],[mu[1,k]],[0]]) #z component is 0 because assume body frame does not change elevation during journey
		wRb_mult = np.dot(wRb,lidar2body)

		# Transformation from body to world pose of particle's lidar scan
		lidar2world = wRb_mult+wpb # x,y,z coordinates of lidar scan end points in world frame

		# Body frame coordinates of particle in world frame
		sx = mu[0,k] # start point of laser in x
		sy = mu[1,k] # start point of laser in y

		########################################
		########################################
		####### 						 #######
		#######	Converting Scan to Cells #######
		#######     					 #######
		########################################
		########################################

		##########################################################
		## Converting Scan to Cells Using bresenham2D function  ##
		##########################################################

		# Occupancy Cells Based on Lidar Scan
		# Returns a 2D array of cell coordinates and associated key which denotes which cells
		# are occupied or unoccupied for current timestamps lidar scan in relation to
		# each particle's world location

		for f in range(0,len(rl[0]),laserSkip):
			# Range Coordinates for single laser in world coordinates
			ex = lidar2world[0,f] # range in x
			ey = lidar2world[1,f] # range in y
			
			# trace ray's coordinates represent 1x1m cells
			rayCoord = bresenham2D(sx,sy,ex,ey)

			# Keep key that records which coordinates are empty or occupied
			# Because only valid rays were kept, the last element corresponds to occupied cell
			# and is assigned a binary 1
			newRayKey = np.full(len(rayCoord[0]),-1)
			newRayKey[-1] = 1

			# Range Coordinates and occupancy for all lasers
			if f == 0:
				scanCoord = rayCoord
				scanKey = newRayKey
			else:
				# Returns empty cell coordinates and obstacle coordinates
				scanCoord = np.concatenate((scanCoord,rayCoord), axis = 1) # world frame coodinates in meters and integers
				scanKey = np.concatenate((scanKey,newRayKey)) # binary array of 1's 0's

		# Convert Coordinates to integers so they can be used for indices later
		intScanCoord.append(scanCoord.astype(int))
		scanKeyParticle.append(scanKey)

		###################################
		## Checking Correlation of Scan  ##
		## 		 for k-th Particle 		 ##
		###################################

		Y = np.vstack((xs0,ys0))
		''' this doesn't work'''
		# Correlation of each cell 
		c = mapCorrelation(m,x_im,y_im,Y,x_range,y_range)

		# Find cell(s) that gives max 
		coordinateMax = np.where(c == np.max(c))

		# save max correlation
		#correlation[k] = np.max(c)
		
	'''for testing, force first particle to always have highest correlation'''
	correlation = np.array([1,0,0])

	########################################
	########################################
	#######     RESAMPLING STEP      #######
	########################################
	########################################
	''' 
	Rescales particles and remove those with weights below threshold
	Was unable to complete due to time constraints.
	'''
	# rescaling particles using their correlation
	a_times_c = alpha*correlation
	alpha = a_times_c / np.sum(a_times_c)

	# filter particles with weights below threshold
	deleted_index = np.where(alpha < alpha_thresh)
	alpha[deleted_index] = 0
	
	# effective number of particles after adjusting weights
	Neff = 1/ np.sum(alpha**2)
	
	# index of highest weighted particle
	alphaMax_index = np.where(alpha == np.max(alpha))[0][0]

	# if the effective number of particles is below or equal to threshold, reintroduce particles
	if Neff <= Nthresh:
		'''
		Particles should be spawned around more than just the current leader, but leader should receive proportionally more?
		unsure of how to implement but will add if time allows
		'''
		# for each deleted particle, assign new current highest correlated particle state
		# then displace new particles as a guassian distribution around copied particle
		for i in range(0,len(deleted_index[0])):

			# Guassian distribution parameters (must be different for each particle - keep inside for loop)
			std_x = mu[0,alphaMax_index]*0.1
			std_y = mu[1,alphaMax_index]*0.1
			std_theta = mu[2,alphaMax_index]*0.1

			# assign highest correlated particle state to new particles, then distribute as gaussian with std of 10% around mean
			displacement = np.array([rand.gauss(0,0.1*std_x), rand.gauss(0, std_y), rand.gauss(0,std_theta)])

			mu[:,deleted_index[0][i]] = mu[:,alphaMax_index] + displacement

	############################
	## Updating Map Log-odds  ##
	############################
	
	newMapLength = len(scanKeyParticle[alphaMax_index])

	# Generate map of initially 0's, with indices corresponding to each cell from ray
	logOddsMap = np.zeros(newMapLength)

	for k in range(0,N):
		# Log-Odds of Cells
		for z in range(0,newMapLength):
			# If an occupied cell, add value, otherwise subtract
			if scanKeyParticle[alphaMax_index][z] == 1 and logOddsMap[z] < logOddsMax:
				logOddsMap[z] = logOddsMap[z] + np.log(9)
			elif scanKeyParticle[alphaMax_index][z] == -1 and logOddsMap[z] > logOddsMin:
				logOddsMap[z] = logOddsMap[z] - np.log(9)
			else:
				print("Error: ray element not 1 or -1")

	''' This is super slow and very brute force-y'''
	''' Should remove this part if can get mapCorrelation to work'''
	# Using cell log-odds to assign values to map of highest weight particle
	for g in range(0,newMapLength):
		# Assign binary values to map
		if logOddsMap[g] > 0:
			# Occupied Cell
			y[alphaMax_index,intScanCoord[alphaMax_index][0, g] - MAP['xmin'],intScanCoord[alphaMax_index][1, g]- MAP['ymin']] = 1
		else:
			# Unoccupied Cell
			y[alphaMax_index,intScanCoord[alphaMax_index][0, g] - MAP['xmin'],intScanCoord[alphaMax_index][1, g]- MAP['ymin']] = -1
	
	########################################
	########################################
	#######       TEXTURED MAP       #######
	########################################
	########################################
	''' 
	Was unable to complete due to time constraints.

	Use RGBD images from the largest-weight particle's pose
	to assign colors to the occupancy grid cells
	'''
	# convert image to RGB and get disparity, d
	'''
	d, RGB_pixel = compute_stereo(int(timeStampLidar[k]))

	INSERT PSEUDOCODE OF HOW THIS WOULD BE DONE
	'''
	for k in range(0,N):
		if t == 0:
			'''Skip prediction - no time has elapsed to get measurements'''
		else:
			########################################
			########################################
			#######      PREDICTION STEP     #######
			####### USING DIFFERENTIAL-DRIVE #######
			#######          MODEL           #######
			########################################
			########################################
			''' 
			Using diff drive model with data from
			encoders and FOG, predict motion of 
			each particle.
			'''			
			#########################################
			#### DIFFERENTIAL DRIVE MOTION MODEL ####
			#########################################

			# Gaussian motion noise Constants - redefine for every particle at each timestep
			sigma_velocity = v*0.05 # .5 percent of velocity
			sigma_angle = omegaDot*0.005 # 0.05 percent of angle
			gaussian_position = rand.gauss(mean,sigma_velocity)
			gaussian_theta = rand.gauss(mean,sigma_angle)

			# prediction motion noise
			w = np.array([gaussian_position,gaussian_position,gaussian_theta])
			
			# Apply Motion Model with Gaussian distributed motion noise to hypotheses
			mu[:,k] = mu[:,k] + tau*np.array([v*np.cos(theta),v*np.sin(theta),omegaDot]) + w

	# Keep Track of time for testing runtime
	
	if t == stepSize:
		toc(tloop,"1 step")
	elif t == stepSize*1000:
		toc(tloop,"30,000 Timestamps")
	elif t == stepSize*2*1000:
		toc(tloop, "60,000 Timestamps")
	elif t == stepSize*4*1000:
		toc(tloop,"120,000 Timestamps")

	# dead-reckoning using first particle trajectory
	# 
	state[n] = mu[:,0]
	
	# Iteration Count
	n = n + 1

toc(ts,"Particle Filter")

##########################################
####  CONSTRUCTING 2D OCCUPANCY MAP   ####
##########################################

# take the best correlated particle's map through time and use it to create map
# pretty sure this is redundant and could have just used y
m[np.where(y[alphaMax_index]==-1)] = -1 # assign -1 to free pixels

fig1 = plt.figure()
finalMap = m.reshape(np.shape(MAP['map']))
# Plot the dead-reckoning trajectory
plt.scatter(state[:,1] - MAP['ymin'],state[:,0] - MAP['xmin'],s=1) # state y, state x
plt.imshow(finalMap,cmap='binary')

#########################################
####    GENERATING TRAJECTORY MAP    ####
#########################################

##################
# DEAD-RECKONING #
##################

fig2 = plt.figure()

# remove extra 0,0 coordinates for plotting
state[np.where(state == 0)] = np.NaN
plt.scatter(state[:,1],state[:,0],s=1) # state y, state x

toc(tt,"Total Runtime")