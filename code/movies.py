import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.animation as animation

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

def show_lidar(t):
    _, lidar_data = read_data_from_csv('../Data/sensor_data/lidar.csv')

    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    ranges = lidar_data[t, :]	#First sweep
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, ranges)
    ax.set_rmax(80)
    ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')

    plt.show()
    
# CREATING STEREO IMAGE MOVIES
def makeStereoMovie():
    image_folder = '../Data/stereo_images/stereo_right'
    video_name = 'rightCameraMovie.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        
    cv2.destroyAllWindows()
    video.release()

def _videoMap(self,final=False): # save map into list when called, then render video of list on final call
    
    mapRender = np.where(self.map['map']>0,1,0) + np.where(self.map['map']<0,-1,0)
    self.map['video'].append(mapRender)
    
    if final == True:
        
        if input('render video? [Y/n]: ') not in ['Y','y']: return
        print('rendering video...')
        snapshots = self.map['video']

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure( figsize=(8,8) )

        im = plt.imshow(snapshots[0],cmap='binary',origin='lower')

        def animate_func(i):
            im.set_data(snapshots[i])
            return [im]

        anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func, 
                                    frames=len(snapshots),
                                    interval = 1, # in ms
                                    )

        anim.save('videoMap.mp4', fps=int(24*5000/self.intervalVid), extra_args=['-vcodec', 'libx264'])
        print('video complete!')