#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag

class Tracker(): # class for Kalman Filter based tracker

    def __init__(self):

        #streetNaN=structures.Street(40,[0,0,0,0])
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id
        self.box = [] # list to store the coordinates for a bounding box
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)
        self.cls = ""
        self.distance = 0
        self.det_coord = []
	self.centroids=[]
	self.frameNum=[]
	self.colorProb = [0,0,255]
        colorList=[[255,0,0],[0,255,0],[0,0,255],[255,255,255],[255,0,255],[51,255,255],[255,153,255],[178,255,102],[255,128,0]]
        self.box_color=colorList[np.random.randint(0,9)]
        self.x_state=[]
        self.dt = 1.   # time interval

	# Process matrix, assuming constant velocity model
	self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
		           [0, 1,  0,  0,  0,  0,  0, 0],
		           [0, 0,  1,  self.dt, 0,  0,  0, 0],
		           [0, 0,  0,  1,  0,  0,  0, 0],
		           [0, 0,  0,  0,  1,  self.dt, 0, 0],
		           [0, 0,  0,  0,  0,  1,  0, 0],
		           [0, 0,  0,  0,  0,  0,  1, self.dt],
		           [0, 0,  0,  0,  0,  0,  0,  1]])

	# Measurement matrix, assuming we can only measure the coordinates

	self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
		       [0, 0, 1, 0, 0, 0, 0, 0],
		       [0, 0, 0, 0, 1, 0, 0, 0],
		       [0, 0, 0, 0, 0, 0, 1, 0]])


	# Initialize the state covariance
	self.L = 100.0
	self.P = np.diag(self.L*np.ones(8))


	# Initialize the process covariance
	self.Q_comp_mat = np.array([[self.dt**4/2., self.dt**3/2.],
				[self.dt**3/2., self.dt**2]])
	self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
			self.Q_comp_mat, self.Q_comp_mat)

	# Initialize the measurement covariance
	self.R_ratio = 1.0/16.0
	self.R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
	self.R = np.diag(self.R_diag_array)
        

    def update_R(self):
        R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)


    def kalman_filter(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int) # convert to integer coordinates
                                     #(pixel values)

    def predict_only(self):
        '''
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)


if __name__ == "__main__":
  
    # Creat an instance
    trk = Tracker()
    
