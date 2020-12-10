"""
This code was created by Iaroslav Kazakov, an Aeronautical student at Imperial College London on 9/11/2020
as part of his Final Year Project.
The data used to recreate a POD analysis is taken from a research paper and cannot be shared with public unless
author's permission is granted.
This code is the first goal set up during a meeting on 9/11/20, whereby classic and snapshot PODs need to
be reproduced for a given data set.
"""

# Import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import scipy.linalg as la
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Open and store the serealized byte stream as a numpy array
with open('flow_field_data0.pickle', 'rb') as file:
    X = pickle.load(file)

# Allocate the velocities from the data file to variables
"""
Note how the data is distributed it is in the form of (2000, 384, 192, 2), i.e. there are 2000 x 2 2D matrices 
where each of these 2000 x 2 matrices contain a velocity (either vertical or horizontal) vector at every grid point.
The grid is asymmetric, it has twice as many points in one direction that the other one.
"""

def velocities_upload(filename='flow_field_data0.pickle', comp=2):
    """

    filename: load the pickle file
    comp: indicate the number of velocity components
    return: two 3D ndarrays where the 0 axis corresponds to the temporal observation and
    axis 1,2 relate to the spatial observation on the grid
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    for i in range(comp):
        if i==0:
          U = data[:, :, :, i]
        elif i==1:
          V = data[:, :, :, i]
    return U,V

U,V = velocities_upload()

def velocity_dash_2D(U,V):
    """"
    1. specify two 3D ndarrays velocities from which the fluctuations have to be derived
    2. reshape the matrices such that the spatial points are stretched out in a single row, i.e. obtain a 2D matrix
    3. concatenate the matrices together
    """
    U_dash = (U-np.mean(U,axis=0)).reshape(U.shape[0], U.shape[1]*U.shape[2], order='F')
    V_dash =  (V-np.mean(V,axis=0)).reshape(V.shape[0], V.shape[1]*V.shape[2], order='F')
    Vel_dash = np.concatenate((U_dash,V_dash), axis=1)
    return Vel_dash

Vel_dash = velocity_dash_2D(U,V)

#U_bar = np.mean(U, axis=0)# Take a mean along the first axis, i.e. time
#V_bar = np.mean(V, axis=0)
#U_dash = U-U_bar
#V_dash = V-V_bar
#U_dash = U_dash.reshape(2000, 384*192, order='F') # Reorder the matrices such that each column is stretched out into a single row
#V_dash = V_dash.reshape(2000, 384*192, order='F')
#Vel_dash = np.concatenate((U_dash, V_dash), axis=1)


def eig(Vel_dash, points=250):
    Vel_dash_T = Vel_dash.transpose()
    Vel_dash_T = Vel_dash_T[:, :1999:points]
    R = np.dot(Vel_dash[:1999:points, :], Vel_dash_T)
    R = R.astype(int)
    eigvals, eigvecs = la.eig(R)
    indices = [i for i in range(eigvals.shape[0]) if eigvals[i] <= 20]
    eigval = np.delete(eigvals, np.where(eigvals <= 20))
    eigvec = np.delete(eigvecs, indices, 1)
    return eigval,eigvec


Vel_dash_T = Vel_dash.transpose()
Vel_dash_T = Vel_dash_T[:, :1999:250]
R = np.dot(Vel_dash[:1999:250, :], Vel_dash_T)
R=R.astype(int)
eigvals, eigvecs = la.eig(R)
indices = [i for i in range(eigvals.shape[0]) if eigvals[i]<=20]

eigval = np.delete(eigvals, np.where(eigvals <= 20))
eigvec = np.delete(eigvecs, indices, 1)
#eigval=np.zeros(eigvals.shape[0]-eigvals[eigvals<=20].shape[0])
#eigvec=np.zeros((eigvals.shape[0],eigvals.shape[0]-eigvals[eigvals<=20].shape[0]))



mod = np.dot(eigvec, np.diag(eigval.real**(-0.5)))
modes = np.dot(Vel_dash_T, mod)
a = np.array([0]*np.shape(modes)[1])

for i in range(np.shape(modes)[1]):
    a[i] = np.inner(Vel_dash_T[:, i], modes[:, i])
    modes[:, i] = a[i] * modes[:, i]
#Extract the necessary modes
Vel_dash_POD = modes.sum(axis=1)
Vel_dash1 = modes[:,0]
Vel_dash2 = modes[:,1]
Vel_dash3 = modes[:,2]
# Create the necessary U vector arrays
U_POD1 = np.array(Vel_dash1[:int(len(Vel_dash1)/2)]).reshape(384,192, order='F')
U_POD2 = np.array(Vel_dash2[:int(len(Vel_dash2)/2)]).reshape(384,192, order='F')
U_POD3 = np.array(Vel_dash3[:int(len(Vel_dash3)/2)]).reshape(384,192, order='F')

U_POD=np.array(Vel_dash_POD[:int(len(Vel_dash_POD)/2)])
#V_POD=np.array(Vel_dash_POD[int(len(Vel_dash_POD)/2):])
U_POD=U_POD.reshape(384,192, order='F')

#U_POD=U_POD.transpose()
#U_POD=U_POD.flatten()
#x_coordinate=[i/384 for i in range(384)]*192
#y_coordinate=[0 for i in range(384*192)]
#for i in range(192):
 #   for k in range(384):
  #     y_coordinate[k+i*384]=i/192

dy = 1/383
dx = 1/191
y, x = np.mgrid[slice(0, 1 + dy, dy), slice(0, 1 + dx, dx)]
U_100=U[100,:,:]

""" Full Reconstruction of the flow using Three modes separetly modes"""
cmap = plt.get_cmap('seismic')
levels1 = MaxNLocator(nbins=100).tick_values(U_POD1.min(), U_POD1.max())
levels2 = MaxNLocator(nbins=100).tick_values(U_POD2.min(), U_POD2.max())
levels3 = MaxNLocator(nbins=100).tick_values(U_POD3.min(), U_POD3.max())
levels4 = MaxNLocator(nbins=100).tick_values(U_100.min(), U_100.max())

norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)
norm3 = BoundaryNorm(levels3, ncolors=cmap.N, clip=True)
norm4 = BoundaryNorm(levels4, ncolors=cmap.N, clip=True)
fig1, ((ax0,ax1),(ax2,ax3))= plt.subplots(2,2)
cf0 = ax0.contourf(x, y, U_POD1, cmap=cmap,norm=norm1)
cf1 = ax1.contourf(x, y, U_POD2, cmap=cmap,norm=norm2)
cf2 = ax2.contourf(x, y, U_POD3, cmap=cmap,norm=norm3)
cf3 = ax3.contourf(x, y, U_100, cmap=cmap,norm=norm4)
fig1.colorbar(cf0, ax=ax0)
fig1.colorbar(cf1, ax=ax1)
fig1.colorbar(cf2, ax=ax2)
fig1.colorbar(cf3, ax=ax3)

ax0.set_title('Mode 1')
ax1.set_title('Mode 2')
ax2.set_title('Mode 3')
ax3.set_title('Instantaneous Snapshot')

plt.show()