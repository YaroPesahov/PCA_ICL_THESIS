"""
This code was created by Iaroslav Kazakov, an Aeronautical student at Imperial College London on 9/11/2020
as part of his Final Year Project.
The data used to recreate a POD analysis is taken from a research paper and cannot be shared with public unless
author's permission is granted.
This code is the first goal set up during a meeting on 9/11/20, whereby classic and snapshot PODs need to
be reproduced for a given data set.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import scipy.linalg as la
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
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
        #elif i==2:
          #W = data[:, :, :, i]
    return U,V


U,V = velocities_upload()


def velocity_dash_2D(U,V):
    """
    1. specify two 3D ndarrays velocities from which the fluctuations have to be derived
    2. reshape the matrices such that the spatial points are stretched out in a single row, i.e. obtain a 2D matrix
    3. concatenate the matrices together
    """
    U_dash = (U-np.mean(U, axis=0)).reshape(U.shape[0], U.shape[1]*U.shape[2], order='F')
    V_dash = (V-np.mean(V, axis=0)).reshape(V.shape[0], V.shape[1]*V.shape[2], order='F')
    Vel_dash = np.concatenate((U_dash, V_dash), axis=1)
    return Vel_dash


Vel_dash = velocity_dash_2D(U,V)


def eig(Vel_dash, points=250, cutoff=20):
    """
    Nested function that calculates eigenvalues and eigenvectors
    ASSUMPTION: The observed flow is NOT periodic, otherwise choose the values corresponding to one full period only
    :param Vel_dash: Input concatenated velocity fluctuations
    :param points: define the step points
    :param cutoff: define the threshold for eigenvalue cutoff point
    :return: eigenvalue, eigenvector and Vel_dash_Transpose
    """
    def sub_eig(V = Vel_dash, step = points):
      V_T = V.transpose()[:, :1999:step]
      R = np.dot(V[:1999:step, :], V_T)
      eigvals, eigvecs = la.eig(R)
      return eigvals, eigvecs, V_T

    eigvals, eigvecs, Vel_dash_T = sub_eig()
    indices = [i for i in range(eigvals.shape[0]) if eigvals[i] <= cutoff]
    eigval = np.delete(eigvals, np.where(eigvals <= cutoff))
    eigvec = np.delete(eigvecs, indices, 1)
    return eigval, eigvec, Vel_dash_T


eigval,eigvec,Vel_dash_T=eig(Vel_dash,points=100)


def all_modes(eigval,eigvec,V=Vel_dash_T):
       mod = np.dot(eigvec, np.diag(eigval.real**(-0.5)))
       modes = np.dot(V, mod)
       def rec_modes(modes=modes,V= V):
           a = np.array([0]*np.shape(modes)[1], dtype=float)
           for i in range(np.shape(modes)[1]):
               a[i] = np.inner(V[:, i], modes[:, i])
               modes[:, i] = a[i] * modes[:, i]
           return modes, a
       recreated_modes, a_coeff = rec_modes()
       return modes, recreated_modes, a_coeff


modes, recreated_modes, a_coeff = all_modes(eigval,eigvec)

def grid (dy = 1 / 383,dx = 1 / 191):
    y, x = np.mgrid[slice(0, 1 + dy, dy), slice(0, 1 + dx, dx)]
    return y,x


y,x=grid()

def recreation_plot(recreated_modes=recreated_modes):
    U = recreated_modes.sum(axis=1)
    U_POD = np.array(U[:int(len(U) / 2)])
    U_POD = U_POD.reshape(384, 192, order='F')
    return U_POD

### PLotting section for Recreated flow
U_full = recreation_plot()
cmap = plt.get_cmap('seismic')
levels11 = MaxNLocator(nbins=20).tick_values(U_full.min(), U_full.max())
norm11 = BoundaryNorm(levels11, ncolors=cmap.N, clip=True)
fig11, ax11 = plt.subplots()
cf11 = ax11.contourf(y, x, U_full, cmap=cmap, norm=norm11)
fig11.colorbar(cf11, ax=ax11)
ax11.set_title('Recreated Normalised Flow')
plt.show()
### End of the section

#def plotting(number=3, modes=modes):
 #   for i in range(number):
  #      Vel=modes[:,i]
   #     U_POD = np.array(Vel[:int(len(Vel)/2)]).reshape(384,192, order='F')
    #    cmap = plt.get_cmap('seismic')
     #   fig, ax = plt.subplots(2, 2)
      #  levels = MaxNLocator(nbins=100).tick_values(U_POD.min(), U_POD.max())
       # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
       # cf0 = ax0.contourf(y, x, U_POD1, cmap=cmap, norm=norm)
       # fig.colorbar(cf0, ax=ax0)
       # ax0.set_title('Mode 1')
     #return plt.show()


Vel_dash1 = modes[:,0]
Vel_dash2 = modes[:,1]
Vel_dash3 = modes[:,2]
# Create the necessary U vector arrays
U_POD1 = np.array(Vel_dash1[:int(len(Vel_dash1)/2)]).reshape(384,192, order='F')
U_POD2 = np.array(Vel_dash2[:int(len(Vel_dash2)/2)]).reshape(384,192, order='F')
U_POD3 = np.array(Vel_dash3[:int(len(Vel_dash3)/2)]).reshape(384,192, order='F')
U_100=U[100,:,:]

""" Full Reconstruction of the flow using Three modes"""
cmap = plt.get_cmap('seismic')
#levels1 = MaxNLocator(nbins=20).tick_values(U_POD.min(), U_POD.max())
levels1 = MaxNLocator(nbins=100).tick_values(U_POD1.min(), U_POD1.max())
levels2 = MaxNLocator(nbins=100).tick_values(U_POD2.min(), U_POD2.max())
levels3 = MaxNLocator(nbins=100).tick_values(U_POD3.min(), U_POD3.max())
levels4 = MaxNLocator(nbins=100).tick_values(U_100.min(), U_100.max())

#norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)
norm3 = BoundaryNorm(levels3, ncolors=cmap.N, clip=True)
norm4 = BoundaryNorm(levels4, ncolors=cmap.N, clip=True)
fig1, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
#fig1, ax0= plt.subplots()
#cf0 = ax0.contourf(y, x, U_POD, cmap=cmap, norm=norm1)
cf0 = ax0.contourf(y, x, U_POD1, cmap=cmap,norm=norm1)
cf1 = ax1.contourf(y, x, U_POD2, cmap=cmap,norm=norm2)
cf2 = ax2.contourf(y, x, U_POD3, cmap=cmap,norm=norm3)
cf3 = ax3.contourf(y, x, U_100, cmap=cmap,norm=norm4)
fig1.colorbar(cf0, ax=ax0)
fig1.colorbar(cf1, ax=ax1)
fig1.colorbar(cf2, ax=ax2)
fig1.colorbar(cf3, ax=ax3)
#fig1.colorbar(cf0, ax=ax0)
ax0.set_title('Mode 1')
ax1.set_title('Mode 2')
ax2.set_title('Mode 3')
ax3.set_title('Instantaneous Snapshot')

plt.show()
