import numpy as np
import pointcloud as pc
import pdb

N = 1000
a = np.random.rand(3,N)
a[0,:] = a[0,:]*10-5
a[1,:] = a[1,:]*2-1
a[2,:] = a[2,:]*5-2.5
P1 = pc.PointCloud(p=a)
P1.signature()

# spherical ball
sphere = np.array([[np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(theta)]
                   for theta in np.linspace(0.02, np.pi, 51)
                   for phi in np.linspace(0.02, 2*np.pi, 51)])

P2 = pc.PointCloud(p=sphere.T)
P2.signature()
