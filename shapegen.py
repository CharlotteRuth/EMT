#!/usr/bin/env python3
"""Shape generation and triangulation tools

This module allows the user to generate a number of different types of
triangulated shapes, including spheres, rectangular prisms, and cylinders. These
tools are intended to be used with Grinnell's PHY-335 Computational Exercise 3,
which covers computationally calculating Gauss's law over different shapes.

All coordinates are expected in (x, y, z) form, and all objects are centered on
the origin.

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

class Tile:
    """A single face of a 3D object

    Parameters
    ----------
    c1 : coordinate 1
    c2 : coordinate 2
    c3 : coordinate 3

    Attributes
    ----------
    area : float
        The area of the Tile
    centroid : tuple
        The coordinates of the centroid of the Tile
    """
    def __init__(self, c1, c2, c3):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def get_centroid(self):
        """Calculate centroid point of triangle with averaging method"""
        """
        x1, y1, z1 = self.c1
        x2, y2, z2 = self.c2
        x3, y3, z3 = self.c3
        centroid = ((x1+x2+x3)/3, (y1+y2+y3)/3, (z1+z2+z3)/3)
        """
        
        centroid = (np.array(self.c1)+np.array(self.c2)+np.array(self.c3))/3

        return centroid

    def get_area(self):
        """Calculate area of Tile"""
        
        
        #Area vector of the Tile
        a = 0.5*np.cross(np.subtract(self.c3,self.c1),np.subtract(self.c2,self.c1))   # get directed area

        #Make sure direction of vector points out
        if (a*self.get_centroid()).sum() < 0:   # Uses algorithmically simpler area*centroid to calculate the dot product but
            a = -a
        
        """
        np.where((a*self.get_centroid()).sum() < 0,-a,a) # reverse the area if you get a minus sign on dot product.  
                            # Here we use the algorithmically simpler area*centroid to calculate the dot product
                            # Results in wrong flux calculation for prism and cylinder because of np.where(). Not sure why  
        """
        return a

    def get_coords(self):
        return (self.c1, self.c2, self.c3)

class Shape:
    """
    A general shape, holds base functions of all other Shape objects.
    """
    def __init__(self):
        pass

    def get_centroids(self):
        """Returns an ordered list of centroids"""
        tr = self.get_triangles()

        x = []
        for n in tr:
            x.append(n.get_centroid())

        return x

    def get_areas(self):
        """Returns an ordered list of areas of the triangles.

        Parameters
        ----------
        point : tuple
            The origin point that the area vectors should point away from.
            Usually the  location of q_enc

        Returns
        -------
        list
            Description of returned object.

        """

        tr = self.get_triangles()

        x = []
        for n in tr:
            x.append(n.get_area())

        return x

    def get_triangles(self):
        """Returns all of the triangles for processing"""
        tr = self.tri.triangles

        x = []
        for n in tr:
            c1 = (self.x[n[0]], self.y[n[0]], self.z[n[0]])
            c2 = (self.x[n[1]], self.y[n[1]], self.z[n[1]])
            c3 = (self.x[n[2]], self.y[n[2]], self.z[n[2]])
            x.append(Tile(c1, c2, c3))

        return x

    def get_coords(self):
        """Returns coordinates of the vertices in every triangle in the Shape"""
        tr = self.get_triangles()

        x = []
        for n in tr:
            x.append(n.get_coords())

        return np.array(x)

    def plot(self, ax, c, color):
        """Handles coloring for plotting a Shape object.

        Parameters
        ----------
        ax : Axes object
            The Axis passed from the Shape subclass.
        c : list
            List of triangulated faces, in order
        color : list
            List of colors for each Tile, in order.

        Returns
        -------
        Axes object
            The axis that the Shape is plotted on.

        """
        #Set color mapping of plot
        if color is not None:
            cmax = np.max(color)
            # cmin = np.min(color[np.where(color > cmax*.05)])
            # cmin = np.min(color)
            cmin = 0
            for i in range(len(c)):
                # print("i: ", i)
                # print("len: ", len(c[i].get_array()))
                if i == 0:
                    start_idx = 0
                    end_idx = len(c[i].get_array())
                else:
                    start_idx = end_idx
                    end_idx = start_idx + len(c[i].get_array())
                # print("idx", start_idx, end_idx)
                c[i].set_array(color[start_idx:end_idx])
                c[i].set_clim(vmin=cmin, vmax=cmax)
                # print(c[i].get_array())

        return ax

class Sphere(Shape):
    """A triangularized sphere object.

    Parameters
    ----------
    r : float
        The radius of the sphere.
    res : int
        The resolution used in generating the sphere.

    Attributes
    ----------
    tri : Triangulation
        The parameterized Triangulation of the sphere.
    x : list
        The x-coordinates of of the sphere.
    y : list
        The y-coordinates of of the sphere.
    z : list
        The z-coordinates of of the sphere.
    len : int
        Number of triangles in the instance
    """
    def __init__(self, r, res=16):
        # Make a mesh in the space of parameterisation variables u and v
        u = np.linspace(0, 2.0 * np.pi, res)
        v = np.linspace(0, np.pi, res)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()

        # Triangulate parameter space to determine the triangles
        self.tri = mtri.Triangulation(u, v)

        self.x = r * np.sin(v) * np.cos(u)
        self.y = r * np.sin(v) * np.sin(u)
        self.z = r * np.cos(v)

        self.len = len(self.get_triangles())


    def plot(self, color=None, cmap="plasma"):
        """Generates a trisurface of the sphere and returns that for plotting


        Returns
        -------
        ax
            Axis with sphere plotted.

        """
        fig = plt.figure(figsize=plt.figaspect(1)*4)  # Square figure
        ax = fig.add_subplot(111, projection='3d')

        c = ax.plot_trisurf(self.x, self.y, self.z, triangles=self.tri.triangles,
                        alpha=.7, cmap="plasma")

        return super().plot(ax, [c], color)

class Prism(Shape):
    """A triangularized rectangular prism object.

    Parameters
    ----------
    corner : tuple
        The coordinate of the corner in the all-positive octant.
    res : int
        The resolution used in generating the prism.

    Attributes
    ----------
    l : float
        Description of attribute `l`.
    w : float
        Description of attribute `w`.
    h : float
        Description of attribute `h`.
    x : list
        X-coordinates for top & bottom
    y : list
        Y-coordinates for top & bottom
    z : list
        Z-coordinates for top & bottom
    tri : Triangulation
        Parameterized triangulation of x-y plane
    x2 : list
        X-coordinates for left & right (I think)
    y2 : list
        Y-coordinates for left & right (I think)
    z2 : list
        Z-coordinates for left & right (I think)
    x3 : list
        X-coordinates for front & back (I think)
    y3 : list
        X-coordinates for front & back (I think)
    z3 : list
        X-coordinates for front & back (I think)
    len : int
        Number of triangles in the instance
    """
    def __init__(self, corner, res=8):
        self.l = corner[0]
        self.w = corner[1]
        self.h = corner[2]

        x = np.linspace(-self.l, self.l, res)
        y = np.linspace(-self.w, self.w, res)
        X, Y = np.meshgrid(x, y)
        Z = (self.h) * np.ones((res, res))

        self.x, self.y, self.z = X.flatten(), Y.flatten(), Z.flatten()

        self.tri = mtri.Triangulation(self.x, self.y)

        self.x2 = self.x
        self.y2 = self.y*self.h/self.w
        self.z2 = self.z*self.w/self.h

        self.x3 = self.x*self.w/self.l
        self.y3 = self.y*self.h/self.w
        self.z3 = self.z*self.l/self.h

        self.len = len(self.get_triangles())

    def plot(self, color=None, cmap="plasma"):
        """Plots a trisurface of the prism.

        Returns
        -------
        ax object
            axis with prism plotted

        """
        fig = plt.figure(figsize=plt.figaspect(1)*4)  # Square figure
        ax = fig.add_subplot(111, projection='3d')

        c = []
        c.append(ax.plot_trisurf(self.x, self.y, self.z, triangles=self.tri.triangles, alpha=.7, shade=False, cmap=cmap))
        c.append(ax.plot_trisurf(self.x, self.y, -self.z, triangles=self.tri.triangles, alpha=.7, shade=False, cmap=cmap))
        c.append(ax.plot_trisurf(self.x2, self.z2, self.y2, triangles=self.tri.triangles, alpha=.7, shade=False, cmap=cmap))
        c.append(ax.plot_trisurf(self.x2, -self.z2, self.y2, triangles=self.tri.triangles, alpha=.7, shade=False, cmap=cmap))
        c.append(ax.plot_trisurf(self.z3, self.x3, self.y3, triangles=self.tri.triangles, alpha=.7, shade=False, cmap=cmap))
        c.append(ax.plot_trisurf(-self.z3, self.x3, self.y3, triangles=self.tri.triangles, alpha=.7, shade=False, cmap=cmap))

        return super().plot(ax, c, color)

    def get_triangles(self):
        """Returns all of the triangles for processing.

        Returns
        -------
        list
            List of all triangles in the prism.

        """
        tr = self.tri.triangles

        x = []
        for n in tr:
            c1 = (self.x[n[0]], self.y[n[0]], self.z[n[0]])
            c2 = (self.x[n[1]], self.y[n[1]], self.z[n[1]])
            c3 = (self.x[n[2]], self.y[n[2]], self.z[n[2]])
            x.append(Tile(c1, c2, c3))

        for n in tr:
            c1 = (self.x[n[0]], self.y[n[0]], -self.z[n[0]])
            c2 = (self.x[n[1]], self.y[n[1]], -self.z[n[1]])
            c3 = (self.x[n[2]], self.y[n[2]], -self.z[n[2]])
            x.append(Tile(c1, c2, c3))

        for n in tr:
            c1 = (self.x2[n[0]], self.z2[n[0]], self.y2[n[0]])
            c2 = (self.x2[n[1]], self.z2[n[1]], self.y2[n[1]])
            c3 = (self.x2[n[2]], self.z2[n[2]], self.y2[n[2]])
            x.append(Tile(c1, c2, c3))

        for n in tr:
            c1 = (self.x2[n[0]], -self.z2[n[0]], self.y2[n[0]])
            c2 = (self.x2[n[1]], -self.z2[n[1]], self.y2[n[1]])
            c3 = (self.x2[n[2]], -self.z2[n[2]], self.y2[n[2]])
            x.append(Tile(c1, c2, c3))

        for n in tr:
            c1 = (self.z3[n[0]], self.x3[n[0]], self.y3[n[0]])
            c2 = (self.z3[n[1]], self.x3[n[1]], self.y3[n[1]])
            c3 = (self.z3[n[2]], self.x3[n[2]], self.y3[n[2]])
            x.append(Tile(c1, c2, c3))

        for n in tr:
            c1 = (-self.z3[n[0]], self.x3[n[0]], self.y3[n[0]])
            c2 = (-self.z3[n[1]], self.x3[n[1]], self.y3[n[1]])
            c3 = (-self.z3[n[2]], self.x3[n[2]], self.y3[n[2]])
            x.append(Tile(c1, c2, c3))

        return x

class Cylinder(Shape):
    """A triangularized cylinder object.

    Parameters
    ----------
    radius : float
        The radius of the cylinder.
    length : float
        The length of the cylinder.
    res : int
        The resolution of the generated triangulation, defaults to 16.

    Attributes
    ----------
    r : float
        Radius.
    l : float
        Length.
    tri : Triangulation
        The triangulation of the side of the cylinder.
    x : list
        The x-coordinates of the side of the cylinder.
    y : list
        The y-coordinates of the side of the cylinder.
    z : list
        The z-coordinates of the side of the cylinder.
    end_x : list
        The x-coordinates of the cap of the cylinder.
    end_y : list
        The y-coordinates of the cap of the cylinder.
    end_z : list
        The z-coordinates of the cap of the cylinder.
    end_tri : Triangulation
        The triangulation of the cap of the cylinder.
    len : int
        Number of triangles in the instance

    """
    def __init__(self, radius, length, res=16):
        self.r = radius
        self.l = length

        x = np.linspace(-self.l/2, self.l/2, res)
        phi = np.linspace(0, 2.0 * np.pi, res)
        X, PHI = np.meshgrid(x, phi)
        X, PHI = X.flatten(), PHI.flatten()

        S = self.r*np.ones((res, res)).flatten()

        # Triangulate parameterized space to determine the triangles
        self.tri = mtri.Triangulation(X, PHI)

        self.x = X
        self.y = S * np.cos(PHI)
        self.z = S * np.sin(PHI)

        self.end_x = self.l/2*np.ones(res**2+1)
        self.end_y = np.append(S * np.cos(PHI), 0)
        self.end_z = np.append(S * np.sin(PHI), 0)
        self.end_tri = mtri.Triangulation(self.end_y, self.end_z)

        self.len = len(self.get_triangles())

    def plot(self, color=None, cmap="plasma"):
        """Generates a trisurface of the Cylinder and returns that for plotting.

        Returns
        -------
        ax
            Axis with cylinder plotted.

        """
        fig = plt.figure(figsize=plt.figaspect(1)*4)  # Square figure
        ax = fig.add_subplot(111, projection='3d')

        c = []
        c.append(ax.plot_trisurf(self.x, self.y, self.z, triangles=self.tri.triangles,
                        alpha=.7, cmap=cmap))
        c.append(ax.plot_trisurf(self.end_x, self.end_y, self.end_z, triangles=self.end_tri.triangles,
                        alpha=.7, cmap=cmap))
        c.append(ax.plot_trisurf(-self.end_x, self.end_y, self.end_z, triangles=self.end_tri.triangles,
                        alpha=.7, cmap=cmap))

        return super().plot(ax, c, color)

    def get_triangles(self):
        """Returns all of the triangles for processing.

        Returns
        -------
        list
            List of all Tiles in the cylinder.

        """
        tr = self.tri.triangles

        x = []
        for n in tr:
            c1 = (self.x[n[0]], self.y[n[0]], self.z[n[0]])
            c2 = (self.x[n[1]], self.y[n[1]], self.z[n[1]])
            c3 = (self.x[n[2]], self.y[n[2]], self.z[n[2]])
            x.append(Tile(c1, c2, c3))

        tr = self.end_tri.triangles

        for n in tr:
            c1 = (self.end_x[n[0]], self.end_y[n[0]], self.end_z[n[0]])
            c2 = (self.end_x[n[1]], self.end_y[n[1]], self.end_z[n[1]])
            c3 = (self.end_x[n[2]], self.end_y[n[2]], self.end_z[n[2]])
            x.append(Tile(c1, c2, c3))

        for n in tr:
            c1 = (-self.end_x[n[0]], self.end_y[n[0]], self.end_z[n[0]])
            c2 = (-self.end_x[n[1]], self.end_y[n[1]], self.end_z[n[1]])
            c3 = (-self.end_x[n[2]], self.end_y[n[2]], self.end_z[n[2]])
            x.append(Tile(c1, c2, c3))

        return x

class nHedron(Shape):
    """A triangularized n-sided polyhedron. Not yet implememnted.

    Parameters
    ----------
    nSides : int
        The desired number of sides.
    diameter : float
        The distance between faces on opposite sides of the object.

    Attributes
    ----------
    d : float
        The distance between faces on opposite sides of the object.

    nSides: int
        The desired number of sides.

    """
    def __init__(self, nSides, diameter):
        self.nSides = nSides
        self.d = diameter
        print("Not yet implememnted")
