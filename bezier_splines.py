# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:42:56 2020

@author: mrloo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 01:48:51 2020

@author: mrloo
"""

import pandas as pd
import numpy as np


import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class Bezier:
    
    def __init__(self, points):
        self.points = points.values
        self.n = len(self.points) - 1
        self.knots = {'x' : np.array([i[0] for i in self.points]), 'y' : np.array([i[1] for i in self.points])}
    
        
    def solve(self, coord_vec):
        """
        Solves for the interior control points of a set of cubic bezier splines 
        such that they are C2 continuous at the knots, given a 1-d vector of knot coordinates
        
        Observe that the coefficient matrix of the system of equations to be solved is a 
        diagonally dominant tridiagonal matrix, hence we can employ the faster O(n) Thomas algorithm 
        with guaranteed stability (as opposed to LU decomposition etc.)
        
        Returns:
            - P1: Vector containing coordinates for the first control point
            - P2: Vector containing coordinates for the second control point
        """
        # R.H.S
        k = [(4 * coord_vec[i]) + (2 * coord_vec[i+1]) for i in range(self.n)]
        k[0] = coord_vec[0] + (2 * coord_vec[1])
        k[-1] = (8 * coord_vec[-2]) + coord_vec[-1]
        
        # L.H.S
        a = [1 for i in range(self.n)]
        b = [4 for i in range(self.n)]
        c = [1 for i in range(self.n)]
        
        a[0] = 0
        a[-1] = 2
        b[0] = 2
        b[-1] = 7
        c[0] = 1
        c[-1] = 0
        
        # Thomas algorithm solves for p1
        for i in range(self.n):
            m = a[i] / b[i-1]
            b[i] = b[i] - (m * c[i-1])
            k[i] = k[i] - (m * k[i-1])
        
        p1 = [k[-1] / b[-1]]
        for i in range(self.n-2, -1, -1):
            x = (k[i] - (c[i] * p1[-1])) / b[i]
            p1.append(x)
        p1.reverse()
        p1 = np.array(p1)
        
        # Get p2 from p1
        p2 = [(2 * coord_vec[i+1]) - p1[i+1] for i in range(self.n-1)]
        p2.append((coord_vec[-1] + p1[-1]) / 2)
        p2 = np.array(p2)
        
        return {'P1' : p1, 'P2' : p2}
    
    def de_casteljau(self, t, control_points):
        """
        De Casteljau's algorithm to evaluate the tangent to the bezier curve defined by the control points
        at given point t
        
        Returns:
            Vector of 3 coordinates
            Note that the 2nd last step of the algorithm gives 3 points on the line tangent to b(t)
        """
        point = control_points
        net = []
        while len(point) > 1:
            point = [(point[i] * (1 - t)) + (point[i+1] * t) for i in range(len(point) - 1)]
            net.append(point)
        
        return [net[-2][0], point[0], net[-2][1]]
    
    def plot(self, arrow_width=0.05):
        """
        Generate the bezier spline using matplotlib pathpatches
        """
        self.fig, self.ax = plt.subplots()
        Path = mpath.Path
        
        # Solve for control points in x and y coordinates
        x = self.solve(self.knots['x'])
        y = self.solve(self.knots['y'])
        
        # Create vector of control points [(x0, y0), ...] for the set of splines
        vertices = []
        for i in range(self.n):
            cp1 = self.points[i]
            cp2 = np.array([x['P1'][i], y['P1'][i]])
            cp3 = np.array([x['P2'][i], y['P2'][i]])
            vertices = vertices + [cp1, cp2, cp3]
        vertices.append(self.points[-1])
        vertices = np.array(vertices)

        # Give the points their respective matplotlib position codes and plot
        codes = [Path.CURVE4] * len(vertices)
        codes[0] = Path.MOVETO
        
        pp = mpatches.PathPatch(Path(vertices, codes), fc='none')
        
        self.ax.add_patch(pp)
        
        # Create arrows to show the order of the bezier curves
        # Draws arrows tangent to the curve using de Casteljau's algorithm
        for i in range(0, len(vertices) - 3, 3):
            cps = vertices[i:i+4]
            bx = [i[0] for i in cps]
            by = [i[1] for i in cps]
            
            px = self.de_casteljau(0.5, bx)
            py = self.de_casteljau(0.5, by)
            
            self.ax.arrow(px[0], py[0], px[1] - px[0], py[1] - py[0],\
                     width=0, length_includes_head=True, head_width=arrow_width, overhang = 0.3, color='none', fc='k')
        
        
        
            


#%%

