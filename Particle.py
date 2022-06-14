import numpy as np
import pygame as pg
from numba import njit

from PyGFun import to_pg, from_pg, WIDTH, HEIGHT
global WIDTH, HEIGHT

global G, a2
G  = 1
a2 = 0.001

class Particle():

    def __init__(self, _pos, _vel, _mass, _r = 2, _color = None, _tolen = None):
        self.x = _pos[0]
        self.y = _pos[1]
        self.x_old, self.y_old = from_pg( (_pos[0], _pos[1]) )
        
        self.aX = 0
        self.aY = 0
        
        self.vx = _vel[0]
        self.vy = _vel[1]
        
        self.mass = _mass

        self.skip = False
        self.first = True
        self.still = False
        
        self.r     = _r
        if _color == None:
            self.color = self.color_mapp()
        else:
            self.color = _color
        self.to_len = _tolen
            
    def color_mapp(self):
        """
        Return a shade of red based on the mass of the particle
        """
        return (25.5 * self.mass/np.sqrt(1 + self.mass**2/100), 0, 0)
            
    def RelDist(self, other):
        """
        Compute the relative distance of 'self' and 'other'.
        'other' can be either a tuple containing the particle coordinates or a Particle() object
        """
        xx, yy = from_pg( (self.x, self.y) )
        if type(other) is tuple:
            return _RelDist(xx, yy, other[0], other[1])
        else:
            xxx, yyy = from_pg( (other.x, other.y) )
            return _RelDist(xx, yy, xxx, yyy)
    
    def checkCases(self, quad):
        """
        Check if the particles in a quad act as a single particle through thei center of mass or if the children need to be checked.
        'quad' is a Quad() object.
        """
        if len(quad.particlesINquad) == 1:
            if quad.particlesINquad[0] == self:   return                                       # Avoid computing forces on itself
            selfx, selfy = from_pg( (self.x, self.y) )                                         
            othx , othy  = from_pg( (quad.particlesINquad[0].x, quad.particlesINquad[0].y) )
            aX, aY = _Acceleration(selfx, selfy, othx , othy, quad.particlesINquad[0].mass)    # Compute the acceleration
            self.aX += aX
            self.aY += aY
        elif len(quad.particlesINquad) > 1:
            d = self.RelDist(from_pg( quad.center_mass ) )                       # Distance between 'self' and the center of mass
            l = quad.w
            if l / d <= quad.theta:                                              # Check if particles act as their center of mass
                selfx, selfy = from_pg( (self.x, self.y) )                             # ↓
                othx , othy  = from_pg( quad.center_mass )                             # ↓
                aX, aY = _Acceleration(selfx, selfy, othx , othy, quad.TotMass)        # ↳  Compute the acceleration
                self.aX += aX 
                self.aY += aY 
            else:                                                                # If condition is not verified 
                for key in quad.keys:                                            # the acceleration is the sum of the
                    self.checkCases(quad.children[key])                          # childrens accelerations
                
    def computeForce(self, tree):
        """
        Start of the algorithm that computes the forces on a particle.
        """
        self.aX, self.aY = 0, 0                             # Reset the accelerations
        for key in tree.RootQuad.keys:                      # Check the children of the Root
            self.checkCases(tree.RootQuad.children[key])
    
    def move(self, tree):
        """
        Computes the trajectory of a particle given its acceleartion using the Basic Störmer–Verlet                                               (https://en.wikipedia.org/wiki/Verlet_integration#Basic_St%C3%B6rmer%E2%80%93Verlet) algorithm.
        It also checks if a particles wander out of the domain, eventually excluding it from the motion.
        
        NOTE: this check needs to be modified and a particle be excluded if its velocity is greater than the escape velocity of the system
        """
        if self.skip or self.still: return      # Skip computations if the particle is outside or is set to be unmovable
        
        x_n, y_n = from_pg( (self.x, self.y) )
        x_nm1, y_nm1   = self.x_old, self.y_old
        
        # Compute the accelerations
        x_np1, y_np1, self.first = _verlet(self.first, x_n, y_n, x_nm1, y_nm1, self.vx, self.vy, self.aX, self.aY, tree.dt)
        
        # Update old and new positions
        self.x_old, self.y_old = x_n, y_n
        self.x, self.y         = to_pg( (x_np1, y_np1) )
        
        # Check if the particle is still in the Tree()
        cond = (0 > self.x > WIDTH*self.to_len) or (0 > self.y > HEIGHT*self.to_len)
        if cond:
            self.skip = True
            self.color = (0,255,0)
    
    def draw(self, SURF):
        """
        Draws a particle.
        """
        if not self.skip:
            pg.draw.circle(SURF, self.color, (self.x, self.y), self.r)
            

@njit 
def _RelDist(xx, yy, xxx, yyy):
    return np.sqrt( (xx - xxx)**2 + (yy - yyy)**2 )    

@njit
def _verlet(first, x_n, y_n, x_nm1, y_nm1, vx, vy, aX, aY, dt):
    if first:
        x_np1 = x_n + vx * dt + 0.5 * aX * dt**2
        y_np1 = y_n + vy * dt + 0.5 * aY * dt**2
        return x_np1, y_np1, False
    else:
        x_np1 = 2 * x_n - x_nm1 + aX * dt**2 
        y_np1 = 2 * y_n - y_nm1 + aY * dt**2 
        return x_np1, y_np1, False

@njit
def _Acceleration(selfx, selfy, othx, othy, mass):
    r2 = _RelDist(selfx, selfy, othx, othy)**2
    aX  =  G * mass * (othx - selfx) / (r2 + a2)**(3/2)
    aY  =  G * mass * (othy - selfy) / (r2 + a2)**(3/2)
    return aX, aY


def DrawAllParticles(tree, SURF):
    """
    Draw all particles in the Tree object, starting from the root and descending into the children.
    """
    quad = tree.RootQuad
    for key in quad.keys:
        quad.children[key].draw_particles(SURF)