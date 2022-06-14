import numpy as np
import pygame as pg
from numba import njit

from Particle import Particle

from PyGFun import to_pg, from_pg, WIDTH, HEIGHT
global WIDTH, HEIGHT


class Tree():
    
    def __init__(self, _quad):
        self.RootQuad = _quad
        self.dt = 1/20
        self.fused = None
        self.N_0 = None
        
    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def createTree(self):
        """
        Starts the algorithm to create the Tree()
        """
        return self.check_and_split(self.RootQuad)
        
    def check_and_split(self, quad):
        """
        Then it checks if the quad needs to be split, keeping track of the subsequent possible collisions.
        """
        if len(quad.particlesINquad) > 1:              # If there is more than one particle in the quad
                quad.SplitQuad()                       # split the quad
                for key in quad.keys:                  # Check the children
                    self.check_and_split(quad.children[key])
    
    def find_particles(self, quad):
        """
        Create a list with all the particles in the tree. Needed to update the praticle list after the collisions.
        """
        outs = []
        if quad.children['00'] == None:
            return quad.particlesINquad
        else:
            for key in quad.keys:
                outs.extend(self.find_particles(quad.children[key]))
        return outs
    
    def draw(self, SURF):
        """
        Draw the entire tree starting from the Root quad.
        """
        quad = self.RootQuad
        for key in quad.keys:
            quad.children[key].draw(SURF)

class Quad():
    
    def __init__(self, _center, _w, _particlesINquad, _color = (3,120,19), _h=None, _theta=0.8):
        self.done  = False
        self.color = _color
        
        if _h == None:
            _h = _w
        
        self.center = _center
        self.w = _w
        self.h = _h
        
        self.particlesINquad = _particlesINquad
        if len(_particlesINquad) > 0:
            self.center_mass, self.TotMass = self.computeCM()
        else:
            self.done = True
            
        self.theta = _theta
            
        self.children = {}           #   __________ 
        self.children['00'] = None   #  | 11 | 10 |
        self.children['01'] = None   #  |____|____|
        self.children['10'] = None   #  | 01 | 00 |
        self.children['11'] = None   #  |____|____|
        self.keys = ['00', '01', '10', '11']
        
    def SplitQuad(self):
        """
        Compute the centers of the four children and passes them to another function.
        """
        center = from_pg(self.center)
        w, h   = self.w, self.h
        # Compute N-W center
        NW_center = (center[0] + w / 4, center[1] + h / 4)
        # Compute N-E center
        NE_center = (center[0] - w / 4, center[1] + h / 4)
        # Compute S-W center
        SW_center = (center[0] + w / 4, center[1] - h / 4)
        # Compute S-E center
        SE_center = (center[0] - w / 4, center[1] - h / 4)
        self.quadChildren([NW_center, NE_center, SW_center, SE_center], w / 2, h / 2)
    
    def quadChildren(self, centers, w, h):
        """
        Find the particles in each children and create the four quads. 
        """
        for i, key in enumerate(self.keys):
            part = self.countParticles(centers[i], w, h)
            self.children[key] = Quad(to_pg(centers[i]), w, part)

    def computeCM(self):
        """
        Compute the center of mass and total mass of a quad.
        """
        cmX = 0
        cmY = 0
        M = 0
        for particle in self.particlesINquad:
            xx, yy = from_pg( (particle.x, particle.y) )
            M += particle.mass
            cmX += particle.mass *  xx
            cmY += particle.mass *  yy
        return to_pg((cmX / M, cmY / M)), M  

    def countParticles(self, center, w, h):
        """
        Find which particles are in a quad's children.
        """
        parts = []
        for particle in self.particlesINquad:
            x, y = from_pg((particle.x, particle.y))
            if (center[0] - w/2 <= x < center[0] + w/2) and (center[1] - h/2 <= y < center[1] + h/2):
                parts.append(particle)
        return parts  
    
    def draw(self, SURF):
        """
        Draw a quad or its children.
        """
        if self.children['00'] == None:
            rect = pg.Rect(0, 0, self.w, self.h)
            rect.center = self.center
            pg.draw.rect(SURF, self.color, rect, width = 1)
        else:
            for key in self.keys:
                self.children[key].draw(SURF)
                
    def draw_particles(self, SURF):
        """
        Draw all the particles in a quad if it has no children or goes deeper in the tree.
        """
        if self.children['00'] == None:
            for particle in self.particlesINquad:
                particle.draw(SURF)
        else:
            for key in self.keys:
                self.children[key].draw_particles(SURF)