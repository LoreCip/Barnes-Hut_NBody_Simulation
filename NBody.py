import numpy as np
import pygame as pg
import time
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt

from pygame.locals import *
from numba import njit

from QuadTree import Tree, Quad
from Particle import Particle, DrawAllParticles
from PyGFun import check_Event_Logic, to_pg, from_pg, WIDTH, HEIGHT

global WIDTH, HEIGHT

from collections import Iterable
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item

def radial_dist(quad):
    r = []
    if len(quad.particlesINquad) == 1:
        xx, yy = from_pg( ( quad.particlesINquad[-1].x, quad.particlesINquad[-1].y) )
        r.append(np.sqrt(xx**2 + yy**2))
    elif len(quad.particlesINquad) > 1:
        for key in quad.keys:
            r.append(radial_dist(quad.children[key]))
    return r

def R_hist(tree):
    r = radial_dist(tree.RootQuad)
    r = list(flatten(r))
    fig = plt.figure(figsize=[4, 4], # Inches
               dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
               )
    ax = fig.gca()
    ax.hist(r, bins = 100, histtype = 'step', color = 'r')
    ax.set_title('Histogram of radial positions')
    ax.set_xlabel('Radius')
    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    return canvas, raw_data, fig

def Update_display(SURF, tree, show_tree, fig, canvas, raw_data):
    # If the button "t" is pressed, show the tree
    if show_tree: tree.draw(SURF)
    # Show histogram
    size = canvas.get_width_height()
    sur = pg.image.fromstring(raw_data, size, "RGB")
    SURF.blit(sur, (WIDTH,0))
    # Draw particles
    DrawAllParticles(tree, SURF)
    # update display
    pg.display.update()
    # Close figure to avoid cluttering
    plt.close(fig)
        
def mainLoop(WIN, particles, N_0, old_tree = None, gif = False):
    """
    Main animation loop of pygame.
            - particles: list of Particle() objects, initial condition
            - N_0      : len(particles)
    """
    run = True
    show_tree = False
    clock = pg.time.Clock()
    
    D = WIDTH

    t = 0
    N_frame = 0
    N_fuse = 0
    
    while run:
        t1 = time.time()
        clock.tick(60)
        WIN.fill((0, 0, 26))
        
        if old_tree == None:
            # Compute the Tree starting from the outer Root quad
            OuterQuad = Quad(to_pg((0,0)), D, particles)       # Define outer quad
            tree = Tree(OuterQuad)                             # Define tree object
            tree.createTree()                     # Create the tree checking for collisions
        else:
            tree = old_tree
            N_fuse = old_tree.fused
            particles = old_tree.find_particles(old_tree.RootQuad)
            old_tree = None
            
        # Compute histogram of radial distances
        canvas, raw_data, fig = R_hist(tree)
        
        # Check if some button has been pressed
        show_tree, run = check_Event_Logic(run, show_tree, tree, WIN)
        
        # Update display
        Update_display(WIN, tree, show_tree, fig, canvas, raw_data)
        
        # Compute serially the forces
        for particle in particles:
            if particle.skip == False:
                particle.computeForce(tree)

        # Move all the particles
        for particle in particles:
            particle.move(tree)
        
        t2 = time.time()
        sss = f"Time between frames: {t2 - t1} s"
        print ("\r" + sss, end='')
        
        t += t2 - t1
        
        if gif:
            filename = "output/screen_%05d.png" % ( N_frame )
            pg.image.save( WIN, filename )
        
        N_frame += 1
    
    print()
    print(f'Mean time between frames: {t/N_frame} s')    
    tree.fused = N_fuse
    tree.N_0 = N_0
    
    return tree