import pygame as pg
import numpy as np
from numba import njit

global WIDTH, HEIGHT
WIDTH, HEIGHT = 1000, 1000

@njit
def _to_pg(coords):
    """Numba-ed version of to_pg."""
    return (WIDTH / 2 + coords[0], HEIGHT / 2 + coords[1])

def to_pg(coords):
    """Convert coordinates into pygame coordinates."""
    return _to_pg(np.array(coords))

@njit
def from_pg(coords):
    """Convert coordinates from pygame coordinates."""
    return (coords[0] - WIDTH / 2, coords[1] - HEIGHT / 2)


def check_Event_Logic(run, show_tree, tree, SURF):
    """
    Handels the button pressed:
        - quit button
        - s        : show/hide tree
        - SpaceBar : pause/unpause
     """
    for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.KEYDOWN:
                # Show Tree
                if event.key == ord ("t"): show_tree = not show_tree 
                # Pause
                if event.key == (pg.K_SPACE):
                    while True: # Infinite loop that will be broken when the user press the space bar again
                        event = pg.event.wait()
                        if event.type == pg.KEYDOWN and event.key == pg.K_SPACE: break #Exit infinite loop
                        # Show/hide tree during pause
                        if event.type == pg.KEYDOWN and event.key == ord ("t"): 
                            show_tree = not show_tree
                            SURF.fill((0, 0, 26, 1))
                            if show_tree: tree.draw(SURF)
                            DrawAllParticles(tree, SURF)
                            pg.display.update()
                        # Quit during pause
                        if event.type == pg.QUIT: run = False; break                        
    return show_tree, run
