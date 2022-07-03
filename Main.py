import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import pickle

from NBody import mainLoop
from InitialCond import Compute_IC
from PyGFun import WIDTH, HEIGHT

global WIDTH, HEIGHT



def Save(tree, N_0, name):
    # Store data (serialize)
    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def Load(path):
    # Load data (deserialize)
    with open(path, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    
    inp = input('Insert 1 to execute a new simulation, 2 to load a previous one: ')
    
    try:
        inp = int(inp)
    except ValueError:
        print('Please, inster a number (1 or 2).')
        sys.exit(1)
    except Exception:
        print('Something went wrong!')
        sys.exit(1)
    else:
        if inp == 1:
            N_0 = int(input('Insert the number of particles: '))
            load = False
        elif inp == 2:
            load = True
        else:
            print('Please, instert 1 or 2.')
            sys.exit(1)

    list_of_savefile = []
    list_of_gif      = []
    if load:
        folders = os.listdir('.')
        for folder in folders:
            if '.pickle' in folder:
                list_of_savefile.append(folder)
            elif '.gif' in folder:
                list_of_gif.append(folder)
        if len(list_of_savefile) == 0:
            print('No .pickle found in this path.')
        else:
            print('These are the .pickle files found:')
            print(list_of_savefile)
        path = input(f'Insert path to save file: ')
        if os.path.isfile(path):
            print('Save file found. Retrieving data...')
            old_tree = Load(path)
        else:
            print('File not found. Abort.')
            sys.exit(1)
    
    svFIN = input('Save final state? (Y/N) ')
    FIN = False
    if svFIN == 'Y' or svFIN == 'y' or svFIN == 'yes' or svFIN == 'Yes': 
        FIN = True
    elif svFIN == 'N' or svFIN == 'n' or svFIN == 'no' or svFIN == 'No' or svFIN == '':
        FIN = False
    else:
        print('Abort.')
        sys.exit(1)
        
    svGIF = input('Generate GIF? (Y/N) ')
    
    if svGIF == 'Y' or svGIF == 'y' or svGIF == 'yes' or svGIF == 'Yes':
        GIF = True
    elif svGIF == 'N' or svGIF == 'n' or svGIF == 'no' or svGIF == 'No' or svGIF == '':
        GIF = False
    else:
        print('Abort.')
        sys.exit(1)
            
    if not load:    
        print(f'Computing initial conditions for {N_0} particles')
        particles = Compute_IC(N_0, seed = 10000)
    print('Done.')    
        
        
    print('Starting simulation...')
    
    #####################################################################################################
    ###          MAIN LOOP
    #####################################################################################################
    
    pg.init()    
    WIN = pg.display.set_mode((WIDTH+400, HEIGHT))
    pg.display.set_caption("B-H Simulation")
    
    if load:
        tree = mainLoop(WIN, particles = None, N_0 = old_tree.N_0, old_tree = old_tree, gif = GIF)
    else:
        tree = mainLoop(WIN, particles, N_0, gif = GIF)
  
    pg.quit()
    
    
#####################################################################################################
###          SAVE
#####################################################################################################
    
    if FIN:
        name = f'savefile_N{tree.N_0}_'
        nnn = 0
        for svfN in list_of_savefile:
            if name in svfN:
                nnn += 1
        name += f'{nnn}'
        print(f"Saving final state to savefile_N{tree.N_0}.pickle")
        Save(tree, tree.N_0, name)
    
    if GIF:
        name = f'simulation_N0{tree.N_0}_'
        nnn = 0
        for gifN in list_of_gif:
            if name in gifN:
                nnn += 1
        name += f'{nnn}'
        
        print(f'Generating {name}.gif')
        os.system(f'ffmpeg -r 60 -f image2 -s {WIDTH+400}x{HEIGHT} -i output/screen_%05d.png {name}.gif -y > /dev/null')
        os.system('rm output/*.png')
    print('Done.')
