import random
import numpy as np


box_size = 3
brange = int(box_size/2+0.5)
init_x = 20
init_y = 20
width = 24
height = 24


# Environment setting (random setting of 2x2 pixels)
def environment(action_num, t):

    prev_state = np.zeros((width,height),dtype=np.float32)
    state = np.zeros((width,height),dtype=np.float32)

    # Box
    nx = int(random.random()*(width-3))+1
    ny = int(random.random()*(height-3))+1

    nx_u = int(random.random()*(width-3))+1
    ny_u = int(random.random()*(height-3))+1

    # Set the box in pixels
    touch_flag = 0
    state, touch_flag = uncontrollable(prev_state, nx, ny, nx_u, ny_u, t)
    if touch_flag == 1:
        nx = init_x
        ny = init_y

    for j in range(nx-brange, nx+brange+1):
        for i in range(ny-brange, ny+brange+1):
            state[i][j] = 1

    return state, nx, ny, nx_u, ny_u


# Get next state based on the action of agent
def next_state_from_environment(x, y, init_x_u, init_y_u, \
                                paction, action_num, t):
            
    prev_state = np.zeros((width,height),dtype=np.float32)
    state = np.zeros((width,height),dtype=np.float32)
    x_shift = int(random.random()*3.0+0.5)
    y_shift = int(random.random()*3.0+0.5)

    # Compute coordinates (x,y) of box based on action
    if paction == 1:# Left shift
        nx = x-x_shift
        ny = y
    elif paction == 2:# Right shift
        nx = x+x_shift
        ny = y
    elif paction == 3:# Up shift
        nx = x
        ny = y-y_shift
    elif paction == 4:# Down shift
        nx = x
        ny = y+y_shift
                
    if nx >= width-1-brange:
        nx = x
    if ny >= height-1-brange:
        ny = y
    if nx < 2:
        nx = x
    if ny < 2:
        ny = y

    paction = int(random.random()*(action_num+0.5))
    x_shift_u = x_shift
    y_shift_u = y_shift
    if paction == 0:# Do nothing
        nx_u = init_x_u
        ny_u = init_y_u
    elif paction == 1:# Left shift_u
        nx_u = init_x_u-x_shift_u
        ny_u = init_y_u
    elif paction == 2:# Right shift_u
        nx_u = init_x_u+x_shift_u
        ny_u = init_y_u
    elif paction == 3:# Up shift_u
        nx_u = init_x_u
        ny_u = init_y_u-y_shift_u
    elif paction == 4:# Down shift_u
        nx_u = init_x_u
        ny_u = init_y_u+y_shift_u
                
    if nx_u >= width-1-brange:
        nx_u = init_x_u
    if ny_u >= height-1-brange:
        ny_u = init_y_u
    if nx_u < 2:
        nx_u = init_x_u
    if ny_u < 2:
        ny_u = init_y_u


            
    # Set the box in pixels
    touch_flag = 0
    state, touch_flag = uncontrollable(prev_state, nx, ny, nx_u, ny_u, t)
    if touch_flag == 1:
        nx = init_x
        ny = init_y
    for j in range(nx-brange, nx+brange+1):
        for i in range(ny-brange, ny+brange+1):
            state[i][j] = 1

    return state

def uncontrollable(state, nx, ny, nx_u, ny_u, t):

    # Beam
    touch_flag = 0

    # Set the box in pixels
    for j in range(nx_u, nx_u+3):
        for i in range(ny_u, ny_u+3):
            state[i][j] = 0.5

    for j in range(nx-brange, nx+brange+1):
        for i in range(ny-brange, ny+brange+1):
            if state[i][j] != 0:
                touch_flag = 1

    return state, touch_flag
