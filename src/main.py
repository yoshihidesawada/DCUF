###
# python main.py ${lam} ${ICF_EPOCH} ${AE_EPOCH} ${DCUF_EPOCH}
###
import sys
import random

import numpy as np
import chainer

import environment as Env
import model as Model

if len(sys.argv) != 5:
    print 'error'
    print sys.argv
    exit(0)

_, lamb, icf_epoch, ae_epoch, dcuf_epoch = sys.argv

def disentangling_controllable_uncontrollable_factors(lamb, icf_epoch, ae_epoch, dcuf_epoch):

    # Hyper-parameters setting
    # policy_num: the number of policies
    # action_num: the number of actions (e.g., up, down, left, right)
    # access_num: hyper-parameter for computing disentangled objective
    # max_epoch: iteration number
    policy_num = 4
    action_num = policy_num
    access_num = 20
    max_epoch = icf_epoch+ae_epoch+dcuf_epoch
    
    # Input state initialized settings
    state = np.zeros((Env.width,Env.height),dtype=np.float32)

    # chainer setup
    optimizer = chainer.optimizers.Adam()
    model_dcuf = Model.DCUF(Env.width, Env.height, access_num, policy_num, \
                            action_num, icf_epoch, ae_epoch, dcuf_epoch)
    optimizer.setup(model_dcuf)
    
    # Training iteration
    for t in range(0,max_epoch):

        # Get state from environment
        # state: t-th image
        # x, y: coordinate of the controllable object 
        # x_u, y_u: coordinate of the uncontrollable object 
        state, x, y, x_u, y_u = Env.environment(action_num,t)

        # Gradient initialization
        model_dcuf.zerograds()

        # next_state: t+1-th images to compute disentangled objective
        # s: t+1-th image computed from (x,y), (x_u,y_u), action
        restate = state.reshape((1, 1, Env.width, Env.height))
        next_state = []
        for k in range(1,action_num+1):
            for j in range(0,access_num): 
                s = Env.next_state_from_environment(x, y, x_u, y_u,\
                                                    k, action_num, t+1)
                next_state.append(s)
                    
        next_state = np.array(next_state, dtype=np.float32)
        renext_state = next_state.reshape((action_num,access_num,Env.width,Env.height))

        # chainer optimization
        loss_dcuf = model_dcuf(restate, renext_state, x, y, lamb, t)
        loss_dcuf.backward()
        optimizer.update()

    # Save model
    outfile = "dcuf"+".model"
    chainer.serializers.save_npz(outfile,model_dcuf)

if __name__ == '__main__':

    # ramdom seed initialization
    # This initialization is not so important
    random.seed(1)
    np.random.seed(1)

    # Hyper-parameters
    # lamb: lambda to balance between disentangled objective and reconstruction error
    # icf_epoch: epoch of ICF
    # ae_epoch: epoch of AE (by setting it, DCUF may more stably detect uncontrollable objects)
    # dcuf_epoch: epoch of DCUF
    lamb = float(lamb)
    icf_epoch = int(icf_epoch)
    ae_epoch = int(ae_epoch)
    dcuf_epoch = int(dcuf_epoch)

    disentangling_controllable_uncontrollable_factors(lamb, icf_epoch, ae_epoch, dcuf_epoch)
