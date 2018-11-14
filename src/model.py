import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class DCUF(chainer.Chain):

    def __init__(self, w, h, access_num, policy_num, \
                 action_num, icf_epoch, ae_epoch, dcuf_epoch):

        self.w = w
        self.h = h
        self.access_num = access_num
        self.policy_num = policy_num 
        self.action_num = action_num
        self.icf_epoch  = icf_epoch
        self.ae_epoch   = ae_epoch
        self.dcuf_epoch = dcuf_epoch

        prev_hidden_dim = 32
        super(DCUF, self).__init__()

        # in_channel, out_channel, filter_size, stride, (pad)
        # In this sample code, dimensions of the highest hidden layers = the number of action  
        links = [('e_conv1_c', L.Convolution2D(1, 16, (4,4), stride=2))]
        links += [('e_conv2_c', L.Convolution2D(16, 16, (3,3), stride=2))]
        links += [('e_fc3_c', L.Linear(400, prev_hidden_dim))]
        links += [('e_fc4_c', L.Linear(prev_hidden_dim, self.action_num))]

        links += [('d_fc4_c', L.Linear(self.action_num, prev_hidden_dim))]
        links += [('d_fc3_c', L.Linear(prev_hidden_dim, 400))]
        links += [('d_conv2_c', L.Deconvolution2D(16, 16, (3,3), stride=2))]
        links += [('d_conv1_c', L.Deconvolution2D(16, 1, (4,4), stride=2))]

        links += [('e_conv1_u', L.Convolution2D(1, 16, (4,4), stride=2))]
        links += [('e_conv2_u', L.Convolution2D(16, 16, (3,3), stride=2))]
        links += [('e_fc3_u', L.Linear(400, prev_hidden_dim))]
        links += [('e_fc4_u', L.Linear(prev_hidden_dim, self.action_num))]

        links += [('d_fc4_u', L.Linear(self.action_num, prev_hidden_dim))]
        links += [('d_fc3_u', L.Linear(prev_hidden_dim, 400))]
        links += [('d_conv2_u', L.Deconvolution2D(16, 16, (3,3), stride=2))]
        links += [('d_conv1_u', L.Deconvolution2D(16, 1, (4,4), stride=2))]

        for policy in range(0,self.policy_num):
            links += [('pai{}'.format(policy), \
                       L.Linear(prev_hidden_dim, self.action_num))]

        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)

        self.forward = links
        self.train = True

    def __call__(self, nnow_state, next_state, x, y, lamb, epoch):

        vnow_state = Variable(nnow_state)
        t = Variable(nnow_state)

        # h_c, h_u: encoded features of controllable (h_c) and uncontrollable (h_u)
        # y_c, y_u: decoded features of controllable (h_c) and uncontrollable (h_u)
        h_c, h_u = self.encode(vnow_state)
        y_c, y_u = self.decode(h_c,h_u)

        # Reconstruction error
        lamb_buf = lamb
        if epoch < self.icf_epoch:
            # Pre-training step for disentangling controllable factors
            loss_ae = F.mean_squared_error(y_c, t)
        elif epoch >= self.icf_epoch and epoch < self.icf_epoch+self.ae_epoch:
            # Only train DNN for uncontrollable objects (frozen DNN for controllable object)
            # By using this process, DCUF may more stably detect uncontrollable objects
            lamb_buf = 0.0
            y_c_hat = y_c.data
            loss_ae = F.mean_squared_error(y_u, t-y_c_hat)
        else:
            # Equation (4)
            loss_ae = F.mean_squared_error(y_u+y_c, t)

        # Disentangling objective (Equation (2))
        loss = loss_ae
        if lamb_buf > 0.0:
            for policy in range(0,self.policy_num):
                act = self.policy(vnow_state,policy)
                sel = -lamb*self.disentanglement_objective(act,h_c,next_state,policy)
                loss = loss + sel

        return loss

    def encode(self, state):
        h1 = F.relu(self.e_conv1_c(state))
        h2 = F.relu(self.e_conv2_c(h1))
        h3 = F.relu(self.e_fc3_c(h2))
        h4_c = F.tanh(self.e_fc4_c(h3))

        h1 = F.relu(self.e_conv1_u(state))
        h2 = F.relu(self.e_conv2_u(h1))
        h3 = F.relu(self.e_fc3_u(h2))
        h4_u = F.tanh(self.e_fc4_u(h3))
        return h4_c, h4_u

    def decode(self, feature1, feature2):

        # For controllable
        bak = (F.log(1.0+feature1)-F.log(1.0-feature1))/2.0
        h4_c = self.d_fc4_c(bak)
        h3 = F.relu(self.d_fc3_c(h4_c))
        vreh3 = F.reshape(h3,(1,16,5,5))
        h2 = F.relu(self.d_conv2_c(vreh3))
        h1_c = F.relu(self.d_conv1_c(h2))

        # For uncontrollable
        bak = (F.log(1.0+feature2)-F.log(1.0-feature2))/2.0
        h4_u = self.d_fc4_u(bak)
        h3 = F.relu(self.d_fc3_u(h4_u))
        vreh3 = F.reshape(h3,(1,16,5,5))
        h2 = F.relu(self.d_conv2_u(vreh3))
        h1_u = F.relu(self.d_conv1_u(h2))

        return h1_c, h1_u

    def policy(self, state, policy):
        h1 = F.relu(self.e_conv1_c(state))
        h2 = F.relu(self.e_conv2_c(h1))
        h3 = F.relu(self.e_fc3_c(h2))
        h = h3.data
        hhat = Variable(h)
        exec("act = F.softmax(self.pai{}(hhat))".format(policy))
        return act

    def disentanglement_objective(self, act, h, next_state, policy):

        sel = None
        accum_sel = None
        for action in range(0,self.action_num):
            sel = self.compute_selectivity \
                      (h, act, next_state, policy, action)
            if accum_sel is None:
                if sel is not None:
                    accum_sel = sel
            else:
                accum_sel = accum_sel + sel

        return accum_sel

    def compute_selectivity(self, h, act, next_state, policy, action):

        sel = None
        accum_sel = None
        eps = 1.0e-10

        zero = np.zeros((1,self.action_num),dtype=np.float32)
        zero[0][policy] = 1
        vzero = Variable(zero)

        one = np.zeros((1,self.action_num),dtype=np.float32)
        for i in range(0,self.action_num):
            one[0][i] = 1
        vone = Variable(one)

        zero_act = np.zeros((1,self.action_num),dtype=np.float32)
        zero_act[0][action] = 1
        vzero_act = Variable(zero_act)

        for i in range(0,self.access_num):

            rnext_state = next_state[action][i].reshape((1,1,self.w,self.h))
            vnext_state = Variable(rnext_state)
            h4_c, h4_u = self.encode(vnext_state)

            # Log(1/K+(|fk-fk|/sum|fk-fk|))
            sel = F.log( F.sum(F.relu(h4_c*vzero-h*vzero))/(F.sum(F.relu(h4_c*vone-h*vone))+eps) \
                         + 1.0/self.action_num ) 
            if accum_sel is None:
                accum_sel = sel
            else:
                accum_sel = accum_sel + sel

        return (F.max(act*vzero_act)*(accum_sel))
