# # Equilibrium Propagation

# #TODO: some introduction to energy-based methods and equilibrum propagation

# +
import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap

import numpy as onp

from network import jit_free_relaxation, jit_clamped_relaxation
from layerednet import LayeredNet
from utils import vmap_mean

from random import randint
from collections import deque
from six.moves.urllib.request import urlretrieve
# -

BATCH_SIZE=20
SEED=0


def train(net, 
          train_loader,
          epochs,
          lr=0.1,
          valid_loader=None,
          valid_interval=200):
    
    cost_fn = net.cost_fn()
    
    free_relaxation = jit_free_relaxation(LayeredNet, batched=True)
    clamped_relaxation = jit_clamped_relaxation(LayeredNet, lr=lr, batched=True)

    for epoch in range(1, epochs+1):
        train_hits = 0
        for step, (x, y) in enumerate(train_loader()):
            # set input and relax states (ie: compute fixed-point)
            net.x = x
            net = free_relaxation(net)

            # log training accuracy
            yi = np.argmax(y, axis=1)
            pred_yi = np.argmax(net.output, axis=1)
            train_hits += np.sum(pred_yi == yi).item()
            
            # update the weights based on expected output
            net = clamped_relaxation(net, y)
            
        log_string = ("epoch: {0} | train_acc: {1:.3f}"
                      .format(epoch, float(train_hits) / ((step+1)*BATCH_SIZE)))
        
        if valid_loader is not None and not epoch % valid_interval:        
            hits = 0
            for step, (x, y) in enumerate(valid_loader()):
                # set input and relax states 
                net.x = x
                net = free_relaxation(net)
                
                yi = np.argmax(y, axis=1)
                pred_yi = np.argmax(net.output, axis=1)
                hits += np.sum(pred_yi == yi).item()
                
            valid_log = ("valid_acc: {1:.3f}"
                         .format(epoch, float(hits) / ((step+1)*BATCH_SIZE)))
            log_string = log_string + " | " + valid_log
                                                             
        print(log_string)


# ### Synthetic Data

# We start by training the network on a synthetic dataset. The input consists of a random one-hot vector and the output is simply the identity on this vector

def dataloader():
    for _ in range(100):
        xs, ys = [], []
        for _ in range(BATCH_SIZE):
            x, y = onp.zeros(3), onp.zeros(3)
            j = randint(0, 2)
            x[j] = 1
            y[j] = 1
            xs.append(x)
            ys.append(y)
        yield np.stack(xs), np.stack(ys)

# Both a network with no hidden layers and a single hidden layer are able to solve the task

# +
net = LayeredNet.new(3, 3, [], random.PRNGKey(SEED))
net = net.batch(BATCH_SIZE)

print("Training Depth 0 Network:")
train(
    net,
    epochs=10,
    train_loader=dataloader, 
    valid_loader=dataloader,
    valid_interval=2)
print()

net = LayeredNet.new(3, 3, [10], random.PRNGKey(SEED))
net = net.batch(BATCH_SIZE)


print("Training Depth 1 Network:")
train(
    net,
    epochs=10,
    train_loader=dataloader, 
    valid_loader=dataloader,
    valid_interval=2)
print()
# -

# While a network with more than one hidden layer struggle to solve the task

# +
net = LayeredNet.new(3, 3, [10, 10], random.PRNGKey(SEED))
net = net.batch(BATCH_SIZE)

print("Training Depth 2 Network:")
train(
    net,
    epochs=10,
    train_loader=dataloader, 
    valid_loader=dataloader,
    valid_interval=2)
# -

# ### MNIST 

# +
path = "mnist.npz"
urlretrieve('https://s3.amazonaws.com/img-datasets/mnist.npz', path)

with onp.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'].astype(np.float32), f['y_train']
    x_test, y_test = f['x_test'].astype(np.float32), f['y_test']
    
print("train size: {0}, test_size: {1}".format(len(x_train), len(x_test)))


# +
def train_mnist():
    for i in range(0, len(x_train), BATCH_SIZE):
        xs = onp.reshape(x_train[i:i+BATCH_SIZE], (-1, 28*28))
        ys = np.eye(10)[y_train[i:i+BATCH_SIZE]]
        yield xs, ys
        
def valid_mnist():
    for i in range(0, len(x_test), BATCH_SIZE):
        xs = onp.reshape(x_test[i:i+BATCH_SIZE], (-1, 28*28))
        ys = np.eye(10)[y_test[i:i+BATCH_SIZE]]
        yield xs, ys

# +
net = LayeredNet.new(28*28, 10, [512, 512], random.PRNGKey(SEED))
net = net.batch(BATCH_SIZE)

lr = [0.4, 0.1, 0.01]
lr = [*lr, *lr]

train(
    net,
    epochs=100,
    lr=lr,
    train_loader=train_mnist, 
    valid_loader=valid_mnist,
    valid_interval=2)
