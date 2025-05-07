#!/usr/bin/env python
# coding: utf-8

# ## A 1D diffusion mode

# Here we develop a one-dimensional model of diffusion. 

# Here is the diffusion equation

# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# 

# Here is the discretized version of diffusion equation 

# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$

# This is the FTCS scheme

# we will use two libraries. Numpy (for arrays) and Matplotlib (for plotting), that are not a part of the core Python distribution

# In[1]:


import numpy as np
import matplotlib .pyplot as plt


# start by setting two fixed model parameters, the diffusivity and the size of the model domain  `

# In[2]:


D = 100
Lx = 300


# In[3]:


dx = 0.5
x = np.arange(start = 0, stop = Lx, step = dx)
nx = len(x)


# Set the initial conditions for the model. the cake C is a step function with a high value of the left, a low value on the right, and a step at the center of the domain

# In[4]:


C = np.zeros_like(x);
C_left = 500
C_right = 0
C[x<=Lx/2] = C_left
C[x>Lx/2] = C_right


# In[5]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initial Profile")


# In[6]:


nt = 5000
dt = 0.5*dx**2/D


# Loop over the time steps of the model, solving the diffusion equation using the FTCS scheme shown above. Note the use of array opearatoons on the variable C. The boundary conditions remain fixed in each time step.

# In[7]:


for t in range(0,nt):
    C[1:-1] += D*dt/dx**2*(C[:-2]-2*C[1:-1]+C[2:])


# In[8]:


plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final Profile")

