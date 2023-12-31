# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:34:04 2023

@author: HMOHAN
"""

import numpy as np
import matplotlib.pyplot as plt
n =100;

Nxi=51;
Neta=21;
Nxc= int((Nxi-1)/2 +1);
xini = np.linspace(0,1,Nxc)

## Insert airfoil configuration
t = 0.12;
c=1.0;
m=0.02;
p=0.4;

xt =xini/c;
yt = 5.*t*((0.2969*np.sqrt(xt)) - (0.1260*xt) - (0.3516*xt**2) + (0.2843*xt**3) - (0.1036*xt**4))


# plt.figure(1)
# plt.plot(x/c, yt, 'k-', lw=2)    
# plt.plot(x/c, - yt, 'k-', lw=2)  
# plt.axis('equal'); plt.ylim((-0.5, 0.5))



## Equation for a cambered 4-digit NACA airfoil#
##  First locate the mean camber line
# def yc1(x,m,p,c):
if (xini.any()>=0.0) and (xini.any()<=p*c):
    yc = (m* xini * ((2.*p) - xini/c))/(p**2.);
    # return yc
elif (xini.any()>p*c) and (xini.any()<=c):
    yc = m * (c - xini) * (1. + (xini/c) - 2. * p) / ((1. - p)**2);
    # return yc
else:
    print("oye bosdike")
    raise ValueError
# def grad(x,m,p,c):
if (xini.any()>=0.0) and (xini.any()<=p*c):
    dyc = np.arctan(2.0 * m * (p - (xini/c)) / ((1. - p)**2))
    # return dyc
        
elif (xini.any()>p*c) and (xini.any()<=c):
    dyc = np.arctan(2.0 * m * (p - (xini/c)) / ((1. - p)**2))
    # return dyc
else:
        raise ValueError
# def airfoil(x,m,p,c):
Xu = (xini/c) - (yt*np.sin(dyc));
Yu = yc+ (yt*np.cos(dyc));
Xl = (xini/c) + (yt*np.sin(dyc));
Yl = yc - (yt*np.cos(dyc));
    # return Xu, Yu, Xl,Yl

# Xu,Yu,Xl,Yl= airfoil(x,p,m,c);
plt.figure(1)
plt.plot(Xu, Yu, 'k-', lw=2)    
plt.plot(Xl, Yl, 'k-', lw=2)  
plt.axis('equal'); plt.ylim((-0.5, 0.5))
plt.title("NACA 2412")


#########  Algebraic grid generation ############################
x1 = np.linspace(0,Nxi-1,Nxi);
y1 = np.linspace(0,Neta-1,Neta);
[X,Y] =np.meshgrid(x1,y1);

plt.figure(2)
plt.title("Computational Grid")
for idx in range(Nxi):
    plt.plot(X[:,idx], Y[:,idx], 'b-', lw=2)
for idx in range(Neta):
    plt.plot(X[idx,:], Y[idx,:], 'b-', lw=2)
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta$')
    
## The inner boundary is to be mapped onto the airfoil or base AA
x=np.zeros((Nxi,Neta));
y=np.zeros((Nxi,Neta));
x[:Nxc,0]= np.flipud(Xl)## :: corresponds to the reverse order slice operation
x[Nxc:,0]=Xu[1:]
# y[:Nxc,0]=Yl[-1::-1].copy()
y[:Nxc,0]=np.flipud(Yl)
y[Nxc:,0]=Yu[1:]


## The outer boundary is the circle or O type grid which is to be mapped along the base CC
## CC and AC on the computational domain are parallel sides of the rectangle which consist of Nxi points
Ro  = 5;
## Divide the angle of the circle by Nxi points

th = -np.linspace(0,2*np.pi,Nxi)
x[:,-1] = Ro*np.cos(th) + (0.5*c);
y[:,-1] = Ro*np.sin(th)
## Now to divide the interior points corresponding to the indices 0<=i<=-1   and 1<=j<=-2
for i in range(Nxi):
    x[i,1:-1] = np.linspace(x[i,0],x[i,-1],Neta)[1:-1]
    y[i,1:-1] = np.linspace(y[i,0],y[i,-1],Neta)[1:-1]
    
    


def plotmesh(x,y):
   
    for i in range(Nxi):
        plt.plot(x[i, :], y[i, :], 'k.-', lw=1)
    for i in range(Neta):
        plt.plot(x[:, i], y[:, i], 'k.-', lw=1)
# plt.figure(3)
# plotmesh(x,y)
# plt.figure(3)
# plt.figure(figsize=(8, 8), dpi=100)
# plotmesh(x,y)
# plt.title('Physical Grid')
# plt.xlabel('x'); plt.ylabel('y');
# plt.axis('equal')
# # plt.xlim((-5.5, 5.5)); plt.ylim((-6,6))
# plt.xlim((-1.5, 1.5)); plt.ylim((-1,1))
#### After creating an algebraic grid around the airfoil, the next step is to refine/cluster this grid via elliptic PDEs
## The algebraic grids serve as a good starting point or initial guess
iter = 0;
  ### Gauss Seidel elliptic differential equation solver
  
  
  
def compute_a_b_c(x,y): ## difference coefficients
    a = ((x[1:-1,2:] - x[1:-1,:-2])/2)**2     +    ((y[1:-1,2:] - y[1:-1,:-2])/2)**2    ### Only compute the interior points, boundary is fixed
    c = ((x[2:,1:-1] - x[:-2,1:-1])/2)**2     +    ((y[2:,1:-1] - y[:-2,1:-1])/2)**2  ;
    b = 0.25*(((x[2:,1:-1] - x[:-2,1:-1]))*((x[1:-1,2:] - x[1:-1,:-2]))  +  ((y[2:,1:-1] - y[:-2,1:-1]))*((y[1:-1,2:] - y[1:-1,:-2])));
    return a,b,c
  
def compute_X_Y_elliptic(a,b,c,F):
    V = 0.5*(a*(F[2:,1:-1]  + F[:-2,1:-1]) +
        (c*(F[1:-1,2:]  + F[1:-1,:-2])) - 
        (0.5*b*(F[2:,2:] - F[2:,:-2] + F[:-2,:-2] - F[:-2,2:])))/(a+c)
    return V
plt.figure(figsize=(8, 8), dpi=100)


xstore=[] 
ystore=[]
while True:
    iter=iter+1
    
    xupdate = x.copy();
    yupdate = y.copy();
     
  
    xzeros = np.append([x[-2,:],x[0,:]],[x[1,:]],axis=0);
    yzeros = np.append([y[-2,:],y[0,:]],[y[1,:]],axis=0);
    
    a,b,c= compute_a_b_c(xzeros,yzeros)
    x[0,1:-1] = compute_X_Y_elliptic(a, b, c, xzeros)
    y[0,1:-1] = compute_X_Y_elliptic(a, b, c, yzeros)
    
    x[-1,1:-1] =  x[0,1:-1];
    y[-1,1:-1] =  y[0,1:-1];
    
    a,b,c= compute_a_b_c(x,y)
    x[1:-1,1:-1] = compute_X_Y_elliptic(a, b, c, x)
    y[1:-1,1:-1] = compute_X_Y_elliptic(a, b, c, y)
    
    xstore.append(x)
    ystore.append(y)
    
    xiter = np.stack(xstore,axis=2)
    yiter = np.stack(xstore,axis=2)
    errx = np.abs(x - xupdate)
    erry = np.abs(y - yupdate)

    if (errx.max() <= 1e-6) and (erry.max() <= 1e-6):
         break


plotmesh(x,y)
plt.show()    

print(iter)
    
