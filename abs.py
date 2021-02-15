import math
import time
import numpy as np
import scipy as sp
import Potential as Pt
from numpy import linalg as LA
import sys


#---------------------------------
# KINETIC ENERGY
def T(R,ns,mass=9267.5654):
 nR = len(R)
 N = nR * ns
 Te = np.zeros((N,N))
 Rmin = float(R[0])
 Rmax = float(R[-1])
 step = float((Rmax-Rmin)/nR)
 K = np.pi/step

 for ri in range(nR):
  for rj in range(nR):
   if ri == rj:  
    Tij = (0.5/mass)*K**2/3*(1+(2.0/nR**2)) 
   else:    
    Tij = (0.5/mass)*(2*K**2/(nR**2))*((-1)**(rj-ri)/(np.sin(np.pi*(rj-ri)/nR)**2)) 
   for i in range(ns):
     row = ri + i*nR
     col = rj + i*nR
     Te[row,col] = Tij  
 return Te
#---------------------------------



#---------------------------------
# CREATION OF THE HAMILTONIAN
#---------------------------------
def H(R, nf):
 Vij = Pt.Help(R, nf)
 Tij = T(R, 2 * nf)
 H = Vij + Tij 
 return H
#---------------------------------
#---------------------------------

#--------------------------------
# CONVERSION VALUE FROM PS TO AU
ps = 41341.37
#--------------------------------

#=================================
# Intial parameters
nf = 20
ns = nf * 2
Rmin = 1.8
Rmax = 5.0
nR = 200
aniskip = 2500
#---------------------------------
dR = float((Rmax-Rmin)/nR)
R = np.arange(Rmin,Rmax,dR)
mass = 9267.5654
#=================================

#---------------------------------
# Diagonalization and reseting of the inital hamiltonian

Ham = H(R,nf)
Mu  = Pt.DM(R, nf)
#hamiltonian = np.savetxt('Hamiltonian.txt',Ham)
E, V = Pt.Diag(Ham)
#np.savetxt("E.txt",E[:nE])
#np.savetxt("V.txt",V[:,:nE])
Ham = 0
print "Diagonalization Done"
fob = open("abs.txt","w+") 
muE = np.matmul(np.matmul(V.T,Mu),V)
print muE[0,0], muE[1,0],muE[2,0]
eps = 0.00002
w = np.arange(0.01,0.2,0.0002)/27.2114
I = np.zeros((len(w)), dtype="complex") 
c = 137.0
for iw in range(len(I)):
  for iE in range(1,100):
    dE = E[iE] - E[0]
    I[iw] += (muE[iE,0]**2)/(dE - w[iw]  - 1j * eps) 
  I[iw] *= 4.0 * np.pi * w[iw] / 137.0
  fob.write("%s\t%s\n"%(w[iw]*27.2114,I[iw].imag)) 
fob.close()
#---------------------------------









