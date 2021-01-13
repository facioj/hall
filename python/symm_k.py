#!/usr/bin/python

#from __future__ import absolute_import, division, with_statement

__all__ = ['sym_k']
__author__  = ['Jorge I. Facio']
__date__    = 'Jan 12, 2021'
__email__   = 'facio.ji@gmail.com'
__version__ = '0.'


import sys,os,re,string,shutil,time
from subprocess import call
import numpy as np
import numpy.linalg as LA
import math
import h5py
import pyfplo.slabify as sla
from cpp_output import OutputGrabber 


class symm_k:
  """
  Class that provides methods to handle symmetry properties of the Berry curvature and Berry curvature dipole.

  Args:

       slabify object

       verbosity: three relevant ranges: [0-1], [1-2] and [2<].

       TOL: tolerance to define the star of k-point

  """

  def __init__(self,**kwargs):
      self.S = kwargs['slabify']
      self.verbosity = kwargs.pop('verbosity',0)
      self.TOL = kwargs.pop('TOL',1e-9)
      self.gauge = kwargs.pop('gauge','periodic')
      self.ms  = 0 #we always have spin-orbit coupling
      self.build_rep_matrices()

  def compute_bc(self,k):
      """
      Computes Berry curvature based on the specified gauge.

      Args:
       
      k: np array with the k-points coordinate in the units used in the +hamdata file

      Returns:
         Output of sla.diagonalize(makef=True)
      """
      (Hk,dHk) = self.S.hamAtKPoint(k,self.ms,gauge=self.gauge,makedhk=True)
      (E,CC,F) = self.S.diagonalize(Hk,dhk=dHk,makef=True)

      return (E,CC,F)


  def find_M_beta_gamma(self,**kwargs):
      """
      Finds on the fly transformation matrix of the Berry curvature associated with a given point symmetry.
      It considers three random k points and inverts a 9x9 system of equations whose solution is the unique transformation matrix searched.

      Args:

      R: transformation matrix in momentum space
      """
      R = np.matrix(kwargs["R"])
      ind = kwargs["index"]

      file = open("""g_%(ind)s.dat"""%locals(),"w")

      A = np.matrix(np.zeros((9, 9)))
      B = np.matrix(np.zeros((1, 9)))
      for i in range(3):


          kp = np.random.rand(3,1)
          Rkp = np.matmul(R,np.matrix(kp))

          (E,CC,F) = self.compute_bc(kp)
          (E2,CC2,F2) = self.compute_bc(Rkp)

          print >> file,"\n Random kp: ",i,"\n", kp
          print >> file,"\n Transformed kp: ",i,"\n", Rkp

          for al in range(3):
             ind_al = 3*i + al
             for be in range(3):
                ind_be = 3*al + be
                A[ind_al,ind_be] = F[be][0]
             B[0,ind_al] = F2[al][0]

      print >> file,"A: \n",A
      print >> file,"B: \n",B

      C = np.matmul(LA.inv(A),B.T)

      print >> file,"C: \n",C.reshape(3,3)
      file.close()
      return C.reshape(3,3)


  def build_rep_matrices(self):
      """
      Reads representations matrices in momentum space and builds the transformation matrices associated with the Berry curvature.
      """

      #Symmetry operations are those of the little group at Gamma
      (Hk,WF)=self.S.hamAtKPoint([0,0,0],ms=self.ms,opindices=None,gauge='relative',makewfsymops=True)

      file = open("rep_matrices.dat",'w')

      self.symm = {}
      self.Rinv = {}
      self.bc_rep = {}

      for w in WF:
        if(w.index != 0 and w.index !=1): #except time-reversal for the moment and the identity
          print >> file,"\n -- ",w.index," --\n"
          print >> file,"\n -- ",w.symbol," --\n"
          print >> file,"R=\n",w.alpha
          print >> file,"TR= ",w.timerev
          self.symm[w.index] = w
          self.Rinv[w.index] = (-1)**w.timerev * LA.inv(np.matrix(w.alpha))
          M_al_be = self.find_M_beta_gamma(R=(-1)**w.timerev * w.alpha,index=w.index)
          self.bc_rep[w.index] = np.matrix(M_al_be)
          print >> file,"\n M=\n",M_al_be

      file.close()

  def is_k_in(self,k,star):
      """
      Check if a k-point is in the set called star
      """
      ans = False
      for kp in star:
         if(LA.norm(np.array(k)-np.array(kp)) < self.TOL):
             return True
      return False

  def compute_k_star(self,k):
      """
      Computes the star of a k-point

      Args:
      
      k: k-point

      Returns

      list of indexes associated with the point symmetries that generate the different elements in the star (one per each if there are more than one)

      """
      indexes = []
      k_star = [k.T]
      for key in self.symm.keys():
          G = self.symm[key]

          R = np.matrix(G.alpha)
          Rkp = R * k.T

          if(self.is_k_in(Rkp,k_star) == False):
             k_star.append(Rkp)
             indexes.append(G.index)

      return indexes

  def compute_partners(self,linear,k,G):
      """
      Computes the contribution to the BCD of the k-points related by symmetry to a certain k-point.

      Args:

      linear: Boolean. True for linear AHC, False for BCD. For the moment only the BCD is implemented.

      k: k-point

      G: for the nonlinear case, BCD at k

      Returns:
      
      For the nonlinear case, the BCD sum of the contributions of all partners (included the original) in list format
      """
      BCD_SUMk = np.copy(G) #start from BCD_k so do not take identity in the following

      indexes = self.compute_k_star(k)
      for index in indexes:
         Rinv = self.Rinv[index]
         M = self.bc_rep[index]

         if(self.verbosity > 2):
            print "\n\n->R: \n",R

         BCD_Rk = np.matrix(np.zeros((3, 3)))
         for al in range(3):
            for be in range(3):
                for gamma in range(3):
                    for delta in range(3):
                        BCD_Rk[al,be] += M[be,gamma] * Rinv[delta,al] * BCD_k[delta,gamma]
         BCD_SUMk += BCD_Rk

         if(self.verbosity > 2):
             print "Partner associated with R: \n",BCD_Rk

      return BCD_SUMk

