#!/usr/bin/python

from __future__ import absolute_import, division, with_statement

__all__ = ['hall_k']
__author__  = ['Jorge I. Facio']
__date__    = 'Sept 3, 2020'
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


class hall_k:
  """
  The pourpose of this class to is to return a callable method that computes the summed-over-bands Berry curvature or Berry curvature dipole at a given k-point. Such callable method is to be interfaced with schemes that perform the three-dimensional integration, like the adaptive mesh project. The __call__ constructor is therefore introduced. This method has as argument the coordinates of the k-point (fplo units), and can return either the contribution to the linear anomalous Hall or the nonlinear (see below).

  Args:

       slabify object

       linear: boolean. If true, the __called__ constructor computes the contribution of a certain k-point to the linear anomalous Hall conductivity. If false, it computes the contribution to the Berry curvature dipole

       energy_fermi: [ef_1,..,ef_n] list of Fermi energies to considered. By default is [0.0].

       verbosity: three relevant ranges: [0-1], [1-2] and [2<].

       gauge: gauge used for the Berry curvature calculation. By default is 'periodic'. `TO DO: implement the use of the relative gauge.`

       delta: displacement to perform the numerical derivative. By default 1e-5.

       centered_scheme: Boolean, it specifies how the numerical derivative of the Berry curvature is done

       Ndim: integer. Number of dimensions, assumed to be 3 (but could be 2).

       energy_bottom:  float. Bottom energy for integrations, by default -100eV. It could be used if, for instance, it is considered that a certain fully occupied and separated set of bands do not contribute to the integration.

       TOLBD: contribution of a Bloch state to the BCD larger than this number are neglected under the assumption that comes from degeneracies that should cancell between each other. By default: 1e25. Recommendation:  change and evaluate if it affects the results.

  """

  def __init__(self,**kwargs):
      self.S = kwargs['slabify']
      self.Ndim = kwargs.pop('Ndim',3)
      self.delta = kwargs.pop('delta',1e-5)
      self.verbosity = kwargs.pop('verbosity',0)
      self.ms  = 0 #we always have spin-orbit coupling
      self.TOLBD = kwargs.pop("TOLBD",1e25)
      self.energy_bottom = kwargs.pop('energy_bottom', [-100.])
      self.energy_fermi = kwargs.pop('energy_fermi',[0.0])
      self.gauge = kwargs.pop('gauge','periodic')
      self.centered_scheme = kwargs.pop('centered_scheme',True)
      self.linear = kwargs.pop('linear',True)
      print("Using gauge, ", self.gauge)
      print("Centered slice, ", self.centered)
      print("Scheme_Centered, ", self.centered_scheme)
      print("k-scale: ", self.S.kscale)

  def compute_E(self,k):
      """
      Computes eigenvalues

      Args:
       
      k: np array with the k-points coordinate in the units used in the +hamdata file

      Returns:
         Output of sla.diagonalize(makef=True)
      """
      Hk = self.S.hamAtKPoint(k,self.ms)
      if(self.verbosity<2):
          out = OutputGrabber()
          out.start()
          (E,CC) = self.S.diagonalize(Hk)
          out.stop()
          del out
      else:
          (E,CC) = self.S.diagonalize(Hk)

      return E

  def compute_bc(self,k):
      """
      Computes Berry curvature based on the specified gauge.

      Args:
       
      k: np array with the k-points coordinate in the units used in the +hamdata file

      Returns:
         Output of sla.diagonalize(makef=True)
      """
      (Hk,dHk) = self.S.hamAtKPoint(k,self.ms,gauge=self.gauge,makedhk=True)
      if(self.verbosity<2):
          out = OutputGrabber()
          out.start()
          (E,CC,F) = self.S.diagonalize(Hk,dhk=dHk,makef=True)
          out.stop()
          del out
      else:
          (E,CC,F) = self.S.diagonalize(Hk,dhk=dHk,makef=True)

      return (E,CC,F)

  #methods for the use of point symmetries

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
      for i in range(_Ndim):

           
          kp = np.random.rand(_Ndim,1)
          Rkp = np.matmul(R,np.matrix(kp))

          (E,CC,F) = self.compute_bc(kp)
          (E2,CC2,F2) = self.compute_bc(Rkp)

          file.write("\n Random kp: ",i,"\n", kp)
          file.write("\n Transformed kp: ",i,"\n", Rkp)

          for al in range(_Ndim):
             ind_al = _Ndim*i + al
             for be in range(_Ndim):
                ind_be = _Ndim*al + be
                A[ind_al,ind_be] = F[be][0]
             B[0,ind_al] = F2[al][0]

      file.write("A: \n",A)
      file.write(,"B: \n",B)

      C = np.matmul(LA.inv(A),B.T)
           
      file.write("C: \n",C.reshape(_Ndim,_Ndim))
      file.close()
      return C.reshape(_Ndim,_Ndim)

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
          file.write("\n -- ",w.symbol," --\n")
          file.write(,"R=\n",w.alpha)
          self.symm[w.index] = w
          self.Rinv[w.index] = LA.inv(np.matrix(w.alpha))
          M_al_be = self.find_M_beta_gamma(R=w.alpha,index=w.index)
          self.bc_rep[w.index] = np.matrix(M_al_be)
          file.write("M=\n",M_al_be)

      file.close()

  def is_k_in(self,k,star):
      """
      Check if a k-point is in the set called star
      """
      TOL = 1e-9
      ans = False
      for kp in star:
         if(LA.norm(np.array(k)-np.array(kp)) < TOL):
             return True
      return False

  def compute_k_star(self,**kwargs):
      """
      Computes the star of a k-point

      Args:
          k: k-point

      Returns:
          list of indexes associated with the point symmetries that generate the different elements in the star (one per each if there are more than one)
      """
      kp = np.matrix(kwargs["k"])
      indexes = [] 
      k_star = [kp.T]
      for key in self.symm.keys():
          G = self.symm[key]

          R = np.matrix(G.alpha)
          Rkp = R * kp.T

          if(self.is_k_in(Rkp,k_star) == False):
             k_star.append(Rkp)
             indexes.append(G.index)

      return indexes


  def compute_bcd_partners(self,**kwargs):
      """
      Computes the contribution to the BCD of the k-points related by symmetry to a certain k-point.

      Args:
          kp: k-point
          indexes: indexes associated with the star of kp
          bcd_k: BCD at k

      Returns:
          The BCD sum of the contributions of all partners (included the original) in list format
      """

      kp = np.matrix(kwargs["kp"])
      indexes = kwargs["indexes"]
      BCD_k = np.matrix(kwargs["bcd_k"])
      BCD_SUMk = np.copy(BCD_k) #start from BCD_k so do not take identity in the following

      for index in indexes:
         #G  = self.symm[index]
         Rinv = self.Rinv[index]
         M = self.bc_rep[index]
         BCD_Rk = np.matrix(np.zeros((_Ndim, _Ndim)))
         for al in range(_Ndim):
            for be in range(_Ndim):
                for gamma in range(_Ndim):
                    for delta in range(_Ndim):
                        BCD_Rk[al,be] += M[be,gamma] * Rinv[delta,al] * BCD_k[delta,gamma] 
         BCD_SUMk += BCD_Rk

      return BCD_SUMk

  def sum_bc_on_point(self,k):
      """
      Computes the sum over bands of the Berry curvature vector at a given k-point

        .. math::
           om_{a}(k) = \sum_{n,E_{nk}<E_f}  \Omega_a  
      
      Args:
          k point

      Returns:
          list of BC vectors (one per Fermi energy)
 
      """

      if(self.verbosity > 2):
            print("----> k: ",k)

      F = None
      (E,CC,F) = self.compute_bc(k)

      Om_k = [[0,0,0] for i in range(len(self.energy_fermi))]

      ind = 0 #band index
      for e in E:
           if(e > self.energy_bottom and e < self.energy_fermi[-1]):
                for int_mu in range(len(self.energy_fermi)):
                     if(e <= self.energy_fermi[int_mu]):
                           for beta in range(_Ndim):
                               Om_k[int_mu][beta] += F[beta][ind]

           ind +=1

      return Om_k

  def sum_bcd_on_point(self,k):
      """
      Computes the summed over bands Berry curvature dipole tensor at a given k-point

        .. math::
           d_{ab}(k) = \sum_{n,E_{nk}<E_f}  \\frac{\partial \Omega_b}{\partial k_a} 

      Args:
          k point

      Returns:
          list of BCD tensors (one tensor per Fermi energy)
 
      """

      start = time.time()
      if(self.verbosity > 2):
            print("----> k: ",k)

      F = None
      if(not self.centered_scheme):
         (E,CC,F) = self.compute_bc(k)
	 Delta = self.delta
      else:
         E = self.compute_bc(k)[0] #to change when the bug in diagonalize is corrected by Klauss
         Delta = 2*self.delta

      BCD_k = [np.matrix(np.zeros((_Ndim, _Ndim))) for i in range(len(self.energy_fermi))]

      #we iterate over the momentum directions to which we will estimate the derivative
      for alfa in range(_Ndim): 
             
           ### evaluate displaced k-point for numerical derivative
           k_disp = np.copy(k)
           k_disp[alfa] += self.delta

           ### evaluate displaced Berry curvature. we rewrite the eigenenergies, because we don't need them here
           (Ed,CC,F_plus) = self.compute_bc(k_disp)

	   if(self.centered_scheme):
              k_disp_minus = np.copy(k)
              k_disp_minus[alfa] -= self.delta
              (Ed,CC,F) = self.compute_bc(k_disp_minus)

           ind = 0 #band index
           band_contribution_to_dipole = [0 for i in range(_Ndim)] #one component for each Berry curvature component
           for e in E:

               if(e > self.energy_bottom and e < self.energy_fermi[-1]): ## i.e.: we only do anything if this state is below the maximum fermi energy considered

               ### evaluate finite difference for this momentum and band
                  for beta in range(_Ndim):
                     band_contribution_to_dipole[beta] = (F_plus[beta][ind]-F[beta][ind]) / Delta

                  if(np.sum(np.abs(band_contribution_to_dipole)) > self.TOLBD):
                     if(self.verbosity > 1):
                        print("WARNING: not adding some large contributions at k-point: ", k, "band index: ", ind, np.sum(np.abs(band_contribution_to_dipole)),e)
               ### add this contribution to different Fermi energy calculations depending on band energy
                  else:
                     for int_mu in range(len(self.energy_fermi)):
                        if(e <= self.energy_fermi[int_mu]):
                           for beta in range(_Ndim):
                               BCD_k[int_mu][alfa,beta] += band_contribution_to_dipole[beta]

               ind +=1

      end = time.time()
#      print("\n dipole_on_point took: ",(end-start)/3600, "hours")
      return BCD_k

  def __call__(self,k):
      if(self.linear):
           return self.sum_bc_on_point(k)
      else:
           return self.sum_bcd_on_point(k)

