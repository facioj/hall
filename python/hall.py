#!/usr/bin/python
"""Implementation of various anomalous Hall conductivities.

   The main pourpose is to return a callable method that computes the integrated in energy Berry curvature or Berry curvature dipole to be interfaced with the adaptive mesh project.

"""

#from __future__ import absolute_import, division, with_statement

__all__ = ['hall','bcd2file']
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


_names = ["x","y","z"]
_derivative_cases = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]] 
#here the second coordinate is the momentum direction with respect to which the derivative is taken
                     #1xx   2yx   3zx   4xy   5yy   6zy   7xz   8yz   9zz

class hall:
  """
  Attributes:

       delta: value of displacement in momentum space used for numerical derivatives.

       energy_bottom:  bottom energy for integrations. It could be the lowest energy considered in the tight-binding model or some other value if, for instance, it is consider that a certain set of bands do not contribute to the integration. By default is -100eV.

       mesh: subdivisions in the Brillouin zone.

       verbosity: three relevant ranges: [0-1], [1-2] and [2<].

       gauge: gauge used for the Berry curvature calculation. By default is 'periodic'. `TO DO: implement the use of the relative gauge.`

       TOLBD: contribution of a Bloch state to the BCD larger than this number are neglected under the assumption that comes from degeneracies that should cancell between each other. By default: 1e25. Recommendation:  change and evaluate if it affects the results.

  Args:

       slabify object

       energy_bottom: number.

       energy_fermi: [ef_1,..,ef_n] list of Fermi energies to considered. By default is [0.0].

       delta: displacement to perform the numerical derivative. By default 1e-5.

       mesh: [N1,N2,N3], N_i is the number of subdivions along the i primitive vector.

       verbosity: integer.

       gauge: 'periodic' or 'relative'. 

       centered_scheme: Boolean, it specifies how the numerical derivative of the Berry curvature is done
  """

  def __init__(self,**kwargs):
      self.S = kwargs['slabify']
      self.delta = kwargs.pop('delta',1e-5)
      self.verbosity = kwargs.pop('verbosity',0)
      self.ms  = 0 #we always have spin-orbit coupling
      self.TOLBD = kwargs.pop("TOLBD",1e25)
      self.energy_bottom = kwargs.pop('energy_bottom', [-100.])
      self.energy_fermi = kwargs.pop('energy_fermi',[0.0])
      self.gauge = kwargs.pop('gauge','periodic')
      self.centered_scheme = kwargs.pop('centered_scheme',True)
      self.centered = kwargs.pop('centered',True)
      print "Using gauge, ", self.gauge
      print "Centered slice, ", self.centered
      print "Scheme_Centered, ", self.centered_scheme
      self.mesh = kwargs['mesh']
      self.origin = kwargs.pop('origin',[0.,0.,0.])
      R=self.S.hamdataCCell()
      G=self.S.hamdataRCell()
      self.G1=G[:,0]
      self.G2=G[:,1]
      self.G3=G[:,2]
      self.dG1=LA.norm(self.G1)
      self.dG2=LA.norm(self.G2)
      self.dG3=LA.norm(self.G3)
      self.bohr_to_cm = 5.29177e-9     
      self.G_0 = 1./12906.4037217 #ohm-1 (conductance quantum: 2e^2/h)
      self.VOL_fplo = np.abs(np.linalg.det(G))
      self.VOL_bohr = np.abs(np.linalg.det(G))*(self.S.kscale)**3
        
      print "VOL_bohr: ", self.VOL_bohr
      print "scale: ", self.S.kscale

  def compute_E(self,k):
      """
      Computes eigenvalues

      Args:
       
      k: np array with the k-points coordinate in the units used in the +hamdata file

      Returns:
         Output of sla.diagonalize(makef=True)
      """
      Hk = self.S.hamAtKPoint(k,self.ms)
      print "Here"
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
          print >> file,"\n -- ",w.symbol," --\n"
          print >> file,"R=\n",w.alpha
          self.symm[w.index] = w
          self.Rinv[w.index] = LA.inv(np.matrix(w.alpha))
          M_al_be = self.find_M_beta_gamma(R=w.alpha,index=w.index)
          self.bc_rep[w.index] = np.matrix(M_al_be)
          print >> file,"M=\n",M_al_be

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

      Returns

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
#         R = np.matrix(G.alpha)

#        if(self.verbosity > 2):
#            print "\n\n->R: \n",R

#         Rinv = LA.inv(R)
         BCD_Rk = np.matrix(np.zeros((3, 3)))
         for al in range(3):
            for be in range(3):
                for gamma in range(3):
                    for delta in range(3):
                        BCD_Rk[al,be] += M[be,gamma] * Rinv[delta,al] * BCD_k[delta,gamma] 
         BCD_SUMk += BCD_Rk

#        if(self.verbosity > 2):
#            print "Partner associated with R: \n",BCD_Rk


      return BCD_SUMk

  def dipole_on_point(self,**kwargs):
      """
      Computes the Berry curvature dipole tensor at a given k-point

        .. math::
           d_{ab}(k) = \sum_{n,E_{nk}<E_f}  \\frac{\partial \Omega_b}{\partial k_a} dk

      Args:
         
        k point

      Returns:

        list of BCD tensors (one tensor per Fermi energy)
 

      """

      start = time.time()
      k = kwargs["k"]
      if(self.verbosity > 2):
            print "----> k: ",k

      F = None
      if(not self.centered_scheme):
         (E,CC,F) = self.compute_bc(k)
	 Delta = self.delta
      else:
         E = self.compute_bc(k)[0] #to change when the bug in diagonalize is corrected by Klauss
         Delta = 2*self.delta

      BCD_k = [np.matrix(np.zeros((3, 3))) for i in range(len(self.energy_fermi))]
      for case in _derivative_cases:
             
           alfa = case[1]
           beta = case[0]

           ### evaluate displaced k-point for numerical derivative
           k_disp = np.copy(k)
           k_disp[case[1]] += self.delta


           ### evaluate displaced Berry curvature
           (E,CC,F_disp) = self.compute_bc(k_disp)

	   if(self.centered_scheme):
              k_disp_minus = np.copy(k)
              k_disp_minus[case[1]] -= self.delta
              (E,CC_disp_minus,F) = self.compute_bc(k_disp_minus)


           ind = 0 #band index
           for e in E:

               if(e > self.energy_bottom and e < self.energy_fermi[-1]): ## i.e.: we only do anything if this state is below the maximum fermi energy considered

               ### evaluate finite difference for this momentum and band
                  band_contribution_to_dipole = (F_disp[case[0]][ind]-F[case[0]][ind]) / Delta

	   	  if(self.verbosity > 3):
	             if(band_contribution_to_dipole > 1e6):
			print "WARNING, here some large contribution:, ", k_disp,alfa,beta,"-F_disp: \n", F_disp[case[0]][ind]
			print "WARNING, here some large contribution:, ", k,alfa,beta,"-F: \n", F[case[0]][ind]
			print "wanna try different delta??\n\n "

                  if(np.abs(band_contribution_to_dipole) > self.TOLBD):
                     if(self.verbosity > 1):
                        print("WARNING: not adding some large contributions at k-point: ", k, "band index: ", ind, band_contribution_to_dipole,e)
               ### add this contribution to different Fermi energy calculations depending on band energy
                  else:
                     for int_mu in range(len(self.energy_fermi)):
                        if(e <= self.energy_fermi[int_mu]):
                           BCD_k[int_mu][alfa,beta] += band_contribution_to_dipole 

               ind +=1

      end = time.time()
#      print "\n dipole_on_point took: ",(end-start)/3600, "hours"
      return BCD_k


#some methods for post-processing. We leave them here for the moment

def decompose_in_sym_assym(**kwargs):
    """
    Obtain symmetric and antisymmetric parts of the BCD for all computed Fermi energies.

    Args:

       arxiv_name: hdf5 file with results of BCD

    Returns:

       Creates files sym.dat and asym.dat
    """

    arxiv_name = kwargs.pop("arxiv_name","results.hdf5")
    ar = h5py.File(arxiv_name,'r')
    fermi_energy = np.array(ar['run_info']['energy_fermi'])
    bcd = np.array(ar['BCD']['tensor'])

    file_s = open("sym.dat",'w')
    file_as = open("asym.dat",'w')

    for n in range(len(bcd)):
      BCD_L = bcd[n]
      BCD_tensor,S,ST = sym_plus_asym(BCD_L)

      print >> file_s,"    ",fermi_energy[n]
      for i in range(3):
        for j in range(3):
            print >> file_s, S[i][j],
        print >> file_s,"\n",

      print >> file_as,"    ",fermi_energy[n]
      for i in range(3):
        for j in range(3):
            print >> file_as, ST[i][j],
        print >> file_as,"\n",

    file_s.close()
    file_as.close()

def diagonalize_bcd(**kwargs):
    """
    Function to diagonalize symmetric part of the BCD

    Args:
       arxiv_name: hdf5 file with results of BCD
       ef: Fermi energy 

    Returns:
         
       -1 if fermi energy is not within the calculated ones

       Creates "out_diag.data" with results of the diagonalization
    """
    arxiv_name = kwargs.pop("arxiv_name","results.hdf5")
    ef = kwargs.pop("ef",0.0)
    ar = h5py.File(arxiv_name,'r')
    fermi_energy = np.array(ar['run_info']['energy_fermi'])
    bcd = np.array(ar['BCD']['tensor'])

    find_fermi_energy = False
    for  n in range(len(fermi_energy)):
      if((fermi_energy[n]-ef)**2 < 1e-15):
        find_fermi_energy = True
        break

    if not find_fermi_energy:
      print "Error: the Fermi energy is not within the calculated ones."
      return -1

    BCD_L = bcd[n]
    BCD_tensor,S,ST = sym_plus_asym(BCD_L)
    file = open("out_diag.dat",'w')
    print >> file,"BCD (original list):\n", BCD_L,"\n"
    print >> file,"BCD:\n", BCD_tensor,"\n"
    print >> file,"Symmetric part of BCD:\n", S,"\n"
    w, v = LA.eig(S)
    print >> file, "Eigenvalues:\n", w,"\n"
    print >> file, "Eigenvectors:\n", v,"\n"


