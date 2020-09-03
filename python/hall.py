#!/usr/bin/python
"""Implementation of various anomalous Hall conductivities.

   .. warning::
      this is evolving!
"""

import sys,os,re,string,shutil,time
from subprocess import call
import numpy as np
import numpy.linalg as LA
import math
import h5py
import pyfplo.slabify as sla
from cpp_output import OutputGrabber 

__all__ = ['hall','get_profile','diagonalize_bcd','decompose_in_sym_assym','bcd2file']

_names = ["x","y","z"]

_derivative_cases = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]] 
#here the second coordinate is the momentum direction with respect to which the derivative is taken
                     #1xx   2yx   3zx   4xy   5yy   6zy   7xz   8yz   9zz

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def sym_plus_asym(L):
    """
    Obtain symmetric and antisymmetric parts of the BCD.

    Args:

      BCD at a given energy as a list

    Returns: 

      matrix form of BCD decomposition in symmetric and antisymmetric parts
    """
    #A =  L.reshape(3,3)
    AT = L.T #that this is the BCD is related with the ordering of the elements in derivative_cases
    S = 0.5 * (L+AT)
    ST = 0.5 * (L-AT)
    return S,ST

class hall:
  """
  Attributes:

       root_name: root for the folder's name corresponding to different slices.

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
      self.root_name = kwargs.pop('root_name','out')
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
      self.DK = LA.norm(np.cross(self.G1,self.G2)) / (self.mesh[0]*self.mesh[1]) #area associated with k-point in each slice (cross product is the area enclose by G1 and G2)
      print "DK: (if no symmetries are used) ", self.DK
        
      print "VOL_bohr: ", self.VOL_bohr
      print "scale: ", self.S.kscale

  def build_box(self,kz_plane):
      """
      Builds a 2D box centered for the slice. It is centered at the origin and has non-zero intervales along the x and y axis which correspond to G1 and G2. Along G3 it is centered at kz_plane.

      Args:

        kz_plane: number assocaited with the value of proyection of the slice onto G3 (i.e. slices are defined as having a constant proyection on G3).


      Returns:
        list of kpoints already scaled by sla.kscale (so, there should be in the same units as used in the +hamdata file)
      """
      box=sla.BoxMesh()
      box.setBox(xaxis=self.G1,yaxis=self.G2,zaxis=self.G3,origin=self.origin)

      if(self.centered):  
           box.setMesh(nx=self.mesh[0],xinterval=[-0.5*self.dG1,(0.5-1./self.mesh[0])*self.dG1],
                  ny=self.mesh[1],yinterval=[-0.5*self.dG2,(0.5-1./self.mesh[1])*self.dG2],
                  nz=1,zinterval=[kz_plane,kz_plane])
      else:
           box.setMesh(nx=self.mesh[0],xinterval=[0,(1.0-1./self.mesh[0])*self.dG1],
                  ny=self.mesh[1],yinterval=[0,(1.0-1./self.mesh[1])*self.dG2],
                  nz=1,zinterval=[kz_plane,kz_plane])


      kpoints = box.mesh(self.S.kscale) #after this the points are in Bohr
      return kpoints

  def build_box_c2v(self,kz_plane):
      """
      Builds a 2D box centered for the slice. It is centered at the origin and has non-zero intervales along the x and y axis which correspond to G1 and G2. Along G3 it is centered at kz_plane.

      Args:

        kz_plane: number assocaited with the value of proyection of the slice onto G3 (i.e. slices are defined as having a constant proyection on G3).

      Returns:
        list of kpoints already scaled by sla.kscale (so, there should be in the same units as used in the +hamdata file)
      """
      box=sla.BoxMesh()
      box.setBox(xaxis=self.G1,yaxis=self.G2,zaxis=self.G3,origin=[0.,0.,0.])
#      box.setMesh(nx=self.mesh[0],xinterval=[-0.5*self.dG1,(0.5-1./self.mesh[0])*self.dG1],
#                  ny=self.mesh[1],yinterval=[-0.5*self.dG2,(0.5-1./self.mesh[1])*self.dG2],
#                  nz=1,zinterval=[kz_plane,kz_plane])

      box.setMesh(nx=self.mesh[0],xinterval=[0,(0.5-1./self.mesh[0])*self.dG1],
                  ny=self.mesh[1],yinterval=[0,(0.5-1./self.mesh[1])*self.dG2],
                  nz=1,zinterval=[kz_plane,kz_plane])

      kpoints = box.mesh(self.S.kscale) #after this the points are in Bohr
#      kpoints_c2v = []
#      for kp in kpoints:
#        if(kp[0]>= 0 and kp[1]>=0):
#          kpoints_c2v.append(kp)

      return kpoints

  def build_box_c4v(self,kz_plane):
      """
      Builds a 2D box centered for the slice. It is centered at the origin and has non-zero intervales along the x and y axis which correspond to G1 and G2. Along G3 it is centered at kz_plane.

      Args:

        kz_plane: number assocaited with the value of proyection of the slice onto G3 (i.e. slices are defined as having a constant proyection on G3).

      Returns:
        list of kpoints already scaled by sla.kscale (so, there should be in the same units as used in the +hamdata file)
      """
      box=sla.BoxMesh()
      box.setBox(xaxis=self.G1,yaxis=self.G2,zaxis=self.G3,origin=[0.,0.,0.])
      box.setMesh(nx=self.mesh[0],xinterval=[-0.5*self.dG1,(0.5-1./self.mesh[0])*self.dG1],
                  ny=self.mesh[1],yinterval=[-0.5*self.dG2,(0.5-1./self.mesh[1])*self.dG2],
                  nz=1,zinterval=[kz_plane,kz_plane])

      kpoints = box.mesh(self.S.kscale) #after this the points are in Bohr
      kpoints_c4v = []
      for kp in kpoints:
        if(kp[0]>= 0 and kp[1]>=0 and kp[0] >= kp[1]):
          kpoints_c4v.append(kp)

      return kpoints_c4v

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

  def write_info_slice(self,plane):
      root = self.root_name
      if(not os.path.isdir("""%(root)s_%(plane)s"""%locals())):
        os.mkdir("""%(root)s_%(plane)s"""%locals())
      self.ar = h5py.File("""%(root)s_%(plane)s/ar.hdf5"""%locals(),"w")
      run_info = self.ar.create_group("run_info")
      run_info.create_dataset('kscale',data=self.S.kscale)
      run_info.create_dataset('energy_bottom',data=self.energy_bottom)
      run_info.create_dataset('energy_fermi',data=self.energy_fermi)
      run_info.create_dataset('mesh',data=self.mesh)
      run_info.create_dataset('delta',data=self.delta)
      run_info.create_dataset('origin',data=self.origin)
      run_info.create_dataset('realspace_cell',data=[self.G1*self.S.kscale,self.G2*self.S.kscale,self.G3*self.S.kscale])
      run_info.create_dataset('DK',data=self.DK)
      self.ar.close()

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
                           BCD_k[int_mu][alfa,beta] += band_contribution_to_dipole * self.DK

               ind +=1

      end = time.time()
#      print "\n dipole_on_point took: ",(end-start)/3600, "hours"
      return BCD_k

  def k3_plane(self,plane):
      if(self.centered):
         return (-0.5 + plane*1./self.mesh[2])*self.dG3
      else:
         return plane*1./self.mesh[2] * self.dG3

  def dipole_on_slice(self,**kwargs):
      """
        Computes the contribution of the slice to the Berry curvature dipole tensor using the formula

        .. math::
           D_{ab} = \sum_{n,E_{nk}<E_f} \iint dk_1 dk_2 d_{ab}(k)

        Notice that the 1/(2\pi)^3 factor is considered when performing the summation over slices.

        Arguments:

         plane: integer between 0 and self.mesh[2] that defines the slice of k-points


         save_bcd_on_plane: boolean. If True ef_to_save also should be provided.

         ef_to_save: fermi energies for which a list of D_{ab} (one per momentum in the slice) will be stored

         use_symmetries: boolean. If True, D_{ab} is symmetrized.

         symm: this is a temporal fix for generating an irreducible BZ for particular point groups
      """

      plane = kwargs["plane"]
      symm = kwargs.pop("symm","")
      assert(plane > -1 and plane < self.mesh[2])

      kz_plane = self.k3_plane(plane)


      save_bcd_on_plane = kwargs.pop("save_bcd_on_plane",False)
      ef_to_save = None
      ef_to_save_index = []
      if(save_bcd_on_plane):
         ef_to_save = kwargs["ef_to_save"]
         for ef in ef_to_save:
            ef_to_save_index.append(find_nearest(self.energy_fermi,ef))
	 print "ef_to_save_index: ", ef_to_save_index
         assert(ef_to_save != None)

      use_symmetries = kwargs.pop("use_symmetries",False)

      self.write_info_slice(plane)
      if(symm == "c2v"):
        kpoints = self.build_box_c2v(kz_plane)
        self.DK = 0.25 * self.DK
        print "DK changed because we are using symmetries. Remember: the mesh in this case correspond to the irreducible Brillouin zone, so the volume element associated with k-point is smaller. "
      elif(symm == "c4v"):
        kpoints = self.build_box_c4v(kz_plane)
      else:
        kpoints = self.build_box(kz_plane)
      self.write_info_slice(plane)

      print "Number of k-points: ",len(kpoints), "\n"
      file = open("kpoints.dat",'w')
      for kp in kpoints:
	print >> file, kp[0],kp[1],kp[2]
      file.close()

      Dipole_integral = [np.matrix(np.zeros((3, 3))) for i in range(len(self.energy_fermi))]

      total_bcd_on_plane = [] #just for saving bcd(k) if wanted: one list with different selected values of Fermi energy per k-point

      print "From the total of k-points we have computed: "
      i_counter = 0
      start = time.time()
      for kp in kpoints:

           #computes and Berry curvature at kp
           k = np.array(kp)
           BCD_k = self.dipole_on_point(k=k)
         
           if(use_symmetries):
              star_index = self.compute_k_star(k=k)

           BCD_k_to_save = [] #one per energy Fermi considered to save the BCD
           if(save_bcd_on_plane):
              for int_s in ef_to_save_index:
                BCD_k_to_save.append(BCD_k[int_s])
              total_bcd_on_plane.append(BCD_k_to_save)

           ## add the contribution of this k-point to the momentum integration for each Fermi energy calculation
           if(use_symmetries):
              start_ = time.time()
              for int_mu in range(len(self.energy_fermi)):
                  if(self.verbosity > 2):
                     print "\nOriginal BCD at k:\n",  Dipole_integral[int_mu]

                  #call symmetrizer for each mu to replace the matrix Dipole_integral[int_mu]
                  Dipole_integral[int_mu] += self.compute_bcd_partners(kp = k,bcd_k = BCD_k[int_mu], indexes = star_index)

                  if(self.verbosity > 2):
                     print "\nSymmetrized BCD at k:\n",  Dipole_integral[int_mu]
              end_ = time.time()
              #print "\n compute_bcd_partners took: ",(end_-start_)/3600, "hours\n"
           else: 
              for int_mu in range(len(self.energy_fermi)):
                  Dipole_integral[int_mu] += BCD_k[int_mu]

           i_counter += 1
           percentage = i_counter * 1. / len(kpoints) *100
           if(i_counter%10000 == 0):
             print percentage,"%,   ",
      end = time.time()
      #print "\n\n dipole_on_slice took: ",(end-start)/3600, "hours\n"

      root = self.root_name
      self.ar = h5py.File("""%(root)s_%(plane)s/ar.hdf5"""%locals(),"a")
      berry_dipole = self.ar.create_group("berry_dipole")
      berry_dipole.create_dataset('xyz',data=Dipole_integral)

      if(save_bcd_on_plane):
         berry_dipole.create_dataset('bcd_on_plane',data=total_bcd_on_plane)
         berry_dipole.create_dataset('kmesh',data=kpoints)
         berry_dipole.create_dataset('ef_to_save',data=ef_to_save)
         berry_dipole.create_dataset('ef_to_save_index',data=ef_to_save_index)
      self.ar.close()

  def hall_on_slice(self,**kwargs):
      """
        Computes the contribution of the slice to the anomalous Hall conductivity using the formula

        .. math::
           \sigma_{ab} = \sum_{n,E_{nk}<E_f} \iint dk_1 dk_2 \Omega_c

        Notice that the 1/(2\pi)^3 factor is considered when performing the summation over slices.

        Arguments:

         plane: integer between 0 and self.mesh[2] that defines the slice of k-points

      """
      plane = kwargs["plane"]
      assert(plane > -1 and plane < self.mesh[2])
      kz_plane = (-0.5 + plane*1./self.mesh[2])*self.dG3

      self.write_info_slice(plane)

      kpoints = self.build_box(kz_plane)

      Omega_integral = [[0,0,0]  for e in self.energy_fermi] #one per energy Fermi considered
      for kp in kpoints:
        #compute Hamiltonian eigenenergies and Berry curvature at kp
        k = np.array(kp)
        (E,CC,F) = self.compute_bc(k)
        ind = 0
        Omega_kp =  [ [0, 0, 0] for e in self.energy_fermi] #one per energy Fermi considered 
        for e in E:

             if(e > self.energy_bottom and e < self.energy_fermi[-1]): ## i.e.: we only do anything if this state is below the maximum fermi energy considered
                 for ind_ef in range(len(self.energy_fermi)):
                     if(e <= self.energy_fermi[ind_ef]):
                        Omega_kp[ind_ef][0] += F[0][ind] * self.DK
                        Omega_kp[ind_ef][1] += F[1][ind] * self.DK
                        Omega_kp[ind_ef][2] += F[2][ind] * self.DK
             ind +=1

        for ind_ef in range(len(self.energy_fermi)):
          for i in range(3):
             Omega_integral[ind_ef][i] += Omega_kp[ind_ef][i]

      root = self.root_name
      self.ar = h5py.File("""%(root)s_%(plane)s/ar.hdf5"""%locals(),"a")
      anomalous_hall = self.ar.create_group("anomalous_hall")
      anomalous_hall.create_dataset('xyz',data=Omega_integral)
      self.ar.close()

  def integrate_slices(self,**kwargs):
      """
      Integrates the contribution of all the slices. 

      Args:
         what: 'hall' or 'dipole' or 'spin_hall'

         cleaning: boolean. If true delets all folders.

         avoid_integral: list of integers associated with slices that won't be consider.

      Raises:

         Warning if some of the folder is not contributing to the summation.

      Returns:

         Creates results.hdf5 which contains the final results as well as information of the paramters used in the calculation.
      """
      what = kwargs["what"]
      cleaning = kwargs.pop("cleaning",False)
      avoid_integral = kwargs.pop("avoid_integral",[])

      file_name = self.root_name
      aux_3 = np.cross(self.G1,self.G2)
      aux_3_norm = LA.norm(aux_3)
      aux_3 = aux_3 / aux_3_norm
      dk_perp = abs(np.dot(self.G3,aux_3)) /self.mesh[2] 
      change_from_fplo_to_bohr = self.S.kscale**3 /(2*np.pi)**3

      final_arc = h5py.File("results.hdf5",'w')
      profile = final_arc.create_group("profile")
      
      if(self.verbosity > 0):
           print("Distance between layers: ", dk_perp)

      files = ["""%(file_name)s_%(i)s/ar.hdf5"""%locals() for i in range(self.mesh[2])]

      if(what == 'hall'):
         #integrate anomalous Hall

         Om_x = [0 for e in self.energy_fermi]
         Om_y = [0 for e in self.energy_fermi]
         Om_z = [0 for e in self.energy_fermi]

         ind = 0
         for name in files:
            everything_ok = True
            AR = h5py.File(name,'r')
            if(AR.__contains__('anomalous_hall')):
              omega_xyz = np.array(AR['anomalous_hall']['xyz'])
              for ind_ef in range(len(omega_xyz)):
                 Om_x[ind_ef] += omega_xyz[ind_ef][0] * dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)
                 Om_y[ind_ef] += omega_xyz[ind_ef][1] * dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)
                 Om_z[ind_ef] += omega_xyz[ind_ef][2] * dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)
                 ind_ef +=1
            else:
               everything_ok = False
               print "WARNING: folder ",name, " is not contributing to Berry curvature integration"

            if(ind == 0):
                AR.copy("run_info",final_arc)

            if(cleaning and everything_ok):
                rm_folder = """rm -r %(file_name)s_%(ind)s"""%locals()
                os.system(rm_folder)

            ind+=1

         result = final_arc.create_group("AH")
         result.create_dataset("sigma_x",data=Om_x)
         result.create_dataset("sigma_y",data=Om_y)
         result.create_dataset("sigma_z",data=Om_z)
  
         file = open("sigma_xy_vs_mu.dat",'w')
         for ind_ef in range(len(self.energy_fermi)):
           print >> file, self.energy_fermi[ind_ef], Om_z[ind_ef]
         file.close()

         file = open("sigma_xz_vs_mu.dat",'w')
         for ind_ef in range(len(self.energy_fermi)):
           print >> file, self.energy_fermi[ind_ef], Om_y[ind_ef] 
         file.close()

         file = open("sigma_yz_vs_mu.dat",'w')
         for ind_ef in range(len(self.energy_fermi)):
           print >> file, self.energy_fermi[ind_ef], Om_x[ind_ef] 
         file.close()


      if(what == "dipole"):
         #integrate Berry curvature dipole
         number_cases = len(_derivative_cases)
         dipole = [np.matrix(np.zeros((3, 3))) for i in range(len(self.energy_fermi))]

         print "In profile, the BCD of each slice is weighted by dk_perp * change_from_fplo_to_bohr: ",dk_perp * change_from_fplo_to_bohr

         ind = 0
         for name in files:
           if ind in avoid_integral:
              print "Avoiding contribution from slice: ",ind
           else:
              contribution_slice = [np.matrix(np.zeros((3, 3))) for i in range(len(self.energy_fermi))]
              everything_ok = True
              AR = h5py.File(name,'r')
              if(AR.__contains__('berry_dipole')):
                 BCD = np.array(AR['berry_dipole']['xyz'])
                 for ind_ef in range(len(BCD)):
                    for al  in range(3):
                      for be  in range(3):
                         dipole[ind_ef][al,be] += BCD[ind_ef][al,be] * dk_perp * change_from_fplo_to_bohr
                         contribution_slice[ind_ef][al,be] += BCD[ind_ef][al,be] * dk_perp * change_from_fplo_to_bohr

                 result_slice = profile.create_group("""%(ind)s"""%locals())
                 result_slice.create_dataset("tensor",data=contribution_slice)
                 result_slice.create_dataset("k_3",data=self.k3_plane(ind))
              else:
                 everything_ok = False
                 print "WARNING: folder ",name, " is not contributing to Berry curvature dipole integration"

              if(ind == 1):
                  AR.copy("run_info",final_arc)

              if(cleaning and everything_ok):
                  rm_folder = """rm -r %(file_name)s_%(ind)s"""%locals()
                  os.system(rm_folder)

           ind+=1

         result = final_arc.create_group("BCD")
         result.create_dataset("tensor",data=dipole)

         file = open("dipole_vs_mu.dat",'w')
     
         #print list of tensors
         for int_mu in range(len(self.energy_fermi)):
             BCD_sym, BCD_asym = sym_plus_asym(dipole[int_mu])

             print >> file,"   ",self.energy_fermi[int_mu]
             for i in range(3):
               for j in range(3):
                  print >> file,dipole[int_mu][i,j],
               print >> file,"\n",
         file.close()

         #also print a file with each component for easier plotting
         for j in range(len(_derivative_cases)):
           alpha = _derivative_cases[j][1]
           beta = _derivative_cases[j][0]

           al_n = _names[alpha]
           be_n = _names[beta]

           file_name = """D_%(al_n)s_%(be_n)s_vs_mu.dat"""%locals()
           file = open(file_name,'w')

           for int_mu in range(len(self.energy_fermi)):
              BCD_sym, BCD_asym = sym_plus_asym(dipole[int_mu])
              print >> file,self.energy_fermi[int_mu],dipole[int_mu][alpha,beta], BCD_sym[alpha,beta], BCD_asym[alpha,beta]

           file.close()


         print("\n Result of Dipole integrations written in adim units \n")

      if(what == 'spin_hall'):
         #integrate anomalous Hall

         Om_x = [0 for e in self.energy_fermi]
         Om_y = [0 for e in self.energy_fermi]
         Om_z = [0 for e in self.energy_fermi]

         ind = 0
         for name in files:
            everything_ok = True
            AR = h5py.File(name,'r')
            if(AR.__contains__('spin_hall')):
              omega_xyz = np.array(AR['spin_hall']['xyz'])
              for ind_ef in range(len(omega_xyz)):
                 Om_x[ind_ef] += omega_xyz[ind_ef][0] * dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)
                 Om_y[ind_ef] += omega_xyz[ind_ef][1] * dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)
                 Om_z[ind_ef] += omega_xyz[ind_ef][2] * dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)
                 ind_ef +=1
            else:
               everything_ok = False
               print "WARNING: folder ",name, " is not contributing to spin Berry curvature integration"

            if(ind == 0):
                AR.copy("run_info",final_arc)

            if(cleaning and everything_ok):
                rm_folder = """rm -r %(file_name)s_%(ind)s"""%locals()
                os.system(rm_folder)

            ind+=1

         result = final_arc.create_group("SH")
         result.create_dataset("x",data=Om_x)
         result.create_dataset("y",data=Om_y)
         result.create_dataset("z",data=Om_z)

         file = open("spinsigma_yz_vs_mu.dat",'w')
         for ind_ef in range(len(self.energy_fermi)):
           print >> file, self.energy_fermi[ind_ef], Om_x[ind_ef]
         file.close()

         file = open("spinsigma_xz_vs_mu.dat",'w')
         for ind_ef in range(len(self.energy_fermi)):
           print >> file, self.energy_fermi[ind_ef], Om_y[ind_ef]
         file.close()
 
         file = open("spinsigma_xy_vs_mu.dat",'w')
         for ind_ef in range(len(self.energy_fermi)):
           print >> file, self.energy_fermi[ind_ef], Om_z[ind_ef]
         file.close()







  #FOR ANOMALOUS NERNST CALCULATIONS-----------------------------------------------

  def dfermi_dmu(x):
      """
      Derivative of the Fermi distribution with respect to mu. `TO DO: check that is what respect to mu and not energy`
      """
      return np.exp(np.float128(x)) / (1+np.exp(np.float128(x)))**2


  def anomalous_nernst(**kwargs):
      """
      Computes anomalous Nernst conductivity as in Phys. Rev. Lett. 97, 026603 Eq. 8. This is an energy integration of the anomalous Hall conductivitiy which must be computed previously.
  
      Requires:

         betas: list [beta_0,beta_1,...] range of temperatures for which the calculation will be performed (given as 1/kbT, in 1/eV).

         energy_fermi: list [mu_0,mu_1,...] of chemical potential for which the calculation will be performed 
      """

      betas = kwargs['betas']

      energy_fermi = kwargs['energy_fermi']

      #read anomalous hall xy,xz,yz 

      file = open("sigma_xy_vs_mu.dat")
      data_xy = file.readlines()
      file.close()
      file = open("sigma_xz_vs_mu.dat")
      data_xz = file.readlines()
      file.close()
      file = open("sigma_yz_vs_mu.dat")
      data_yz = file.readlines()
      file.close()

      hall_data = [data_xy,data_xz,data_yz]
      labels = ["xy","xz","yz"]
      #read energies
      energies = []
      for fila in data_xy:
        vals = map(eval,fila.split())
        energies.append(vals[0])

      kb = 1.38064852 #10**-23 J/K (boltzmann constant)
      A = 6.25 #10**18 e/s (Ampere)

      unit_factor = kb * A *0.001 #the las factor is the result of 10**(-23+18+2) = 10**-3, where the +2 is for conversing cm to m

      case = 0
      for data in hall_data:
         anom_hall = []
         label = labels[case]
         for fila in data:
           vals = map(eval,fila.split())
           anom_hall.append(vals[1])

         for mu in energy_fermi:
           file_mu = open("""anomalous_nernst_%(label)s_%(mu)s"""%locals(),'w')
   #       """ The format of this file is: 
   #           beta[eV], nernst[A/(meter K)]
   #        """


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

def get_profile(**kwargs):
    """
    Obtain profile of contributions to BCD from different slices.

    Args:
       arxiv_name: hdf5 file with results of BCD

       ef: Fermi energy 

       avoid_integral: list of integers to avoid in the integral of the profile

    Returns:
         
       -1 if fermi energy is not within the calculated ones

       Creates files starting with the profile for each BCD component.
    """

    arxiv_name = kwargs.pop("arxiv_name","results.hdf5")
    ef = kwargs.pop("ef",0.0)
    avoid_integral = kwargs.pop("avoid_integral",[])
    ar = h5py.File(arxiv_name,'r')
    fermi_energy = np.array(ar['run_info']['energy_fermi'])
    profile = ar['profile']

    find_fermi_energy = False
    for  n in range(len(fermi_energy)):
      if((fermi_energy[n]-ef)**2 < 1e-15):
        find_fermi_energy = True
        break

    print "index of Fermi energy: ",n

    if not find_fermi_energy:
      print "Error: the Fermi energy is not within the calculated ones."
      return -1

    file_output = open("out_profile.dat",'w')

    for j in range(len(_derivative_cases)):
       component = _derivative_cases[j]
       alpha = component[1]
       beta = component[0]

       al_n = _names[alpha]
       be_n = _names[beta]

       file_name = """profile_%(al_n)s_%(be_n)s.dat"""%locals()
       file = open(file_name,'w')

       integral = 0
#       print profile.keys()
       for k in profile.keys():
          BCD_L = np.array(profile[k]['tensor'])
          k_3 = np.array(profile[k]['k_3'])
          print >> file,k,k_3,BCD_L[n][alpha,beta]
          if k in avoid_integral:
            print "avoiding",k
          else:
            integral += BCD_L[n][alpha,beta]

       print >> file_output, """Profile %(file_name)s integrates to %(integral)s"""%locals()

       file.close()

def bcd2file(name):
       AR = h5py.File(name,'r')
       if(AR.__contains__('berry_dipole')):
            BD = AR['berry_dipole']
            if(BD.__contains__('bcd_on_plane')):
		number_of_fermi_energies = len(BD['ef_to_save'])
		number_of_kpoints = len(BD['kmesh'])

		for i_ef in range(number_of_fermi_energies):
  		    ef = BD['ef_to_save'][i_ef]

		    for alpha in range(3):
		       for beta in range(3):

		           al_n = _names[alpha]
                           be_n = _names[beta]
  		           file_name = """bcd_%(al_n)s_%(be_n)s_ef_%(ef)s"""%locals()
 		           file = open(file_name,'w')

			   suma = 0
		           for i_kp in range(number_of_kpoints):
  			       kp = BD['kmesh'][i_kp]
			       bcd_component = BD['bcd_on_plane'][i_kp][i_ef][alpha,beta]
			       suma += bcd_component

			       print >> file, kp[0], kp[1], kp[2], bcd_component
			   print >> file, "#SUM= ",suma

                           file.close()

	    else:
		print "Error: missing bcd_on_plane. Did you ask for it?"

