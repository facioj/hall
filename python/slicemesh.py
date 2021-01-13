#!/usr/bin/python

import sys,os,re,string,shutil,time
from subprocess import call
import numpy as np
import numpy.linalg as LA
import math
import h5py
import pyfplo.slabify as sla
from cpp_output import OutputGrabber 

__all__ = ['slicemesh','get_profile']


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

class slicemesh:
  """
  This class provides tools to compute a Brillouin zone integration based a "slicing" method. If the primitive cell is formed by vectors G1, G2, G3, the mesh consists in 2D planes of fixing G3 coordinate.

  Args:

      G : matrix with primitive lattice vectors as rows. 

      gk : callable method. This is the function that will be integrated.

      kscale : conversion factor between units of G and units needed in gk.

      mesh: array indicating the number of subdivions along the i primitive vector.

      linear: boolean. If true, the it is understood that the linear anomalous Hall is computed. This is only for defining units and structure of data in the reading/writting of results

      energy_fermi: list of Fermi energies. 

      verbosity: three relevant ranges: [0-1], [1-2] and [2<].

      origin: array indicating the origin of the slices. By default is [0,0,0]

      centered: boolean. If True the slices are centered at the origin. If False, the origin is one of the vertices. This flexibility is meant for debugging and testing only. By default is True.

      root_name: common name that will be used for the folder corresponding to different slices.

  """
  def __init__(self,**kwargs):
      self.G = kwargs['G']
      self.gk = kwargs['gk']
      self.kscale = kwargs['kscale']
      self.mesh = kwargs['mesh']
      self.verbosity = kwargs.pop('verbosity',0)
      self.origin = kwargs.pop('origin',[0.,0.,0.])
      self.centered = kwargs.pop('centered',True)
      self.root_name = kwargs.pop('root_name','out')
      self.linear = kwargs.pop('linear',True) 
      self.energy_fermi = kwargs.pop('energy_fermi',[0.0]) 

      self.G1=self.G[:,0]
      self.G2=self.G[:,1]
      self.G3=self.G[:,2]

      print("G1 = (" + str(self.G1[0])+", "+ str(self.G1[1])+", "+str(self.G1[2])+")")
      print("G2 = (" + str(self.G2[0])+", "+ str(self.G2[1])+", "+str(self.G2[2])+")")
      print("G3 = (" + str(self.G3[0])+", "+ str(self.G3[1])+", "+str(self.G3[2])+")")

      self.dG1=LA.norm(self.G1)
      self.dG2=LA.norm(self.G2)
      self.dG3=LA.norm(self.G3)

      self.bohr_to_cm = 5.29177e-9     
      self.G_0 = 1./12906.4037217 #ohm-1 (conductance quantum: 2e^2/h)
      self.VOL = np.abs(np.linalg.det(self.G))
      self.DK = LA.norm(np.cross(self.G1,self.G2)) / (self.mesh[0]*self.mesh[1]) #area associated with k-point in each slice (cross product is the area enclose by G1 and G2)
        
      print("Centered? : ", self.centered)
      print("origin? : ", self.origin)
      print("VOL: ", self.VOL)

  def build_box(self,kz_plane):
      """
      Builds a 2D box centered for the slice. It is centered at the origin and has non-zero intervales along the x and y axis which correspond to G1 and G2. Along G3 it is centered at kz_plane.

      Args:

        kz_plane: number associated with the value of proyection of the slice onto G3 (i.e. slices are defined as having a constant proyection on G3).


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


      kpoints = box.mesh(self.kscale) #after this the points are in Bohr
      return kpoints

  def write_info_slice(self,plane):
      root = self.root_name
      if(not os.path.isdir("""%(root)s_%(plane)s"""%locals())):
        os.mkdir("""%(root)s_%(plane)s"""%locals())
      self.ar = h5py.File("""%(root)s_%(plane)s/ar.hdf5"""%locals(),"w")
      run_info = self.ar.create_group("run_info")
      run_info.create_dataset('kscale',data=self.kscale)
      run_info.create_dataset('energy_fermi',data=self.energy_fermi)
      run_info.create_dataset('mesh',data=self.mesh)
      run_info.create_dataset('origin',data=self.origin)
      run_info.create_dataset('DK',data=self.DK)
      self.ar.close()

  def k3_plane(self,plane):
      if(self.centered):
         return (-0.5 + plane*1./self.mesh[2])*self.dG3
      else:
         return plane*1./self.mesh[2] * self.dG3

  def slice_integral(self,**kwargs):
      """
        Computes the integration within a slice using the formula

        .. math::
           G = \sum_{n,E_{nk}<E_f} \iint dk_1 dk_2 g(k)

        Notice that the usual 1/(2\pi)^3 factor is included at a later stage (when performing the summation over slices)

        Args:

           plane: integer between 0 and self.mesh[2] that defines the slice of k-points

           save_gk_on_plane: boolean. If true, g(k) is stored. In that case, ef_to_save also should be provided.

           ef_to_save: fermi energies for which a list of g(k) (one per momentum in the slice) will be stored

      """

      plane = kwargs["plane"]
      assert(plane > -1 and plane < self.mesh[2])

      kz_plane = self.k3_plane(plane)
      save_gk_on_plane = kwargs.pop("save_gk_on_plane",False)

      ef_to_save = None
      ef_to_save_index = []
      if(save_gk_on_plane):
         ef_to_save = kwargs["ef_to_save"]
         for ef in ef_to_save:
            ef_to_save_index.append(find_nearest(self.energy_fermi,ef))
	 print("ef_to_save_index: ", ef_to_save_index)
         assert(ef_to_save != None)

      kpoints = self.build_box(kz_plane)
      self.write_info_slice(plane)

      print("Number of k-points: ",len(kpoints), "\n")

      G_0 = np.matrix(np.zeros((3, 3))) 
      if(self.linear):
           G_0 = np.matrix(np.zeros((3, 1)))

      print("G_0: ",G_0)

      G = [np.copy(G_0) for i in range(len(self.energy_fermi))]

      total_G_on_plane = [] #just for saving g(k) if wanted: one list with different selected values of Fermi energy per k-point

      print("From the total of k-points we have computed: ")
      i_counter = 0
      start = time.time()
      for kp in kpoints:

           #computes and Berry curvature at kp
           k = np.array(kp)
           g_k = self.gk(k)
         
           g_k_to_save = [] #one per energy Fermi considered to save the BCD
           if(save_gk_on_plane):
              for int_s in ef_to_save_index:
                g_k_to_save.append(g_k[int_s])
              total_G_on_plane.append(g_k_to_save)
           else: 
              for int_mu in range(len(self.energy_fermi)):
                  G[int_mu] += g_k[int_mu] * self.DK

           i_counter += 1
           percentage = i_counter * 1. / len(kpoints) *100
           if(i_counter%10000 == 0):
             print(percentage,"%,   ",)
      end = time.time()
      #print "\n\n dipole_on_slice took: ",(end-start)/3600, "hours\n"

      root = self.root_name
      self.ar = h5py.File("""%(root)s_%(plane)s/ar.hdf5"""%locals(),"a")
      G_data = self.ar.create_group("G_data")
      G_data.create_dataset('G',data=G)

      if(save_gk_on_plane):
         G_data.create_dataset('bcd_on_plane',data=total_bcd_on_plane)
         G_data.create_dataset('kmesh',data=kpoints)
         G_data.create_dataset('ef_to_save',data=ef_to_save)
         G_data.create_dataset('ef_to_save_index',data=ef_to_save_index)
      self.ar.close()

  def integrate_slices(self,**kwargs):
      """
      Integrates the contribution of all the slices. 

      Args:
         cleaning: boolean. If true delets all folders.

         avoid_integral: list of integers associated with slices that won't be consider.

      Raises:

         Warning if some folder is not contributing to the summation.

      Returns:

         Creates results.hdf5 which contains the final results as well as information of the paramters used in the calculation.
      """
      cleaning = kwargs.pop("cleaning",False)
      avoid_integral = kwargs.pop("avoid_integral",[])

      file_name = self.root_name
      aux_3 = np.cross(self.G1,self.G2)
      aux_3_norm = LA.norm(aux_3)
      aux_3 = aux_3 / aux_3_norm
      dk_perp = abs(np.dot(self.G3,aux_3)) /self.mesh[2] 
      change_from_fplo_to_bohr = self.kscale**3 /(2*np.pi)**3

      final_arc = h5py.File("results.hdf5",'w')
      profile = final_arc.create_group("profile")
      
      if(self.verbosity > 0):
           print("Distance between layers: ", dk_perp)

      files = ["""%(file_name)s_%(i)s/ar.hdf5"""%locals() for i in range(self.mesh[2])]

      factor =  dk_perp * change_from_fplo_to_bohr
      G_0 = np.matrix(np.zeros((3, 3)))
      if(self.linear):
           G_0 = np.matrix(np.zeros((3, 1)))
           factor =  dk_perp * change_from_fplo_to_bohr * self.G_0 *np.pi / self.bohr_to_cm * (-1.)

      G = [np.copy(G_0) for i in range(len(self.energy_fermi))]

      ind = 0
      for name in files:
         everything_ok = True
         AR = h5py.File(name,'r')
         if(AR.__contains__('G_data')):
             Gs = np.array(AR['G_data']['G'])
             for ind_ef in range(len(self.energy_fermi)):
                 G[ind_ef] += Gs[ind_ef] * factor  

             print(G[0])

             result_slice = profile.create_group("""%(ind)s"""%locals())
             result_slice.create_dataset("tensor",data=Gs[ind_ef]*factor)
             result_slice.create_dataset("k_3",data=self.k3_plane(ind))
         else:
             everything_ok = False
             print("WARNING: folder ",name, " is not contributing to the integration")

         if(ind == 0):
             AR.copy("run_info",final_arc)

         if(cleaning and everything_ok):
            rm_folder = """rm -r %(file_name)s_%(ind)s"""%locals()
            os.system(rm_folder)

         ind+=1

      result = final_arc.create_group("G_integrated")
      result.create_dataset("G",data=G)
  
      #printing to files

      names = ["x","y","z"]
      if(self.linear):
         for beta in range(3):
             name = names[beta]
             file = open("""sigma_%(name)s_vs_mu.dat"""%locals(),'w')
             for ind_ef in range(len(self.energy_fermi)):
                file.write(str(self.energy_fermi[ind_ef])+ " " + str(G[ind_ef][beta,0]) +"\n")
             file.close()

      else:

         file = open("dipole_vs_mu.dat",'w')
     
         #print list of tensors
         for int_mu in range(len(self.energy_fermi)):
             BCD_sym, BCD_asym = sym_plus_asym(G[int_mu])
             file.write(str(self.energy_fermi[int_mu])+ "\n")
#             file.write("   ",self.energy_fermi[int_mu])
             for i in range(3):
               for j in range(3):
                  file.write(str(G[int_mu][i,j])+" ")
               file.write("\n",)
         file.close()

         #also print a file with each component for easier plotting
         for alpha in range(3):
           for beta in range(3):

             al_n = names[alpha]
             be_n = names[beta]

             file_name = """D_%(al_n)s_%(be_n)s_vs_mu.dat"""%locals()
             file = open(file_name,'w')

             for int_mu in range(len(self.energy_fermi)):
                BCD_sym, BCD_asym = sym_plus_asym(G[int_mu])
                file.write(str(self.energy_fermi[int_mu])+" "+str(G[int_mu][alpha,beta])+" "+str(BCD_sym[alpha,beta]) +" "+ str(BCD_asym[alpha,beta])+"\n")

             file.close()

 
      print("\n Result of integrations succesfully written  \n")

def get_profile(**kwargs):
    """
    Obtain profile of contributions from different slices.

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

    names = ["x","y","z"]

    find_fermi_energy = False
    for  n in range(len(fermi_energy)):
      if((fermi_energy[n]-ef)**2 < 1e-15):
        find_fermi_energy = True
        break

    print("index of Fermi energy: ",n)

    if not find_fermi_energy:
      print("Error: the Fermi energy is not within the calculated ones.")
      return -1

    file_output = open("out_profile.dat",'w')

    for j in range(len(_derivative_cases)):
       component = _derivative_cases[j]
       alpha = component[1]
       beta = component[0]

       al_n = names[alpha]
       be_n = names[beta]

       file_name = """profile_%(al_n)s_%(be_n)s.dat"""%locals()
       file = open(file_name,'w')

       integral = 0
#       print profile.keys()
       for k in profile.keys():
          BCD_L = np.array(profile[k]['tensor'])
          k_3 = np.array(profile[k]['k_3'])
          file.write(k,k_3,BCD_L[n][alpha,beta])
          if k in avoid_integral:
            print("avoiding",k)
          else:
            integral += BCD_L[n][alpha,beta]

       file_output.write("""Profile %(file_name)s integrates to %(integral)s"""%locals())

       file.close()

