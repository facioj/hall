#! /usr/bin/env python


import sys,os

sys.path.append(os.path.abspath("/home/jf11/hall/python/"))
import numpy as np
import numpy.linalg as LA
import pyfplo.slabify as sla
import pyfplo.fploio as fploio
import pyfplo.common as com

import hall
import slicemesh

print('\npyfplo version=: {0}\nfrom: {1}\n'.format(sla.version,sla.__file__))

verbosity = 1
linear = True

if __name__ == '__main__':

    hamdata='+hamdata'
    s=sla.Slabify()
    s.object='3d'
    s.printStructureSettings()
    s.prepare(hamdata)

    plane = eval(sys.argv[1])
    what_to_do = eval(sys.argv[2])

    energy_bottom = -9.5
    energy_step = 0.01
    width = 0.5
    N_e = int(width/energy_step)
    energy_fermi = [-0.25+energy_step*i for i in range(N_e)]

    ms = 0
    N1 = 100
    N2 = 100
    N3 = 100
    g_k = hall.hall_k(slabify=s,
                       linear = linear,
                       energy_bottom = energy_bottom,
                       energy_fermi = energy_fermi,
                       verbosity = verbosity
                       )


    mesh = slicemesh.slicemesh(G = s.hamdataRCell(),
                               gk = g_k,
                               mesh = [N1,N2,N3],
                               energy_fermi = energy_fermi,
                               kscale = s.kscale,
                               linear = linear,
                               verbosity = verbosity
                              )


    if(what_to_do == 0):
       mesh.slice_integral(plane = plane)

    if(what_to_do == 1):
       mesh.integrate_slices(avoid_integral=[])
