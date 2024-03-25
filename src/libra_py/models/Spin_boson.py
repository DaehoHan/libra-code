#*********************************************************************************                     
#* Copyright (C) 2018-2022 Alexey V. Akimov                                                   
#*                                                                                                     
#* This file is distributed under the terms of the GNU General Public License                          
#* as published by the Free Software Foundation, either version 3 of
#* the License, or (at your option) any later version.                                                 
#* See the file LICENSE in the root directory of this distribution   
#* or <http://www.gnu.org/licenses/>.          
#***********************************************************************************
"""
.. module:: models_Tully
   :platform: Unix, Windows
   :synopsis: This module implements a spin-boson Hamiltonian for testing NA-MD dynamics
.. moduleauthor:: Alexey V. Akimov, Daeho Han

"""
import os
import sys
import math
import copy

if sys.platform=="cygwin":
    from cyglibra_core import *
elif sys.platform=="linux" or sys.platform=="linux2":
    from liblibra_core import *
import util.libutil as comn
import libra_py.units as units


class tmp:
    pass    

def Spin_boson(q, params, full_id):
    """
    This is the spin-boson Hamiltonian or dissipative two-state system under the harmonic baths. 
    The number of nuclear DOFs is considered as the number of classical oscillators.
    Specific parametrizations follow the work of Templaar and Reichman (J. Chem. Phys. 2018, 148, 102309),
    where all parameters are given using the reference thermal energy at 300 K.
    
    Args: 
        q ( MATRIX(ndof,1) ): coordinates of the nuclear DOFs
        params ( dictionary ): model parameters

            * **params["V]** ( double ): diabatic coupling  [ default: 0.5*9.5e-4, units: Ha ]            
            * **params["E"]** ( double ): diabatic energies [ default: 0.5*9.5e-4, units: Ha ]
            * **params["lambda"]** ( double ): the reorgarnization energy  [ default: 1.0*9.5e-4, units: Ha]
            * **params["omega"]** ( double ): the system-bath coupling energy [ default: 0.1*9.5e-4, units: Ha] 
            * **params["mass"]** ( double ): mass of each nuclear DOF [ units: a.u. of mass ]

    Returns:       
        PyObject: obj, with the members:

            * obj.ham_dia ( CMATRIX(2,2) ): diabatic Hamiltonian 
            * obj.ovlp_dia ( CMATRIX(2,2) ): overlap of the basis (diabatic) states [ identity ]
            * obj.d1ham_dia ( list of 1 CMATRIX(2, 2) objects ): 
                derivatives of the diabatic Hamiltonian w.r.t. the nuclear coordinate
            * obj.dc1_dia ( list of 1 CMATRIX(2, 2) objects ): derivative coupling in the diabatic basis [ zero ]
    """
    
    Id = Cpp2Py(full_id)
    indx = Id[-1]
    X = q.col(indx)       # coordinates of all particles for this trajectory
    ndof = q.num_of_rows  # the total number of DOFs, the first one is the quantum DOF
    
    critical_params = [ ]
    default_params = {}
    T = 9.5e-4 # 300 K
    default_params.update({"V": 0.5*T, "E": 0.5*T, "lambda": 1.0*T, "omega": 0.1*T, "mass": 1.0})
    comn.check_input(params, default_params, critical_params)

    # Parameters describing the diabatic PES for quantum DOF
    V = params["V"] 
    E = params["E"] 
    L = params["lambda"]     
    w = params["omega"]
    m = params["mass"]

    nstates = 2 
    
    Hdia = CMATRIX(nstates,nstates)
    Sdia = CMATRIX(nstates,nstates)
    Sdia.identity()
    basis_transform = CMATRIX(nstates,nstates)
    basis_transform.identity()
    
    
    d1ham_dia = CMATRIXList();
    dc1_dia   = CMATRIXList();
    for k in range(ndof):
        d1ham_dia.append( CMATRIX(nstates,nstates) )
        dc1_dia.append( CMATRIX(nstates,nstates) ) 

    
    #========= Classical (bath) contributions =========    
    w_k = [w*math.tan( (k+0.5)/(2*ndof)*math.pi ) for k in range(ndof)] # mode frequencies from the Debye spectral density
    g_k = [w_k[k] * math.sqrt(2 * L / ndof) for k in range(ndof)] # coupling strengths
    
    H_bath = sum([0.5*m*w_k[k]**2 * X.get(k)**2 for k in range(ndof)])
    H_coup = sum([g_k[k] * X.get(k) for k in range(ndof)])

    dH_bath = [ m*w_k[k]**2 * X.get(k) for k in range(ndof)]
    dH_coup = list(g_k)

    Hdia.add(0,0, (H_bath + H_coup)*(1.0+0.0j)); Hdia.add(1,1, (H_bath - H_coup)*(1.0+0.0j))

    for k in range(ndof):
        d1ham_dia[k].add(0, 0, dH_bath[k] * (1.0+0.0j) ); d1ham_dia[k].add(1, 1, dH_bath[k] * (1.0+0.0j) )
        d1ham_dia[k].add(0, 0, dH_coup[k] * (1.0+0.0j) ); d1ham_dia[k].add(1, 1, -dH_coup[k] * (1.0+0.0j) )

    #========== Quantum system ==================    
    Hdia.add(0,0, E ); Hdia.add(1,1, -E )
    Hdia.add(0,1, V ); Hdia.add(1,0, V )

    obj = tmp()
    obj.ham_dia = Hdia
    obj.ovlp_dia = Sdia
    obj.d1ham_dia = d1ham_dia
    obj.dc1_dia = dc1_dia
    #obj.basis_transform = basis_transform
    
    return obj

