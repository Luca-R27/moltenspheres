from __future__ import print_function
import sys, json, os
import numpy as np
import ctypes

import random
import numpy as np

#################################################################################################################
# LAMMPS-Python script to use the molten spheres algorithm to perform athermal irradiation. 
# Script (mainly) written by Luca Reali (luca.reali@ukaea.uk), with lots of help by Max Boleininger, at UKAEA.
# February 2025

# The algorithm is described in L. Reali, M. Boleininger, D.R. Mason, S.L. Dudarev, Acta Materialia (2025),
# "Atomistic simulations of athermal irradiation creep and swelling of copper and tungsten at high dose"

# Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied. 

#################################################################################################################

def molten_sphere_coord(x_melt, y_melt, z_melt, N_sph, N_tot, R_target, density, x_lo, y_lo, z_lo, x_hi, y_hi, z_hi):
# This function carves a sphere of molten atoms and return the coordinates so that they fit in a desired radius
    R = (3*N_sph/(4*np.pi*density))**(1/3)
    x_c = 0.0
    y_c = 0.0
    z_c = 0.0
    ID = random.randint(1, N_tot)
    while ( (x_melt[ID-1]-x_lo) < R or (x_hi - x_melt[ID-1]) < R or (y_melt[ID-1]-y_lo) < R or (y_hi - y_melt[ID-1]) < R or (z_melt[ID-1]-z_lo) < R or (z_hi - z_melt[ID-1]) < R ):
        ID = random.randint(1, N_tot)
    # Create an array with x, y, z coordinates plus a fourth column of distance from the centre, to sort all the atoms and pick the N_sph that are closest
    molten_sphere = np.vstack((x_melt-x_melt[ID-1], y_melt-y_melt[ID-1], z_melt-z_melt[ID-1], (x_melt-x_melt[ID-1])**2+(y_melt-y_melt[ID-1])**2+(z_melt-z_melt[ID-1])**2)).T
    molten_sphere = molten_sphere[molten_sphere[:,-1].argsort()]
    molten_sphere = molten_sphere[:N_sph, :3]
    # Scale coordinates so that the atoms fill a sphere of radus R_target
    scaling_factor = R_target/np.sqrt(molten_sphere[-1, 0]**2+molten_sphere[-1, 1]**2+molten_sphere[-1, 2]**2)
    return scaling_factor*molten_sphere

def create_melt_centres(N_melt_per_step, xhi, xlo, yhi, ylo, zhi, zlo, R):
# This function return the location of the N_melt_per_step points where the spheres of molten atoms are to be inserted
    min_D_centres = 2*R + 15.0   # A minimum separation between sphere surfaces of 15 Angstrom is hardcoded here, but could be played with.
    melt_centres = np.zeros((N_melt_per_step, 3))
    melt_centres[0, :] = np.array([np.random.uniform(xlo+(R+7.5), xhi-(R+7.5)), np.random.uniform(ylo+(R+7.5), yhi-(R+7.5)), np.random.uniform(zlo+(R+7.5), zhi-(R+7.5))])

    for i in np.arange(1, N_melt_per_step):
        flag_retry = True
        while flag_retry:
            try_centre = np.array([np.random.uniform(xlo+(R+7.5), xhi-(R+7.5)), np.random.uniform(ylo+(R+7.5), yhi-(R+7.5)), np.random.uniform(zlo+(R+7.5), zhi-(R+7.5))])
            dist_centres_sq = melt_centres[:i, :] - try_centre # calculate in 2 steps squared distances between new and all previous
            dist_centres_sq = dist_centres_sq[:, 0]**2+dist_centres_sq[:, 1]**2+dist_centres_sq[:, 2]**2 
            if (np.size(dist_centres_sq[dist_centres_sq<min_D_centres**2]) == 0):
                melt_centres[i, :] = try_centre
                flag_retry = False
                
    return melt_centres


me = 0

from mpi4py import MPI
comm = MPI.COMM_WORLD
me = comm.Get_rank()
nprocs = comm.Get_size()

# load json
inputfile = sys.argv[1]

if (me == 0):
    with open(inputfile) as fp:
        input_data = json.loads(fp.read())
else:
    input_data = None
comm.barrier()

# broadcast imported data to all cores
input_data = comm.bcast(input_data, root=0)


# Set parameters
new_simulation = bool(input_data['new_simulation'])
simul_name = input_data['simul_name']
dump_every = int(input_data["dump_every"]) 

potfile = input_data['potential_file']
element = input_data['element']         # only single element so far 

# algorithm specific quantities
molten_coordinates = input_data["moltenatoms"] # pre-computed molten configuration
R = float(input_data["R"]) # single value here, it can be substituted by a distribution
N_melt_per_step = int(input_data["N_melt_per_step"]) # How many spheres per step
N_ins = int(input_data["N_insertions"]) # How many sphere insertion steps

etol = float(input_data["etol"])
etolstring = "%.5e" % etol

# enforced external pressure. Ensure correct units and mind the minus sign (stress vs pressure tensors, see LAMMPS documentation).
sxx = float(input_data["sxx"])
syy = float(input_data["syy"])
szz = float(input_data["szz"])
sxy = float(input_data["sxy"])
syz = float(input_data["syz"])
sxz = float(input_data["sxz"])


# input for lattice definition
cryst_struct = input_data['cryst_struct'] # only fcc or bcc are supported
latt_param = input_data['latt_param']
tri = input_data['is_triclinic']
nx = input_data['nx']
ny = input_data['ny']
nz = input_data['nz']

if new_simulation :
        tri = bool(input_data['is_triclinic'])
else :
        with open(simul_name+'_last.dump', 'r') as filedump:
                filedump.readline()
                filedump.readline()
                filedump.readline()
                filedump.readline()
                string = filedump.readline()
                if "xy xz yz" in string:
                        tri = True
                else:
                        tri = False

# Load the coordinates of molten atoms
if (me == 0):
        N_molten = int(np.loadtxt(molten_coordinates, skiprows=3, delimiter=' ', max_rows=1))
        melt_box = np.loadtxt(molten_coordinates, skiprows=5, delimiter=' ', max_rows=3) # first row (x_min, x_max) second row (y_min, y_max) third row (z_min, z_max)
        coords_melt = np.loadtxt(molten_coordinates, skiprows=9, delimiter=' ')
        x_melt = (melt_box[0,1]-melt_box[0,0])*coords_melt[:, 2]
        y_melt = (melt_box[1,1]-melt_box[1,0])*coords_melt[:, 3]
        z_melt = (melt_box[2,1]-melt_box[2,0])*coords_melt[:, 4]
        density = N_molten/(melt_box[0,1]-melt_box[0,0])/(melt_box[1,1]-melt_box[1,0])/(melt_box[2,1]-melt_box[2,0])
else:
        x_melt = np.array([0.0, 0.0])
        y_melt = np.array([0.0, 0.0])
        z_melt = np.array([0.0, 0.0])
        density = 0.0
comm.barrier()
x_melt = comm.bcast(x_melt, root=0)
y_melt = comm.bcast(y_melt, root=0)
z_melt = comm.bcast(z_melt, root=0)
density = comm.bcast(density, root=0)
comm.barrier()


from lammps import lammps
lmp = lammps()
lmp.command("log none")

# initialise the LAMMPS simulation
lmp.command("dimension       3")
lmp.command("boundary        p p p")
lmp.command("units           metal")
lmp.command("atom_style      atomic")

# fcc/bcc and triclinic simulations are assumed here.
lmp.command("lattice         %s %s" % (cryst_struct, latt_param))
if tri:
        lmp.command("region          prsm prism 0 %d 0 %d 0 %d 0.0 0.0 0.0" % (nx, ny, nz))
        lmp.command("create_box      1 prsm")  # Number of atom types to be used
else:
        lmp.command("region          cuboid block 0 %d 0 %d 0 %d" % (nx, ny, nz))
        lmp.command("create_box      1 cuboid")  # Number of atom types to be used

lmp.command("create_atoms    1 box")


# Define potential for use in simulation
lmp.command("pair_style      eam/fs")
lmp.command("pair_coeff      * * %s %s" % (potfile, element))


lmp.command("compute ID all pe/atom")
lmp.command("thermo 200")
lmp.command("thermo_style custom step press lx ly lz xy yz xz pe pxx pyy pzz pxy pxz pyz")
lmp.command("fix recentre all recenter INIT INIT INIT")

comm.barrier()


# If new simulation relax box under stress and dump initial info, if restarting then load last dump file
if new_simulation :
        if tri:
                lmp.command('fix rel all box/relax x %d y %d z %d xy %d yz %d xz %d couple none vmax 0.0005' % (sxx, syy, szz, sxy, syz, sxz))
        else:
                lmp.command('fix rel all box/relax x %d y %d z %d couple none vmax 0.0005' % (sxx, syy, szz))
        lmp.command('minimize %s 0.0 100000 100000' % (etolstring))
        lmp.command('unfix rel')
        i_start = 0

        lmp.command("write_dump all atom %s" % ( simul_name+".0.dump") )

        # for convenience print the full dat file and also one that will match the dumped files. One is simply a subset of the other.
        if tri:
                lmp.command("print 'step lx ly lz xy yz xz pxx pyy pzz pxy pxz pyz pe' file %s" % (simul_name+".dat") )
                lmp.command("print '0 $(lx) $(ly) $(lz) $(xy) $(yz) $(xz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+".dat")  )
                lmp.command("print 'step lx ly lz xy yz xz pxx pyy pzz pxy pxz pyz pe' file %s" % (simul_name+"_full.dat") )
                lmp.command("print '0 $(lx) $(ly) $(lz) $(xy) $(yz) $(xz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+"_full.dat")  )        
        else:
                lmp.command("print 'step lx ly lz pxx pyy pzz pxy pxz pyz pe' file %s" % (simul_name+".dat") )
                lmp.command("print '0 $(lx) $(ly) $(lz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+".dat")  )
                lmp.command("print 'step lx ly lz pxx pyy pzz pxy pxz pyz pe' file %s" % (simul_name+"_full.dat") )
                lmp.command("print '0 $(lx) $(ly) $(lz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+"_full.dat")  )        
else :
        # for the time being, support only reading from a dump file (so that timestep can be read from the second line). Also assume there's a full dat file where to get the last step of the simulation to restart.
        i_start = int(np.loadtxt(simul_name+'_full.dat', delimiter=' ', skiprows=1)[-1, 0])
        timestep_last = str(int(np.loadtxt(simul_name+'_last.dump', skiprows=1, delimiter=' ', max_rows=1)))
        lmp.command("delete_atoms group all")
        lmp.command("read_dump %s %s x y z box yes add yes" % (simul_name+'_last.dump', timestep_last) )
comm.barrier()


# here perform the subsequent molten spheres insertions
for i in np.arange(i_start, N_ins):

        lmp.command("variable i equal %s" % (str(i+1))) # LAMMPS variable for the filename

        # Shift the atoms by a random distance to simplify the insertion steps without biasing against the edges, then will shift back
        if me == 0:
                print("************* STEP "+str(i+1)+" out of "+str(N_ins)+". *******************")
                # get box dimensions to have lower and upper bounds for the centres of the spheres 
                box = lmp.extract_box()
                xlo, ylo, zlo = box[0]
                xhi, yhi, zhi = box[1]
                xy, yz, xz = box[2], box[3], box[4]
                x_shift = np.random.uniform(0.0, (xhi-xlo))
                y_shift = np.random.uniform(0.0, (yhi-ylo))
                z_shift = np.random.uniform(0.0, (zhi-zlo))
                # create the array with the centres of the spheres
                melt_centres = create_melt_centres(N_melt_per_step, xhi, xlo, yhi, ylo, zhi, zlo, R)
                x_tot = np.array([])
                y_tot = np.array([])
                z_tot = np.array([])
                sys.stdout.flush()
        else:
                x_shift = 0.0
                y_shift = 0.0
                z_shift = 0.0
                melt_centres = np.array([0.0, 0.0 ,0.0])
        comm.barrier()
        x_shift = comm.bcast(x_shift, root=0)
        y_shift = comm.bcast(y_shift, root=0)
        z_shift = comm.bcast(z_shift, root=0)
        melt_centres= comm.bcast(melt_centres, root=0)
        comm.barrier()
        lmp.command("displace_atoms all move %f %f %f units lattice" % (x_shift, y_shift, z_shift) )



        # Here substitute the atoms in the simulation by molten atoms
        for k in np.arange(N_melt_per_step):
                if (me == 0):
                        x_melt_k = melt_centres[k, 0]
                        y_melt_k = melt_centres[k, 1]
                        z_melt_k = melt_centres[k, 2]
                else:
                        x_melt_k = 0.0
                        y_melt_k = 0.0
                        z_melt_k = 0.0
                comm.barrier()
                x_melt_k = comm.bcast(x_melt_k, root=0)
                y_melt_k = comm.bcast(y_melt_k, root=0)
                z_melt_k = comm.bcast(z_melt_k, root=0)
                comm.barrier()
                lmp.command("region melt sphere %f %f %f %f units box" % (x_melt_k, y_melt_k, z_melt_k, R))
                # Find how many atoms in the spherical region...
                lmp.command("variable N_inside equal count(all,melt)")
                N_inside = int(lmp.extract_variable("N_inside"))
                # ... delete them ...
                lmp.command("delete_atoms region melt")
                lmp.command("region melt delete")
                # and replace them.
                if (me == 0):
                        melt_sph_coord = molten_sphere_coord(x_melt, y_melt, z_melt, N_inside, N_molten, R, density, melt_box[0,0], melt_box[1,0], melt_box[2,0], melt_box[0,1], melt_box[1,1], melt_box[2,1])
                        x_tot = np.append(x_tot, melt_sph_coord[:, 0]+x_melt_k)
                        y_tot = np.append(y_tot, melt_sph_coord[:, 1]+y_melt_k)
                        z_tot = np.append(z_tot, melt_sph_coord[:, 2]+z_melt_k)

                comm.barrier()
                

        if (me == 0):
                # use an auxiliary file to read in all the molten atoms at once for efficiency.
                arr = np.ndarray((np.size(x_tot), 5),dtype = object)
                arr[:, 0] = np.arange(1, np.size(x_tot)+1, dtype=np.uint32)
                arr[:, 1] = np.ones(np.size(x_tot), dtype=np.uint8)
                arr[:, 2] = x_tot
                arr[:, 3] = y_tot
                arr[:, 4] = z_tot
                with open(simul_name+'_auxil.txt', 'w') as f:
                        if tri :
                                f.write("# molten sphere \n \n"+str(np.size(x_tot))+" atoms\n1 atom types\n\n"+str(xlo)+" "+str(xhi)+" xlo xhi\n"+str(ylo)+" "+str(yhi)+" ylo yhi\n"+str(zlo)+" "+str(zhi)+" zlo zhi\n"+str(xy)+" "+str(xz)+" "+str(yz)+" xy xz yz\n \nAtoms\n \n")
                        else :
                                f.write("# molten sphere \n \n"+str(np.size(x_tot))+" atoms\n1 atom types\n\n"+str(xlo)+" "+str(xhi)+" xlo xhi\n"+str(ylo)+" "+str(yhi)+" ylo yhi\n"+str(zlo)+" "+str(zhi)+" zlo zhi \n \nAtoms\n \n")
                        np.savetxt(f, arr, delimiter=' ', fmt='%s')

        comm.barrier()
        lmp.command("read_data %s add append" % (simul_name+'_auxil.txt') )

        # First minimisation to cool down the melts
        lmp.command('minimize %s 0 100000 100000' % (etolstring))

        # Second minimisation to relax the box
        if tri:
                lmp.command('fix free all box/relax x %d y %d z %d xy %d yz %d xz %d couple none vmax 0.0005' % (sxx, syy, szz, sxy, syz, sxz))
        else:
                lmp.command('fix free all box/relax x %d y %d z %d couple none vmax 0.0005' % (sxx, syy, szz))

        #lmp.command('fix free all box/relax x %d y %d z %d xy %d yz %d xz %d couple none vmax 0.0005' % (sxx, syy, szz, sxy, syz, sxz))
        lmp.command('minimize %s 0 100000 100000' % (etolstring))
        lmp.command('unfix free')


        # this is needed to wrap atoms back into the box
        lmp.command('run 0 post no')

        # Shift the atoms back in order not to have a moving crystal
        lmp.command("displace_atoms all move %f %f %f units lattice" % (-x_shift, -y_shift, -z_shift) )
        comm.barrier()

        if tri:
                lmp.command("print '$(v_i) $(lx) $(ly) $(lz) $(xy) $(yz) $(xz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+"_full.dat")  )
        else:
                lmp.command("print '$(v_i) $(lx) $(ly) $(lz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+"_full.dat")  )

        lmp.command("write_dump all atom %s" % (simul_name+'_last.dump'))
        # Save info every so many steps 
        if ((i+1)%dump_every==0):
                lmp.command("write_dump all atom %s" % (simul_name+"."+str(i+1)+".dump") )
                if tri:
                        lmp.command("print '$(v_i) $(lx) $(ly) $(lz) $(xy) $(yz) $(xz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+".dat")  )
                else:
                        lmp.command("print '$(v_i) $(lx) $(ly) $(lz) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe)' append %s" % (simul_name+".dat")  )

