import warnings
from m3gnet.models import Relaxer
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure, Molecule
from pymatgen.transformations.advanced_transformations import *
import os
import csv
from pymatgen.core.operations import SymmOp
from pymatgen.alchemy.materials import TransformedStructure
import numpy as np
import itertools

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

"""
This code is used for obtaining m3gnet adsorption energies for labelling of molecules.
"""

def get_directories():
    molecules_dir = []
    for root, _, files in os.walk("./jsons"):
        for file in files:
            if file.endswith(".json"):
                cid =file.split("/")[-1][:-5]
                if not os.path.isdir(cid):
                    molecules_dir.append(os.path.join(root, file))
    return molecules_dir

molecules_dir = get_directories()
count = len(molecules_dir)
relaxer = Relaxer()

if not os.path.exists("m3gnet_adsorption_data.csv"):
    with open('m3gnet_adsorption_data.csv', 'a', newline='') as f:
        header = ["CID", "formula", "slab", "adsorbed formula", "Final energy (eV)", "Molecule energy (eV)", "Adsorption energy (eV)"]
        writer = csv.writer(f)
        writer.writerow(header)

### Get energies for pure Li slab
Li_cif = Structure.from_file("./Li_cif.cif")
slabs = []
for miller_index in [(1,0,0), (1,1,0), (1,1,1)]:
    if miller_index == (1,1,1):
        min_slab_size = 4
    else:
        min_slab_size = 7
    slab = SlabTransformation(miller_index=miller_index, min_slab_size=min_slab_size, min_vacuum_size=20, primitive=False, max_normal_search=1)
    Li_slab = slab.apply_transformation(Li_cif)
    relax_results = relaxer.relax(Li_slab)
    Li_energy = float(relax_results['trajectory'].energies[-1])
    Li_no_orig = Li_slab.composition["Li0+"]
    name = str(miller_index).replace(",","").replace(" ","")
    slabs.append((Li_slab, name, Li_energy, Li_no_orig))


### specify range of orientations to consider
angles = range(0, 360, 15)
symm100 = []
symm010 = []
for angle in angles:
    symm100.append(SymmOp.from_axis_angle_and_translation((1,0,0), angle))
    symm010.append(SymmOp.from_axis_angle_and_translation((0,1,0), angle))
sym_ops = list(itertools.product(symm100, symm010))

curr = 0
### Begin relaxation of molecule and put in on Li surfaces
for mol_file in molecules_dir:
    cid = mol_file.split("/")[-1][:-5]
    print(f"Starting calculation for CID:{cid}")
    ### relax molecule first
    molecule = Molecule.from_file(mol_file)
    #molecule = Molecule.from_sites(structure)
    molecule_box = molecule.get_boxed_structure(20,20,20)
    relax_results = relaxer.relax(molecule_box)
    final_mol_energy = float(relax_results['trajectory'].energies[-1])
    mol_comp = molecule.composition.formula.replace(" ","")
    lowest_structures = {"(100)":None, "(110)":None, "(111)":None}
    mol_data = []
    
    for Li_slab, slab_name, Li_energy, Li_no_orig in slabs:
        a = Li_slab.lattice.a
        b = Li_slab.lattice.b

        ### Get average z-coordinate of all atoms on a slab surface, then put it in a dict
        pq = {}
        for sym in sym_ops:
            copy_mol = molecule.copy()
            copy_mol.apply_operation(sym[0])
            copy_mol.apply_operation(sym[1])

            x_coords = [site.coords[0] for site in copy_mol.sites]
            x_len = max(x_coords) - min(x_coords)
            y_coords = [site.coords[1] for site in copy_mol.sites]
            y_len = max(y_coords) - min(y_coords)
            xrep = np.ceil((x_len+8) / a)
            yrep = np.ceil((y_len+8) / b)
            repeat = [xrep, yrep, 1]
     
            adsorption = AddAdsorbateTransformation(copy_mol, selective_dynamics=False, \
                reorient=False, translate=True, repeat=repeat)
            structures = adsorption.apply_transformation(Li_slab, return_ranked_list=100)

            z_coords_all = [site.coords[-1] for site in structures[0]["structure"].sites if site.species != "Li"]
            avg_z_all = np.mean(z_coords_all)

            pq[avg_z_all] = structures

        ### Dequeue top 10 orientations and relax the structures using M3GNet
        pq = sorted(list(pq.items()))
        for i in range(10):
            _, structures = pq.pop(0)
            for structure in structures:
                initial_structure = structure["structure"]
                relax_results = relaxer.relax(initial_structure)
                final_structure = relax_results['final_structure']
                total_energy = float(relax_results['trajectory'].energies[-1])
                adsorbed_comp = initial_structure.composition.formula.replace(" ","")
                no_Li = initial_structure.composition["Li0+"]
                adsorption_energy = total_energy - (Li_energy * no_Li / Li_no_orig + \
                    final_mol_energy)
                #print(adsorption_energy)
                mol_data.append([cid, mol_comp, slab_name, adsorbed_comp, total_energy, final_mol_energy, adsorption_energy])
                ### Save the lowest energy structures from relaxation
                if lowest_structures[slab_name] is not None:
                    if lowest_structures[slab_name][2] > adsorption_energy:
                        lowest_structures[slab_name] = (initial_structure, final_structure, adsorption_energy)
                else:
                    lowest_structures[slab_name] = (initial_structure, final_structure, adsorption_energy)
                    
    with open('m3gnet_adsorption_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(mol_data)
        
    os.system(f"mkdir {cid}")
    for slab, lowests in lowest_structures.items():
        initial_structure, final_structure, energy = lowests
        formula = initial_structure.composition.formula.replace(" ","")
        initial_structure.to("cif", f"{cid}/{formula}_{slab}_initial.cif", refine_struct=False)
        final_structure.to("cif", f"{cid}/{formula}_{slab}_relaxed.cif", refine_struct=False)

    curr += 1
    print(f"{curr}/{count} molecules done")
