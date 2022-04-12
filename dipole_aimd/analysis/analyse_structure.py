"""Analyse the structure from an AIMD calculation."""
import json
from dataclasses import dataclass
from ase.db import connect
from collections import defaultdict
import numpy as np


@dataclass
class ParseZCoordinate:
    """Parses the ASE for the z-coordinate of a
    given atom in the structure. 
    Inputs
    ------
    ase_db_file: str
        The path to the ASE database file.
    output_file: str
        The path to the output ASE file.
    atom_type: str
        The atom type to be parsed.
    """
    ase_db_file: str
    output_file: str
    atom_type: str

    def __post_init__(self):
        print('Database parsed is: {}'.format(self.ase_db_file))
        print('The structure of the {} atom has been analysed.'.format(self.atom_type))
        self.atom_structures = defaultdict(list)
        self._atom_structures = defaultdict(list)
        self.parse_structures()
        self.sort_structures()
    
    def get_ase_database(self):
        """Read the ASE database and yield the dipole moment."""
        with connect(self.ase_db_file) as handle:
            for row in handle.select():
                    yield row.toatoms(), row.run_number, row.timestep, row.state
    
    def parse_structures(self):
        """Store the dipole moment and the structure in a dict of lists in sorted order of sampling."""
        for atoms, run_number, timestep, state in self.get_ase_database():
            # get the z-coordinate of the atom
            arg_atoms = [ a == self.atom_type for a in atoms.get_chemical_symbols() ]
            z_coordinate = atoms.positions[arg_atoms][:, 2]
            if len(z_coordinate) == 1:
                z_coordinate = z_coordinate[0]
            elif len(z_coordinate) == 0:
                z_coordinate = 0
            else:
                z_coordinate = list(z_coordinate) 
            self._atom_structures[state].append(np.array([run_number, timestep, z_coordinate]))
    
    def sort_structures(self):
        """Plot the dipole moment for the different structures in one graph based on ordered sampling."""
        # Sort the dipole moment based on the sampling for each structure
        for structure, atoms_data in self._atom_structures.items():
            atoms_data = np.array(atoms_data)
            sorted_index = np.lexsort((atoms_data[:, 0], atoms_data[:, 1]))
            store_atoms = atoms_data[sorted_index]
            # convert every part of array store_atoms to a list
            store_atoms = [ i.tolist() for i in store_atoms ]
            self.atom_structures[str(structure)].append(store_atoms)

        # Save the file as a json
        with open(self.output_file, 'w') as handle:
            json.dump(self.atom_structures, handle, indent=4)
