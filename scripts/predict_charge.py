'''
Script to predict charges using pre-trained gimlet model
currently the gimlet code can only read in sdf files, mdanalysis can read in anything but sdf
mdanalysis is only needed to load the molecules, assign the charges, and save it again

for now both files are needed, the mol2 file being created with obabel

'''

import os
import sys
import tensorflow as tf
# import tensorflow_probability as tfp
# tf.enable_eager_execution()
sys.path.append('../')
import gin
import lime
import numpy as np

from model.model import gin_model

import MDAnalysis as mda
from rdkit import Chem

from model.helper import get_q_total_per_mol, get_q_i_hat_total_per_mol
import model.train as training

print('Done import')


def predict_charges(input_file):
    # read molecules into a tf.data.Dataset
    ds_all = gin.i_o.from_sdf.to_ds(input_file)
    # by default, there is coordinates in dataset created from sdf
    # now we get rid of it
    ds_all = ds_all.map(lambda atoms, adjacency_map, coordinates, charges:\
        (atoms, adjacency_map, charges))
    # put them in batches
    BATCH_SIZE = 256
    ds_all = gin.probabilistic.gn.GraphNet.batch(ds_all, BATCH_SIZE, per_atom_attr=True)

    model = gin_model()
    model.gn.load_weights('../saved_model/model.checkpoint')


    charges = []

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, q_i, attr_in_mol in ds_all:

        Qs = get_q_total_per_mol(q_i, attr_in_mol)
        e, s = model.gn(atoms, adjacency_map, atom_in_mol, bond_in_mol, attr_in_mol)
            
        e = tf.boolean_mask(e, tf.reduce_any(attr_in_mol, axis=1))
        s = tf.boolean_mask(s, tf.reduce_any(attr_in_mol, axis=1))

        q_i_hat = get_q_i_hat_total_per_mol(e, s, Qs, attr_in_mol)
        
        charges.append(q_i_hat.numpy())

        
    
    return list(*charges)




if __name__ == "__main__":
    #input_file = sys.argv[1]
    input_file = 'hit.sdf'
    charges = predict_charges(input_file)
    input_file = 'hit.mol2'
    u = mda.Universe(input_file)
    u.atoms.charges = charges
    ag = u.select_atoms("all")
    ag.write('output.mol2')
    
        
        
        
        
        
        
        
    
    
    
