

import os
import sys
import tensorflow as tf
# import tensorflow_probability as tfp
# tf.enable_eager_execution()
sys.path.append('../')
import gin
import lime
import pandas as pd
import numpy as np
from .helper import get_q_total_per_mol, get_q_i_hat_total_per_mol

# @tf.function
def train(gn, optimizer, ds_tr, N_EPOCHS=10):
    losses = tf.constant([-1], dtype=tf.float32)
    for dummy_idx in range(N_EPOCHS):
        print('Done '+str(dummy_idx)+' out of '+str(N_EPOCHS)+' epochs.')
        for atoms, adjacency_map, atom_in_mol, bond_in_mol, q_i, attr_in_mol in ds_tr:
            with tf.GradientTape() as tape:
                Qs = get_q_total_per_mol(q_i, attr_in_mol)
                
                e, s = gn(atoms, adjacency_map, atom_in_mol, bond_in_mol, attr_in_mol)
                
                e = tf.boolean_mask(e, tf.reduce_any(attr_in_mol,axis=1))
                s = tf.boolean_mask(s, tf.reduce_any(attr_in_mol, axis=1))

                q_i_hat = get_q_i_hat_total_per_mol(e, s, Qs, attr_in_mol)
                
                q_i = tf.boolean_mask(q_i, tf.reduce_any(attr_in_mol, axis=1))
                
                loss = tf.losses.mean_squared_error(q_i, q_i_hat)

            variables = gn.variables
            grad = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grad, variables))

            losses = tf.concat([losses,[loss],],axis=0)
            #gn._set_inputs(atoms, adjacency_map)
            gn.save_weights('model.checkpoint')
            
    return losses, gn