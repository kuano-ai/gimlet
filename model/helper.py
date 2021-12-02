import os
import sys
import tensorflow as tf
# import tensorflow_probability as tfp
# tf.enable_eager_execution()
sys.path.append('../gimlet')
import gin
import lime
import pandas as pd
import numpy as np


@tf.function
def get_charges(e, s, Q):
    """ Solve the function to get the absolute charges of atoms in a
    molecule from parameters.

    Parameters
    ----------
    e : tf.Tensor, dtype = tf.float32, shape = (34, ),
    electronegativity.
    s : tf.Tensor, dtype = tf.float32, shape = (34, ),
    hardness.
    Q : tf.Tensor, dtype = tf.float32, shape=(),
    total charge of a molecule.

    We use Lagrange multipliers to analytically give the solution.

    $$

    U({\bf q})
    &= \sum_{i=1}^N \left[ e_i q_i +  \frac{1}{2}  s_i q_i^2\right]
    - \lambda \, \left( \sum_{j=1}^N q_j - Q \right) \\
    &= \sum_{i=1}^N \left[
    (e_i - \lambda) q_i +  \frac{1}{2}  s_i q_i^2 \right
    ] + Q

    $$

    This gives us:

    $$

    q_i^*
    &= - e_i s_i^{-1}
    + \lambda s_i^{-1} \\
    &= - e_i s_i^{-1}
    + s_i^{-1} \frac{
    Q +
     \sum\limits_{i=1}^N e_i \, s_i^{-1}
    }{\sum\limits_{j=1}^N s_j^{-1}}

    $$

    """

    return tf.math.add(
    tf.math.multiply(
        tf.math.negative(
            e),
        tf.math.pow(
            s,
            -1)),

    tf.math.multiply(
        tf.math.pow(
            s,
            -1),
        tf.math.divide(
            tf.math.add(
                Q,
                tf.reduce_sum(
                    tf.math.multiply(
                        e,
                        tf.math.pow(
                            s,
                            -1)))),
            tf.reduce_sum(
                tf.math.pow(
                    s,
                    -1)))))
                

 


@tf.function
def get_q_i_hat_total_per_mol(e, s, Qs, attr_in_mol):
    """ Calculate the charges per molecule based on 
    `attr_in_mol`.
    
    """
    attr_in_mol.set_shape([None, None])
    
    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=1),
        axis=0)

    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=0),
    axis=1)

    q_i = tf.tile(
        tf.expand_dims(
            tf.constant(
                0,
                dtype=tf.float32),
            0),
        [tf.shape(attr_in_mol, tf.int64)[0]])
    
    def loop_body(q_i, idx, 
            e=e, 
            s=s, 
            Qs=Qs, 
            attr_in_mol=attr_in_mol):
        
        # get attr
        _attr_in_mol = attr_in_mol[:, idx]
        
        # get the attributes of each molecule
        _Qs = Qs[idx]
        
        _e = tf.boolean_mask(
            e,
            _attr_in_mol)
        
        _s = tf.boolean_mask(
            s,
            _attr_in_mol)
        
        _idxs = tf.where(_attr_in_mol)
        
        # update
        q_i = tf.tensor_scatter_nd_update(
            q_i,
        
            # idxs
            _idxs,
        
            # update
            tf.reshape(
                    get_charges(
                        _e,
                        _s,
                        _Qs),
                [-1]))
        
        return q_i, tf.add(idx, tf.constant(1, dtype=tf.int64))
    
    idx = tf.constant(0, dtype=tf.int64)
    
    # loop_body(q_i, idx)
    
    
    q_i, idx = tf.while_loop(
        lambda _, idx: tf.less(
            idx,
            tf.shape(attr_in_mol, tf.int64)[1]),
    
        loop_body,
        
        [q_i, idx])
    
    
    return q_i

@tf.function
def get_q_total_per_mol(q_i, attr_in_mol):
    # attr_in_mol.set_shape([None, None])
    
    q_i = tf.boolean_mask(
        q_i,
        tf.reduce_any(
            attr_in_mol,
            axis=1))
    
    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=1),
        axis=0)

    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=0),
    axis=1)
    
    attr_in_mol = tf.where(
        attr_in_mol,
    
        tf.ones_like(
            attr_in_mol,
            dtype=tf.float32),
    
        tf.zeros_like(
            attr_in_mol,
            dtype=tf.float32))
    
    q_per_mol = tf.reduce_sum(
        tf.multiply(
            attr_in_mol,
            tf.tile(
                tf.expand_dims(
                        q_i,
                        1),
                [
                    1,
                    tf.shape(attr_in_mol, tf.int64)[1]
                ])),
        axis=0)
    
    return q_per_mol
    
    