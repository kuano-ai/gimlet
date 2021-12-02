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


# define the dimension of nodes, edges, and global attributes
D_V = 32
D_E = 32
D_U = 32

class node_feat(tf.keras.Model):
    """ Featurization of nodes.
    Here we simply featurize atoms using one-hot encoding.
    
    """
    def __init__(self, units=D_V):
        super(node_feat, self).__init__()
        self.d = tf.keras.layers.Dense(units)

    @tf.function
    def call(self, x):
        x = tf.one_hot(x, 8)
        # set shape because Dense doesn't like variation
        x.set_shape([None, 8]) 
        return self.d(x)
        
class edge_feat(tf.keras.Model):
    """ Featurization of edges.
    Here we split the $\sigma$ and $\pi$ component of bonds
    into two channels, and featurize them seperately.
    
    """
    def __init__(
            self, 
            d_sigma_units=64, 
            d_pi_units=64,
            units=D_E):
        
        super(edge_feat, self).__init__()
        self.D_E = D_E
        
        # sigma
        self.d_sigma_0 = tf.Variable(
            tf.zeros(
                shape=(1, d_sigma_units),
                dtype=tf.float32))
        self.d_sigma_1 = tf.keras.layers.Dense(
            int(D_E // 2))
        
        # pi
        self.d_pi_0 = tf.keras.layers.Dense(
            d_pi_units)
        self.d_pi_1 = tf.keras.layers.Dense(
            int(D_E // 2))
        
    @tf.function
    def call(self, x):
        # determine whether there is $\pi$ component in the bond
        has_pi = tf.greater(
            x,
            tf.constant(1, dtype=tf.float32))
        
        # calculate the sigma component of the bond
        x_sigma = tf.tile(
            self.d_sigma_1(self.d_sigma_0),
            [tf.shape(x, tf.int64)[0], 1])
        
        # calculate the pi component of the bond
        x_pi = tf.where(
            has_pi,
            
            # if has pi:
            self.d_pi_1(
                self.d_pi_0(
                    tf.math.subtract(
                        x,
                        tf.constant(1, dtype=tf.float32)))),
            
            # else:
            tf.zeros(
                shape=(D_E // 2, ),
                dtype=tf.float32))
        
        
        x = tf.concat(
            [
                x_sigma,
                x_pi
            ],
            axis=1)
        
        
        return x
        
# f_u
global_feat=(lambda atoms, adjacency_map, batched_attr_mask: \
    tf.tile(
        tf.zeros((1, D_U)),
        [
             tf.math.count_nonzero(
                 tf.reduce_any(
                     batched_attr_mask,
                     axis=0)),
             1
        ]
    ))
    
# phi_v
update_node = lime.nets.for_gn.ConcatenateThenFullyConnect(
    (64, 'elu', 64, D_V))
    
# phi_e
update_edge = lime.nets.for_gn.ConcatenateThenFullyConnect(
    (64, 'elu', 64, D_E))
    
# phi_u
class update_global(tf.keras.Model):
    def __init__(self, config=(64, 'elu', D_U)):
        super(update_global, self).__init__()
        self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)

    @tf.function
    def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
        return self.d(h_u, h_u_0, h_e_bar, h_v_bar)
        
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
                        

         
    
class f_r(tf.keras.Model):
    """ Readout function.
    """
    
    def __init__(self, units=128):
        super(f_r, self).__init__()
        self.d_e_0 = tf.keras.layers.Dense(units)
        self.d_s_0 = tf.keras.layers.Dense(units)
        self.d_e_1 = tf.keras.layers.Dense(1)
        self.d_s_1 = tf.keras.layers.Dense(1)
    
    @tf.function
    def call(self,
            h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history,
            atom_in_mol, bond_in_mol):
        
        # although this could take many many arguments,
        # we only take $h_e$ for now
        e = self.d_e_1(self.d_e_0(h_v))
        s = self.d_s_1(self.d_s_0(h_v))
        
        return e, s
        
class gin_model():
    
    def __init__(self):
        self.gn = gin.probabilistic.gn.GraphNet(
            f_e=edge_feat(),
            f_v=node_feat(),
            f_u=global_feat,
            phi_e=update_edge,
            phi_v=update_node,
            phi_u=update_global(),
            f_r=f_r())
        
        