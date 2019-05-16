"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center
and Authors

Authors:
Yuanqing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# =============================================================================
# imports
# =============================================================================
import tensorflow as tf
tf.enable_eager_execution

import gin.molecule

# =============================================================================
# utility classes
# =============================================================================
class GCN(tf.keras.Model):
    """ A group of functions trainable under back-propagation to update atoms
    and molecules based on their neighbors and global attributes.

    Structures adopted from:
    arXiv:1806.01261v3

    Attributes
    ----------
    phi_e : function,
        applied per edge, with arguments $(e_k, v_r_k, v_s_k, u)$, and returns
        $e_k'$.
    rho_e_v : function,
        applied to $E'_i$, and aggregates the edge updates updates for edges
        that project to vertex $i$, into $\bar{e_i'}, which will be used in the
        next step's node update.
    phi_v : function,
        applied to each node $i$, to compute an updated node attribute, $v_i'$,.
    rho_e_u : function.
        applied to $E'$, and aggregates all edge updates, into $\bar{e'}$,
        which will then be used in the next step's global update.
    rho_v_u : function,
        applied to $V'$, and aggregates all node updates, into $\bar{v'}$,
        which will then be used in the next step's global update.
    phi_u : function,
        applied once per graph, and computes and update for the global
        attribute, $u'$.

    """

    def __init__(
            self,

            # building blocks for GCN
            phi_e=lambda x: x,
            rho_e_v=lambda x: x,
            phi_v=lambda x: x,
            rho_e_u=lambda x: x,
            rho_v_u=lambda x: x,
            phi_u=lambda x: x,

            # readout phase
            f_r=lambda x:x,

            # featurization
            f_e=lambda x:x,
            f_v=lambda x:x,
            f_u=lambda x:x,

            # the shape of the target
            y_bar_shape = tf.TensorShape([None, ]),

            repeat=3):

        super(GCN, self).__init__()
        self.phi_e = phi_e
        self.rho_e_v = rho_e_v
        self.phi_v = phi_v
        self.rho_e_u = rho_e_u
        self.phi_u = phi_u
        self.f_r = f_r
        self.f_e = f_e
        self.f_v = f_v
        self.f_u = f_u
        self.repeat=repeat

    @tf.contrib.eager.defun
    def propagate(
            self,
            molecules, # note that the molecules here could be featurized
            repeat=3):

        """ Propagate between nodes and edges.

        Parameters
        ----------
        molecules : a list of molecules to be
        """
        with tf.init_scope():
            # get the number of the molecules
            n_molecules = len(molecules)

        # define the function needed for propagate one molecule
        def propagate_one_molecule(y_bar_all, idx, molecules=molecules):
            # get the specific molecule
            mol = molecules[idx]

            # get the attributes of the molecule
            adjacency_map = mol.adjacency_map
            atoms = mol.atoms

            n_atoms = tf.cast(tf.shape(atoms)[0], tf.int64)

            adjacency_map_full = adjacency_map \
                + tf.transpose(adjacency_map)

            # dirty stuff to get the bond indices to update
            all_idxs_x, all_idxs_y = tf.meshgrid(
                tf.range(n_atoms, dtype=tf.int64),
                tf.range(n_atoms, dtype=tf.int64))

            # (n_atoms, n_atoms, 2)
            all_idxs_stack = tf.stack(
                [
                    all_idxs_y,
                    all_idxs_x
                ],
                axis=2)

            # (n_atoms, n_atoms, 2) # boolean
            is_bond = tf.greater(
                adjacency_map,
                tf.constant(0, dtype=tf.int64))

            # (n_bonds, 2)
            bond_idxs = tf.boolean_mask(
                all_idxs_stack,
                is_bond)

            n_bonds = tf.cast(tf.shape(bond_idxs)[0], tf.int64)

            # (n_bonds, n_atoms)
            bond_is_connected_to_atoms = tf.logical_or(
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0)
                        [n_bonds, 1]),
                    tf.tile(
                        tf.expand_dims(
                            bond_idxs[:,0],
                            1),
                        [1, n_atoms])),

                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0)
                        [n_bonds, 1]),
                    tf.tile(
                        tf.expand_dims(
                            bond_idxs[:,1],
                            1),
                        [1, n_atoms])))

            # (n_atoms, n_bonds)
            atom_is_connected_to_bonds = tf.transpose(
                bond_is_connected_to_atoms)

            # (n_bonds, )
            bond_orders = tf.gather_nd(
                adjacency_map,
                bond_idxs)

            # initialize the hidden layers
            # (n_bonds, ...)
            h_e = self.f_e(bond_orders)

            # (n_atoms, ...)
            h_v = self.f_v(atoms)

            # (...)
            h_u = self.f_u(atoms, adjacency_map)

            def propagate_one_time(h_e, h_v, h_u, iter_idx):
                # update $ e'_k $
                # $$
                # e'_k = \phi^e (e_k, v_{rk}, v_{sk}, u)
                # $$

                h_left = tf.gather(
                    h_v,
                    bond_idxs[:, 0])

                h_right = tf.gather(
                    h_v,
                    bond_idxs[:, 1])

                h_e = self.phi_e(h_e, h_left, h_right, h_u)

                # aggregate $ \bar{e_i'} $
                # $$
                # \bar{e_i'} = \rho^{e \rightarrow v} (E'_i)
                # $$

                # (n_atoms, ...)
                h_e_bar_i = self.rho_e_v(h_e, atom_is_connected_to_bonds)

                # update $ v'_i $
                # $$
                # v'_i = phi^v (\bar{e_i}, v_i, u)
                # $$

                # (n_atoms, ...)
                h_v = self.phi_v(h_e_bar, h_v, h_u)

                # aggregate $ \bar{e'} $
                # $$
                # \bar{e'} = \rhp^{e \rightarrow u} (E')
                # $$

                # (...)
                h_e_bar = self.rho_e_u(h_e)

                # aggregate $ \bar{v'} $
                # $$
                # \bar{v'} = \rho^{v \rightarrow u} (V')
                # $$

                # (n_bonds, ...)
                h_v_bar = self.rho_v_u(h_v)

                # update $ u' $
                # $$
                # u' = \phi^u (\bar{e'}, \bar{v'}, u)
                # $$

                # (...)
                h_u = phi_u(h_e_bar, h_v_bar, h_u)

                return h_e, h_v, h_u, iter_idx + 1

            # use while loop to execute the graph multiple times
            iter_idx = tf.constant(0, dtype=tf.int64)

            h_e, h_v, h_u, iter_idx = tf.while(
                # condition
                tf.less(iter_idx, self.repeat),

                # loop body
                propagate_one_time,

                # loop vars
                [h_e, h_v, h_u],

                # shape_invariants
                shape_invariants = [
                    h_e.get_shape(),
                    h_v.get_shape(),
                    h_u.get_shape(),
                    iter_idx.get_shape()])

            y_bar = self.f_r(h_e, h_v, h_u)

            y_bar_all = tf.concat(
                [
                    y_bar_all,
                    tf.expand_dims(
                        y_bar,
                        0)
                ],
                axis=0)

            return y_bar_all, idx + 1


        # while loop
        idx = tf.constant(0, dtype=tf.int64)
        y_bar_all = tf.expand_dims(
            tf.zeros(y_bar_shape, dtype=tf.float32))

        y_bar_all, idx = tf.while_loop(
            # while idx < n_molecules
            tf.less(
                idx,
                tf.cast(
                    n_molecules,
                    tf.int64)),

            # loop body
            propagate_one_molecule,

            # loop var
            [y_bar, idx],

            # shape_invariants
            shape_invariants = [
                y_bar_all.get_shape(),
                idx.get_shape()])

        # get rid of the first placeholder
        # (n_molecules, )
        y_bar_all = y_bar_all[1:]

        return y_bar_all