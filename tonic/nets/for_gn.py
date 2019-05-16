"""
for_gn.py

Here we present the common featurization functions, initialization functions,
and update and aggregation functions for graph nets.

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

# ===========================================================================
# imports
# ===========================================================================
import tensorflow as tf
tf.enable_eager_execution()

# ===========================================================================
# utility classes
# ===========================================================================
class ConcatThenFullyConnect(tf.keras.Model):
    """ Project all the input to the same dimension and then concat, followed
    by subsequent fully connected layers.

    """
    def __init__(self, config):
        super(ProjectToSameDimensionThenConcat, self).__init__()
        self.config = config
        self.flow = flow
        self.is_virgin = True

    def build(self, n_vars):
        """ Build the network.

        Note that this function is called when the first data point is
        passed on.

        Parameters
        ----------
        n_vars : int,
            number of variables taken in this layer.

        """
        # concatenation step
        for idx in range(n_vars):
            # put the layer as attribute of the model
            setattr(
                self,
                'D_0_%s' % idx,
                tf.keras.layers.Dense(config[0]))

            # NOTE:
            # here we don't put the name into the workflow
            # since we already explicitly expressed the flow
            # self.flow.append('D_0_%s' % idx)

        # the rest of the flow
        for idx in range(1, len(config)):
            if isinstance(config[idx], int):
                # put the layer as attribute of the model
                setattr(
                    self,
                    'D_%s' % idx,
                    tf.keras.layers.Dense(config[idx]))

                # put the name into the workflow
                self.flow.append('D_%s' % idx)

            elif isinstance(config[idx], float):
                assert config[idx] < 1

                # put the layer as attribute of the model
                setattr(
                    self,
                    'O_%s' % idx,
                    tf.keras.layers.Dense(config[idx]))

                # put the name into the workflow
                self.flow.append('O_%s' % idx)

            elif isinstance(config[idx], str):
                # put the layer as attribute of the model
                activation = config[idx]

                if activation == 'tanh':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.tanh)

                elif activation == 'relu':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.relu)

                elif activation == 'sigmoid':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.sigmoid)

                elif activation == 'leaky_relu':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.leaky_relu)

                elif activation == 'elu':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.elu)

                elif activation == 'softmax':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.softmax)

                self.flow.append('A_%s' % idx)

    @tf.contrib.eager.defun
    def _call(self, *args):
        """ The function to be compiled into TensorFlow graph computation.
        """
        x = tf.concat(
            # list of the projected first layer input
            [getattr(self, 'D_0_%s' % idx)(args[idx]) for idx in range(n_vars)],
            axis=-1) # note that here we concat it at the last dimension

        for fn in self.flow:
            x = getattr(self, fn)(x)

        return x

    def call(self, *args):
        """ The wrapper function for __call__.
        """
        # build the graph if this is the first time this is called
        if self.is_virgin:
            n_vars = len(args)
            self.build(n_vars)
            self.is_virgin = False

        return self._call(x)