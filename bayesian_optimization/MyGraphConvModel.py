from deepchem.models import GraphConvModel
from deepchem.models.graph_models import _GraphConvKerasModel, TrimGraphOutput
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from typing import List, Union, Tuple, Iterable, Dict, Optional
from collections.abc import Sequence as SequenceCollection
from deepchem.models import KerasModel, layers
import tensorflow as tf
import deepchem as dc

class _MyGraphConvKerasModel(_GraphConvKerasModel):
    """
    Private class in deepchem redefined to use more than 1 dense layer
    """
    def __init__(self,
               n_tasks,
               graph_conv_layers= [128, 128],
               dense_layers = [64, 64],
               dropout=0.01,
               mode="regression",
               number_atom_features=75,
               batch_normalize=True,
               uncertainty=True,
               batch_size=100):

        super(_GraphConvKerasModel, self).__init__()
        if mode not in ['classification', 'regression']:
          raise ValueError("mode must be either 'classification' or 'regression'")
    
        self.mode = mode
        self.uncertainty = uncertainty
    
        if not isinstance(dropout, SequenceCollection):
          dropout = [dropout] * (len(graph_conv_layers) + len(dense_layers))
        if len(dropout) != len(graph_conv_layers) + len(dense_layers):
          raise ValueError('Wrong number of dropout probabilities provided')
        if uncertainty:
          if mode != "regression":
            raise ValueError("Uncertainty is only supported in regression mode")
          if any(d == 0.0 for d in dropout):
            raise ValueError(
                'Dropout must be included in every layer to predict uncertainty')
        
        self.graph_convs = [
            layers.GraphConv(layer_size, activation_fn=tf.nn.relu)
            for layer_size in graph_conv_layers
        ]
        self.dense_layers = [
            Dense(layer_size, activation = tf.nn.relu)
            for layer_size in dense_layers
        ]
        self.batch_norms = [
            BatchNormalization(fused=False) if batch_normalize else None
            for _ in range(len(graph_conv_layers) + len(dense_layers))
        ]
        self.dropouts = [
            Dropout(rate=rate) if rate > 0.0 else None for rate in dropout
        ]
        self.graph_pools = [layers.GraphPool() for _ in graph_conv_layers]
        self.graph_gather = layers.GraphGather(batch_size=batch_size,
                                               activation_fn=tf.nn.tanh)
        self.trim = TrimGraphOutput()
        self.regression_dense = Dense(n_tasks)
        if self.uncertainty:
            self.uncertainty_dense = Dense(n_tasks)
            self.uncertainty_trim = TrimGraphOutput()
            self.uncertainty_activation = Activation(tf.exp)

    def call(self, inputs, training=False):
        atom_features = inputs[0]
        degree_slice = tf.cast(inputs[1], dtype=tf.int32)
        membership = tf.cast(inputs[2], dtype=tf.int32)
        n_samples = tf.cast(inputs[3], dtype=tf.int32)
        deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]
    
        in_layer = atom_features
        for i in range(len(self.graph_convs)):
            gc_in = [in_layer, degree_slice, membership] + deg_adjs
            gc1 = self.graph_convs[i](gc_in)
            if self.batch_norms[i] is not None:
                gc1 = self.batch_norms[i](gc1, training=training)
            if training and self.dropouts[i] is not None:
                gc1 = self.dropouts[i](gc1, training=training)
            gp_in = [gc1, degree_slice, membership] + deg_adjs
            in_layer = self.graph_pools[i](gp_in)
        
        start = len(self.graph_convs)
        end = len(self.dense_layers) + len(self.graph_convs)
        for i in range(start, end): # cont index for batchnorms and dropouts
            dense = self.dense_layers[i-start](in_layer)
            if self.batch_norms[i] is not None:
                dense = self.batch_norms[i](dense, training = training)
            if training and self.dropouts[i] is not None:
                dense = self.dropouts[i](dense, training = training)
            in_layer = dense
        
        neural_fingerprint = self.graph_gather([dense, degree_slice, membership] +
                                               deg_adjs)

        output = self.regression_dense(neural_fingerprint)
        output = self.trim([output, n_samples])
        if self.uncertainty:
            log_var = self.uncertainty_dense(neural_fingerprint)
            log_var = self.uncertainty_trim([log_var, n_samples])
            var = self.uncertainty_activation(log_var)
            outputs = [output, var, output, log_var, neural_fingerprint]
        else:
            outputs = [output, neural_fingerprint]
    
        return outputs
    
class MyGraphConvModel(GraphConvModel):
    """
    GraphConvModel redefined to use more than 1 dense layer
    """
  
    def __init__(self,
                 n_tasks: int,
                 graph_conv_layers: List[int] = [128, 128],
                 dense_layers: List[int] = [64, 64],
                 dropout: float = 0.01,
                 mode: str = "regression",
                 number_atom_features: int = 75,
                 batch_size: int = 100,
                 batch_normalize: bool = True,
                 uncertainty: bool = True,
                 **kwargs):

        self.mode = mode
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.uncertainty = uncertainty
        model = _MyGraphConvKerasModel(n_tasks,
                                     graph_conv_layers=graph_conv_layers,
                                     dense_layers=dense_layers,
                                     dropout=dropout,
                                     mode=mode,
                                     number_atom_features=number_atom_features,
                                     batch_normalize=batch_normalize,
                                     uncertainty=uncertainty,
                                     batch_size=batch_size)
        if mode == "classification":
          output_types = ['prediction', 'loss', 'embedding']
          loss: Union[Loss, LossFn] = SoftmaxCrossEntropy()
        else:
          if self.uncertainty:
            output_types = ['prediction', 'variance', 'loss', 'loss', 'embedding']
    
            def loss(outputs, labels, weights):
              output, labels = dc.models.losses._make_tf_shapes_consistent(
                  outputs[0], labels[0])
              output, labels = dc.models.losses._ensure_float(output, labels)
              losses = tf.square(output - labels) / tf.exp(outputs[1]) + outputs[1]
              w = weights[0]
              if len(w.shape) < len(losses.shape):
                if tf.is_tensor(w):
                  shape = tuple(w.shape.as_list())
                else:
                  shape = w.shape
                shape = tuple(-1 if x is None else x for x in shape)
                w = tf.reshape(w, shape + (1,) * (len(losses.shape) - len(w.shape)))
              return tf.reduce_mean(losses * w) + sum(self.model.losses)
          else:
            output_types = ['prediction', 'embedding']
            loss = L2Loss()
        super(GraphConvModel, self).__init__(model,
                                             loss,
                                             output_types=output_types,
                                             batch_size=batch_size,
                                             **kwargs)