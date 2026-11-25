Python API
==========

The examples use the public classes imported from ``path_augmented_transformer``.
In the shapes below, ``B`` is the batch size, ``n`` is the padded number of
graph vertices, and ``k`` is the padded path length.

Edge–Vertex Transformer
-----------------------

.. autoclass:: path_augmented_transformer.EdgeVertexTransformerLayer
   :members: forward

.. autoclass:: path_augmented_transformer.EdgeVertexTransformer
   :members: forward, enable_tiled_attention

.. autoclass:: path_augmented_transformer.EdgeVertexOutput

   ``X`` holds the vertex states with shape ``[B, n, d_h]``.
   ``Y`` holds the ordered-pair states with shape ``[B, n, n, d_h]``.

StrikeTwice
-----------

.. autoclass:: path_augmented_transformer.StrikeTwice
   :members: forward, select_occurrence_graph

.. autoclass:: path_augmented_transformer.GraphPathBatch
   :members: to

   * ``X`` and ``Y`` are floating-point graph features with shapes
     ``[B, n, d_x]`` and ``[B, n, n, d_y]``.
   * ``p`` has shape ``[B, k]`` and uses ``torch.long`` vertex indices.
     A vertex may appear multiple times in a path.
   * ``node_mask`` and ``path_mask`` are boolean masks with shapes
     ``[B, n]`` and ``[B, k]``. Valid path positions come before padding.
   * Optional ``graph_index`` is a ``torch.long`` tensor with shape ``[B]``.
     Give rows the same ID when their graph features and vertex masks
     are identical. StrikeTwice encodes each graph ID once in that batch.

.. autoclass:: path_augmented_transformer.PathEncoderOutput

   In StrikeTwice, ``X`` and ``Y`` contain the first EVT's graph states,
   with shapes ``[B, n, d_h]`` and ``[B, n, n, d_h]``. ``Z`` and ``R``
   contain the second EVT's path-occurrence states, with shapes
   ``[B, k, d_h]`` and ``[B, k, k, d_h]``.

Scalar prediction
-----------------

.. autoclass:: path_augmented_transformer.ScalarPathHead
   :members: forward

.. autoclass:: path_augmented_transformer.data_types.ScalarTaskOutput

   ``u`` is the masked mean of the path vertex states and has shape
   ``[B, d_h]``. ``prediction`` has shape ``[B, output_dim]``.
