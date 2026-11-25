StrikeTwice
===========

StrikeTwice is the PyTorch implementation accompanying
*StrikeTwice: Reusable Graph Representations for Path Prediction* by
Gwenn Guichaoua and Elodie Laine.

An Edge–Vertex Transformer (EVT) updates vertex states and ordered-pair
states together. StrikeTwice applies EVT to a graph, gathers its states in
path order, and applies a second EVT to the resulting path occurrences.
Paths on the same graph can share the first encoding.

Installation
------------

Use Python 3.9 or newer. Run this command from the release repository to
install the package and its PyTorch dependency.

.. code-block:: shell

   python -m pip install .

Import the models from ``path_augmented_transformer``. The StrikeTwice
tutorial includes the Task 2 held-out evaluation and cache timing benchmark
with CUDA, Apple Metal (MPS), and CPU support.

Download the :download:`standalone EVT notebook <tutorials/evt.ipynb>` or the
:download:`StrikeTwice notebook <tutorials/striketwice.ipynb>` to run the examples
interactively.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   tutorials/evt
   tutorials/striketwice

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
