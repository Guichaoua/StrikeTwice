<img src="docs/images/striketwice-logo.png" alt="StrikeTwice logo" width="300">

# StrikeTwice

Core PyTorch implementation for *StrikeTwice: Reusable Graph Representations for Path Prediction* by Gwenn Guichaoua and Elodie Laine.

The Edge–Vertex Transformer (EVT) encodes vertex and ordered-pair features. StrikeTwice uses an EVT to encode the graph, selects states in path-occurrence order, and applies a second EVT to the selected states. Several paths can share the same graph encoding.

## Installation

Use Python 3.9 or newer. Install StrikeTwice and its PyTorch dependency from PyPI.

```sh
python -m pip install StrikeTwice
```

To install from a source checkout, run this command from the repository root.

```sh
python -m pip install .
```

Import the models from `path_augmented_transformer`.

## Documentation

The [documentation on Read the Docs](https://striketwice.readthedocs.io/en/latest/) includes two runnable notebooks and a Python API reference. Documentation builds use CPU by default.

The [Task 2 notebook](https://striketwice.readthedocs.io/en/latest/tutorials/striketwice.html#Task-2-benchmark) uses CUDA or Apple Metal (MPS) when available. Set `STRIKETWICE_TASK2_DEVICE` to `cuda`, `mps`, or `cpu` before launching Jupyter to choose a backend explicitly.

To build locally, put Pandoc on your PATH and run these commands from the repository root.

```sh
python -m pip install . -r docs/requirements.txt
python -m sphinx -W -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to browse the result.

## Use EVT on its own

### One EVT block

Call `EdgeVertexTransformerLayer` directly on vertex states `X` of shape `[B, n, d_model]` and ordered-pair states `Y` of shape `[B, n, n, d_model]`. The boolean `node_mask` identifies valid vertices. The block runs on the graph alone and returns updated states with the same shapes.

```python
import torch
from path_augmented_transformer import EdgeVertexTransformerLayer

X = torch.randn(1, 3, 8)
Y = torch.randn(1, 3, 3, 8)
node_mask = torch.ones(1, 3, dtype=torch.bool)

block = EdgeVertexTransformerLayer(d_model=8, num_heads=2)
X_next, Y_next = block(X, Y, node_mask)
```

### EVT with input projections

For raw vertex features `X` of shape `[B, n, d_x]` and ordered-pair features `Y` of shape `[B, n, n, d_y]`, use `EdgeVertexTransformer` to project both streams to width `d_h`. Set `num_layers=1` for a single EVT block. The encoder returns the contextualized states in `out.X` and `out.Y`.

```python
import torch
from path_augmented_transformer import EdgeVertexTransformer

X = torch.tensor([[[1.0], [2.0], [3.0]]])
Y = torch.zeros(1, 3, 3, 1)
Y[0, 0, 1, 0] = 1.0  # Directed edge 0 -> 1
Y[0, 1, 2, 0] = 1.0  # Directed edge 1 -> 2
node_mask = torch.ones(1, 3, dtype=torch.bool)

evt = EdgeVertexTransformer(d_x=1, d_y=1, d_h=8, num_layers=1, num_heads=2)
out = evt(X, Y, node_mask)
print(out.X.shape, out.Y.shape)  # [1, 3, 8] and [1, 3, 3, 8]
```

## Run the StrikeTwice CPU demo

```sh
python demo.py
```

The demo builds a four-vertex graph in memory and runs two paths, one of which revisits a vertex. It checks that the vertex states, pair states and scalar predictions agree when using a shared cache or recomputing the graph encoding for each path. It uses randomly initialized weights.

## Tiled inference at 1,024 vertices

Use `--paper` to run the full Atlas architecture from the paper's scaling study. It has 778,897 parameters, width 64, two EVT layers per stage, four graph heads and eight path heads. The synthetic workload uses the study's seeds, seven vertex features, six pair features and one batch of 16 paths of length 16. The graph encoder uses the aggressive tiling settings from the paper.

```sh
# Smoke check of the full model on CPU
python demo_scaling.py --paper --nodes 64 --device cpu

# Full-size synthetic inference workload on a CUDA GPU
python demo_scaling.py --paper --nodes 1024 --device cuda --warmups 2 --repeats 10
```

The release measures inference with initialized weights. The forward/backward timing and memory in the paper used activation checkpointing on an A100. The separate trained-checkpoint amortization measurements used their own query workload.

### Examples from observed traceroutes

```sh
python demo_scaling.py  # quick synthetic example
python demo_scaling.py --device cpu examples/atlas-de-1024.json examples/atlas-us-1024.json
```

The second command builds one cache per graph and reuses it for four routes. It reports cache construction and path-processing times separately, the size of the stored cache, and the size of one triangular score tensor before and after tiling. With one FP32 head, the score tensor drops from 4 GiB to 32 MiB per tile at the default eight source rows. This is a tensor-size calculation. Use `--device cuda` to also measure the PyTorch allocator's peak on a CUDA GPU.

The demo checks that tiled and untiled inference agree on a small graph using the same weights. The opt-in PyTorch implementation tiles triangular attention over source rows and value channels, and cross-stream attention over query rows. Each softmax keeps all valid keys or intermediate vertices. Graph-encoder arithmetic remains cubic in the number of vertices, and the cache remains quadratic.

Both JSON fixtures contain 1,024 vertices from public IPv4 traceroutes collected by RIPE Atlas Region Meshes in August 2026. They pool several destinations within each region and retain directed edges between consecutive responding hops. Vertex IDs are local integers. The fixtures contain topology and route segments, and the default demo uses a width-four model for a quick CPU run. Add `--paper` to use the full model on these topologies with zero-filled input channels apart from adjacency. These regional examples are separate from the per-destination latency benchmark in the paper. [RIPE Atlas Region Meshes](https://atlas.ripe.net/docs/tools-and-code/region-meshes/) · [RIPE Atlas terms](https://www.ripe.net/about-us/legal/ripe-atlas-service-terms-and-conditions/)

The fixtures record source URLs and SHA-256 digests. To rebuild them, download each archive into `DATE/REGION/` under a local data directory and run `python examples/build_atlas_fixtures.py --raw-root /path/to/data`.
