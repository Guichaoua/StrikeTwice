"""Run StrikeTwice inference on two paths through a toy directed graph."""

import torch

from path_augmented_transformer import (
    GraphPathBatch,
    PathEncoderOutput,
    ScalarPathHead,
    StrikeTwice,
)


@torch.inference_mode()
def main():
    torch.manual_seed(7)
    # One feature per vertex and one adjacency channel per ordered pair.
    vertices = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    pairs = torch.zeros((1, 4, 4, 1))
    for source, target in [(0, 1), (1, 0), (0, 2), (2, 3)]:
        pairs[0, source, target, 0] = 1.0
    node_mask = torch.ones((1, 4), dtype=torch.bool)

    model = StrikeTwice(
        d_x=1,
        d_y=1,
        d_h=8,
        max_path_length=4,
        graph_num_layers=1,
        path_num_layers=1,
        num_heads=2,
    ).eval()
    head = ScalarPathHead(d_h=8).eval()
    graph_cache = model.graph_backbone(vertices, pairs, node_mask)

    # The first path revisits vertex 0. The second has a padded position.
    for path in [[0, 1, 0, 2], [0, 2, 3, -1]]:
        indices = torch.tensor([path], dtype=torch.long)
        batch = GraphPathBatch(
            X=vertices,
            Y=pairs,
            p=indices,
            node_mask=node_mask,
            path_mask=indices >= 0,
        )
        selected = model.select_occurrence_graph(graph_cache, batch)
        occurrence = model.path_backbone(selected.X, selected.Y, batch.path_mask)
        cached = PathEncoderOutput(
            X=graph_cache.X, Y=graph_cache.Y, Z=occurrence.X, R=occurrence.Y
        )
        fresh = model(batch)
        torch.testing.assert_close(cached.Z, fresh.Z)
        torch.testing.assert_close(cached.R, fresh.R)

        prediction = head(cached, batch.path_mask).prediction
        torch.testing.assert_close(prediction, head(fresh, batch.path_mask).prediction)
        valid_path = indices[batch.path_mask].tolist()
        print(f"Path {valid_path}  prediction {prediction.item():.6f}")

    print("Cached and recomputed predictions agree for both paths.")


if __name__ == "__main__":
    main()
