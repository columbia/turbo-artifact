import json
from precycle.budget.histogram import build_sparse_tensor


class TensorConverter:
    def __init__(self, block_metadata_path) -> None:
        with open(block_metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.attribute_domain_sizes = self.metadata["attributes_domain_sizes"]

    def query_vector_to_tensor(self, query_vector):
        tensor = build_sparse_tensor(
            bin_indices=query_vector,
            values=[1.0] * len(query_vector),
            attribute_sizes=self.attribute_domain_sizes,
        )
        return tensor


def main():

    query_vector = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 3],
        [0, 0, 0, 4],
        [0, 0, 0, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 7],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 4],
        [0, 0, 1, 5],
        [0, 0, 1, 6],
        [0, 0, 1, 7],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 3],
        [0, 0, 2, 4],
        [0, 0, 2, 5],
        [0, 0, 2, 6],
        [0, 0, 2, 7],
        [0, 0, 3, 0],
        [0, 0, 3, 1],
        [0, 0, 3, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 4],
        [0, 0, 3, 5],
        [0, 0, 3, 6],
        [0, 0, 3, 7],
    ]

    tensor_converter = TensorConverter()
    tensor = tensor_converter.query_vector_to_tensor(query_vector)
    print(tensor)


if __name__ == "__main__":
    main()
