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