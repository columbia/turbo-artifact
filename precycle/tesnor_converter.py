from precycle.budget.histogram import build_sparse_tensor


class TensorConverter:
    def __init__(self, blocks_metadata) -> None:
        self.attribute_domain_sizes = blocks_metadata["attributes_domain_sizes"]

    def query_vector_to_tensor(self, query_vector):
        tensor = build_sparse_tensor(
            bin_indices=query_vector,
            values=[1.0] * len(query_vector),
            attribute_sizes=self.attribute_domain_sizes,
        )
        return tensor
