from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import math


class DenseHistogram:  # We use it to represent the PMW Histogram
    def __init__(
        self,
        domain_size: Optional[int] = None,
        attribute_sizes: Optional[List[int]] = None,
    ) -> None:
        # TODO: optimize this later, maybe we only need to store the "diff", which is sparse
        self.N = domain_size if domain_size else get_domain_size(attribute_sizes)
        self.tensor = (
            torch.ones(  # TODO: consider naming this bins to hide internal implem
                size=(1, self.N),
                dtype=torch.float64,
            )
        )
        self.normalize()

    def normalize(self) -> None:
        F.normalize(self.tensor, p=1, out=self.tensor)

    def multiply(self, tensor) -> None:
        # elementwise multiplication
        torch.mul(self.tensor, tensor, out=self.tensor)

    def run(self, query: torch.Tensor) -> float:
        # sparse (1,N) x dense (N,1)
        return torch.smm(query, self.tensor.t()).item()


class SparseHistogram:  # We use it to represent the block data
    def __init__(
        self, bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
    ) -> None:
        # Flat representation of shape (1, N)
        self.tensor = build_sparse_tensor(
            bin_indices=bin_indices,
            values=np.array(values) / len(values),
            attribute_sizes=attribute_sizes,
        )
        self.domain_size = self.tensor.shape[1]  # N

    @classmethod
    def from_dataframe(
        cls,
        df,
        attribute_domain_sizes,
    ) -> "SparseHistogram":

        cols = list(df.columns)
        df = df.groupby(cols).size()
        return cls(
            bin_indices=list(df.index),              # [(0, 0, 1), (1, 0, 5), (0, 1, 2)],
            values=list(df.values),                  # [4, 1, 2],
            attribute_sizes=attribute_domain_sizes,  # [2, 2, 10],
        )

    def dump(self):
        return {
            "id": self.id,
            "initial_budget": self.initial_budget.dump(),
            "budget": self.budget.dump(),
        }

    def run(self, query: torch.Tensor) -> float:
        # `query` has shape (1, N), we need the dot product, or matrix mult with (1,N)x(N,1)
        # return torch.mm(self.tensor, query.t()).item()
        return torch.sparse.mm(self.tensor, query.t()).item()


# ------------- Helper functions ------------- #
def get_flat_bin_index(
    multidim_bin_index: List[int], attribute_sizes: List[int]
) -> int:
    index = 0
    size = 1
    # Row-major order like PyTorch (inner rows first)
    for dim in range(len(attribute_sizes) - 1, -1, -1):
        index += multidim_bin_index[dim] * size
        size *= attribute_sizes[dim]
    return index


# TODO: write the inverse conversion


def get_domain_size(attribute_sizes: List[int]) -> int:
    return math.prod(attribute_sizes)


def build_sparse_tensor(
    bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
):
    # One row only
    column_ids = []
    column_values = []

    for b, v in zip(bin_indices, values):
        column_ids.append(get_flat_bin_index(b, attribute_sizes))
        column_values.append(v)     # In case we lose the correct order?

    return torch.sparse_coo_tensor(
        [[0] * len(column_ids), column_ids],
        column_values,
        size=(1, get_domain_size(attribute_sizes)),
        dtype=torch.float64,
    )


def build_sparse_tensor_multidim(
    bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
):
    return torch.sparse_coo_tensor(
        list(zip(*bin_indices)),
        values,
        size=attribute_sizes,
        dtype=torch.float64,
    )


# ------------- / Help functions ------------- #


def debug2():

    attribute_sizes = [2, 2, 10]

    print(
        get_flat_bin_index(
            multidim_bin_index=(0, 0, 0), attribute_sizes=attribute_sizes
        )
    )

    print(
        get_flat_bin_index(
            multidim_bin_index=(0, 1, 2), attribute_sizes=attribute_sizes
        )
    )

    h = DenseHistogram(attribute_sizes=attribute_sizes)
    print(h.tensor)
    # q = build_sparse_tensor({(0, 0, 0): 1.0, (0, 1, 5): 1.0})

    block = build_sparse_tensor(
        bin_indices=[[0, 0, 1], [0, 1, 5], [0, 0, 0]],
        values=[1.0, 4.0, 3.0],
        attribute_sizes=attribute_sizes,
    )

    q = build_sparse_tensor(
        bin_indices=[[0, 0, 0], [0, 1, 5]],
        values=[1.0, 1.0],
        attribute_sizes=attribute_sizes,
    )
    print(q)

    print(torch.sparse.mm(block, q.t()).item())

    print(
        build_sparse_tensor(
            bin_indices=[],
            values=[],
            attribute_sizes=attribute_sizes,
        )
    )

    print(h.run_query(q))

    v = torch.sparse_coo_tensor(
        indices=[[0, 0, 0], [0, 1, 2]],
        values=[1.0, 1.0, 1.0],
        size=(1, 40),
        dtype=torch.float64,
    )
    q = torch.sparse_coo_tensor(
        indices=[[0, 0], [1, 2]],
        values=[3.0, 4.0],
        size=(1, 40),
        dtype=torch.float64,
    )
    print(torch.sparse.mm(v, q.t()).item())

    # print(h)
    # print(q)
    # print(h.run_query(q))


if __name__ == "__main__":
    debug2()
