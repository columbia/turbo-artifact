import numpy as np
from precycle.utils.utils import satisfies_constraint
from precycle.utils.utility_theorems import get_pmw_epsilon
import time


class SparseVector:
    def __init__(self, id, alpha, beta, n) -> None:
        self.n = n
        self.id = id
        self.alpha = alpha
        self.beta = beta

        self.epsilon = get_pmw_epsilon(self.alpha, self.beta, self.n, 1)
        self.b = 1 / (self.n * self.epsilon)
        self.noisy_threshold = None
        self.initialized = False
        self.outstanding_payment_blocks = {}

    def initialize(self):
        self.noisy_threshold = self.alpha / 2 + np.random.laplace(loc=0, scale=self.b)
        self.initialized = True

    def check(self, true_output, noisy_output):
        assert self.noisy_threshold is not None
        true_error = abs(true_output - noisy_output)
        # print("true_error", true_error)
        error_noise = np.random.laplace(loc=0, scale=self.b)
        noisy_error = true_error + error_noise
        # print("noisy_error", noisy_error, "noisy_threshold", self.noisy_threshold)
        # time.sleep(2)

        if noisy_error < self.noisy_threshold:
            return True
        else:
            return False


class CacheKey:
    def __init__(self, node_id):
        self.key = node_id


class MockSparseVectors:
    def __init__(self, config):
        self.kv_store = {}
        self.config = config
        self.blocks_metadata = self.config.blocks_metadata
        self.block_size = self.config.blocks_metadata["block_size"]

    def covers_request(self, node, blocks):
        return node[0] <= blocks[0] and node[1] >= blocks[1]

    def get_lowest_common_ancestor(self, blocks):
        # Find the lowest common ancestor of <blocks>
        x = blocks[0]
        node = (x, x)
        while not self.covers_request(node, blocks):
            node_size = node[1] - node[0] + 1
            p1 = (node[0] - node_size, node[1])
            p2 = (node[0], node[1] + node_size)
            node = p2 if satisfies_constraint(p2) else p1
        return node

    def create_new_entry(self, node_id):

        # Find the smallest request that will be handled by this sparse vector.
        node_size = node_id[1] - node_id[0] + 1
        smallest_request_size = (node_size / 2) + 1
        n = smallest_request_size * self.block_size

        # NOTE: n = smallest_SV_request * block_size
        sparse_vector = SparseVector(
            id=node_id,
            alpha=self.config.alpha,
            beta=self.config.beta,
            n=n,
        )
        return sparse_vector

    def write_entry(self, entry):
        self.kv_store[entry.id] = entry

    def read_entry(self, node_id):
        if node_id in self.kv_store:
            return self.kv_store[node_id]
        return None
