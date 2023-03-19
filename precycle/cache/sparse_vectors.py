import numpy as np
from precycle.utils.utils import satisfies_constraint, get_blocks_size
from precycle.utils.utility_theorems import get_sv_epsilon
import redis
import time


class SparseVector:
    def __init__(self, id, alpha=None, beta=None, n=None, sv_state=None) -> None:
        self.n = n
        self.id = id
        self.alpha = alpha
        self.beta = beta

        if not sv_state:
            self.epsilon = get_sv_epsilon(self.alpha, self.beta, self.n)
            self.b = 1 / (self.n * self.epsilon)
            self.noisy_threshold = None
            self.initialized = False
            self.outstanding_payment_blocks = {}
        else:
            self.epsilon = sv_state["epsilon"]
            self.b = sv_state["b"]
            self.noisy_threshold = sv_state["noisy_threshold"]
            self.initialized = sv_state["initialized"]
            self.outstanding_payment_blocks = sv_state["outstanding_payment_blocks"]

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
        if noisy_error < self.noisy_threshold:
            return True
        else:
            return False


class CacheKey:
    def __init__(self, node_id):
        self.key = str(node_id)


class SparseVectors:
    def __init__(self, config):
        self.config = config
        self.kv_store = self.get_kv_store(config)
        self.blocks_metadata = self.config.blocks_metadata
        # self.block_size = self.config.blocks_metadata["block_size"]

    def get_kv_store(self, config):
        return redis.Redis(host=config.cache.host, port=config.cache.port, db=0)

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

    def min_sum_subarray(self, blocks, k):

        arr = []
        for i in blocks:
            if not str(i) in self.blocks_metadata["blocks"]:
                break
            else:
                arr.append(get_blocks_size((i, i), self.blocks_metadata))
        assert not len(arr) < k
        start = 0
        end = k - 1
        window_sum = sum(arr[start : end + 1])
        min_sum = window_sum

        for i in range(k, len(arr)):
            window_sum += arr[i] - arr[i - k]
            if window_sum < min_sum:
                min_sum = window_sum
                start = i - k + 1
                end = i

        return min_sum

    def create_new_entry(self, node_id):
        # Find the smallest request that will be handled by this sparse vector.
        covered_blocks = range(node_id[0], node_id[1] + 1)
        node_size = len(covered_blocks)
        smallest_request_size = 1 if node_size == 1 else int((node_size / 2) + 1)
        # Find the <smallest-request-size> number of continuous blocks with smallest possible population n
        n = self.min_sum_subarray(covered_blocks, smallest_request_size)
        # print("smallest size request", n)
        # n = smallest_request_size * 65318

        sparse_vector = SparseVector(
            id=node_id,
            beta=self.config.beta,
            alpha=self.config.alpha,
            n=n,
        )
        return sparse_vector

    def write_entry(self, cache_entry):
        key = CacheKey(cache_entry.id).key
        self.kv_store.hset(key + ":sparse_vector", "epsilon", cache_entry.epsilon)
        self.kv_store.hset(key + ":sparse_vector", "b", cache_entry.b)
        self.kv_store.hset(
            key + ":sparse_vector", "noisy_threshold", str(cache_entry.noisy_threshold)
        )
        self.kv_store.hset(
            key + ":sparse_vector", "initialized", int(cache_entry.initialized)
        )
        if cache_entry.outstanding_payment_blocks:
            self.kv_store.hmset(
                key + ":sparse_vector:outstanding_payment_blocks",
                cache_entry.outstanding_payment_blocks,
            )

    def read_entry(self, node_id):
        key = CacheKey(node_id).key
        sv_state = {}
        sv_info = self.kv_store.hgetall(key + ":sparse_vector")
        sv_outstanding_payments_info = self.kv_store.hgetall(
            key + ":sparse_vector:outstanding_payment_blocks"
        )
        if sv_info:
            sv_state["epsilon"] = float(sv_info[b"epsilon"])
            sv_state["b"] = float(sv_info[b"b"])
            sv_state["noisy_threshold"] = float(sv_info[b"noisy_threshold"])
            sv_state["initialized"] = (
                True if str(sv_info[b"initialized"]) == "1" else False
            )
        if sv_outstanding_payments_info:
            sv_state["outstanding_payment_blocks"] = {
                str(key): int(value)
                for key, value in sv_outstanding_payments_info.items()
            }
        # print("sv state", sv_state)
        if sv_state:
            return SparseVector(id=node_id, sv_state=sv_state)
        return None


class MockSparseVectors(SparseVectors):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, cache_entry):
        self.kv_store[cache_entry.id] = cache_entry

    def read_entry(self, node_id):
        if node_id in self.kv_store:
            return self.kv_store[node_id]
        return None
