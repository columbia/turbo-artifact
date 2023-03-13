import redis
import yaml


class CacheKey:
    def __init__(self, query_id, blocks):
        self.key = f"{query_id}:{blocks}"


class CacheEntry:
    def __init__(self, result, noise_std, noise):
        self.result = result  # True result without noise
        self.noise_std = noise_std  # std of Laplace distribution
        self.noise = noise  # The actual noise sampled from the distribution
        # TODO: store list of epsilons (or b?). Encapsulates multiblock case neatly?


# TODO: MonteCarlo instead of utility?


class LaplaceCache:
    def __init__(self, config):
        self.config = config
        self.kv_store = self.get_kv_store(config)

    def get_kv_store(self, config):
        return redis.Redis(host=config.cache.host, port=config.cache.port, db=0)

    def write_entry(self, query_id, blocks, cache_entry):
        key = CacheKey(query_id, blocks).key
        self.kv_store.hset(key, "result", cache_entry.result)
        self.kv_store.hset(key, "noise_std", cache_entry.noise_std)
        self.kv_store.hset(key, "noise", cache_entry.noise)

    def read_entry(self, query_id, blocks):
        key = CacheKey(query_id, blocks).key
        entry = self.kv_store.hgetall(key)
        if entry:
            return CacheEntry(
                float(entry[b"result"]),
                float(entry[b"noise_std"]),
                float(entry[b"noise"]),
            )
        return None

    def dump(self):
        pass


class MockLaplaceCache(LaplaceCache):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, query_id, blocks, cache_entry):
        key = CacheKey(query_id, blocks).key
        self.kv_store[key] = {
            "result": cache_entry.result,
            "noise_std": cache_entry.noise_std,
            "noise": cache_entry.noise,
        }

    def read_entry(self, query_id, blocks):
        key = CacheKey(query_id, blocks).key
        if key in self.kv_store:
            entry = self.kv_store[key]
            return CacheEntry(entry["result"], entry["noise_std"], entry["noise"])
        return None

    def dump(self):
        print("Cache", yaml.dump(self.key_values))
