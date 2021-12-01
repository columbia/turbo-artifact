import gzip
import json

from amazon.ion import simpleion as ion

d = {"a": 1, "b": 2, "c": 3}

# Let's see if there is a better format to store our logs
# - json: 214M
# - binary ion: 38M
# - text ion: 98M
# - gzip json: 2.0M
# That was fun, Amazon Ion might be useful to communicate with Go stuff but not for compression
# Well, maybe not so worth it then.

with open(
    "/home/pierre/privacypacking/logs/time_based_budget_unlocking_DominantShares/1115-103325_ed70ba.json",
    "r",
) as f:
    d = json.load(f)

print(len(d.items()))

with gzip.open("test.json.gz", "wt", encoding="UTF-8") as f:
    json.dump(d, f)


# ion gives nested "IonPyDict" objects and not dictionaries directly

# with open("ion_test.ion", "wb") as f:
#     ion.dump(d, f)


# with open("ion_test.ion", "rb") as f:
#     d2 = dict(ion.load(f))
#     print(d2)

# with open("ion_test_str.ion", "wb") as f:
#     ion.dump(d, f, binary=False)


# with open("ion_test_str.ion", "rb") as f:
#     d2 = dict(ion.load(f))
#     print(d2)
