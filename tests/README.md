# Tests and benchmarks

To profile:
```bash
LOGURU_LEVEL=WARNING python -m cProfile -s cumtime tests/profile.py
```

You can also pipe to a file (`> out.txt`) or save the result and explore it interactively:
```bash
LOGURU_LEVEL=WARNING python -m cProfile -o prof.stats tests/profile.py
python -m pstats prof.stats
prof.stats% sort cumtime
prof.stats% reverse
prof.stats% stats
```