"""Allow the same interface through ``python -m de_latency``."""
from .cli import main

raise SystemExit(main())
