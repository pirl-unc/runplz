"""Legacy entry point for :mod:`runplz.bootstrap`.

Backends emit ``python -m runplz._bootstrap``, and containers running an
older runplz only have this path. It stays for as long as that is true —
see the wire-contract note in :mod:`runplz.bootstrap`.
"""

from runplz.bootstrap import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
