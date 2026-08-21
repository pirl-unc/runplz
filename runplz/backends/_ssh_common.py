"""Compatibility import for :mod:`runplz.backends.ssh_common`.

Shared SSH staging and lifecycle behavior is public as of runplz 3.15.3.
New code should import ``runplz.backends.ssh_common`` directly.
"""

from runplz.backends import ssh_common as _impl


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
