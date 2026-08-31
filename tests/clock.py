"""A stand-in for the `time` module, patched per-module.

Patching `time.sleep` globally turns every wall-clock loop reached during a
test into a busy-loop, and patching `sc.time.sleep` does exactly that
because `ssh_common.time is time`. Replacing a module's own `time`
reference keeps the patch where it was advertised and lets a test advance
the clock deliberately.
"""


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def strftime(self, *a, **kw):
        import time as _time

        return _time.strftime(*a, **kw)

    def gmtime(self, *a, **kw):
        import time as _time

        return _time.gmtime(*a, **kw)
