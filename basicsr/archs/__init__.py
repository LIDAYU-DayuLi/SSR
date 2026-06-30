# Minimal lazy-compat shim to avoid circular imports.
# Do NOT import basicsr.models at module import time.

def build_network(opt):
    # Import at call-time to break the cycle
    from basicsr.models.archs import define_network as _define_network
    return _define_network(opt)

def define_network(opt):
    # Some repos call define_network directly; forward to the same target
    from basicsr.models.archs import define_network as _define_network
    return _define_network(opt)
