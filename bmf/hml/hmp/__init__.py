import platform
if platform.system().lower() == 'windows':
    from bmf.bin._hmp import *
    from bmf.bin._hmp import __version__
    from bmf.bin._hmp import __config__
else:
    from bmf.lib._hmp import *
    from bmf.lib._hmp import __version__
    from bmf.lib._hmp import __config__
from . import tracer
