try:
    import cyipopt
except ImportError:
    raise ImportError(
        "\n"
        "The 'dmf' package requires the 'cyipopt' library.\n"
        "Please install cyipopt from conda-forge BEFORE installing dmf:\n\n"
        "    conda install -c conda-forge cyipopt \n\n"
    )
from .dmf import *
