import numpy as np
import irk_coefficients as irk

# Initializing IRK coefficient matrix...
IRK = irk.IRK(order=32)


IRK.build_matrix()
