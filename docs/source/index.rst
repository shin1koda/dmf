.. dmf documentation master file, created by
   sphinx-quickstart on Wed Feb 28 14:57:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documantation of shin1koda/dmf
==============================

dmf.py is a Python module of the direct MaxFlux method. When you use this module in your research, please cite [1] Shin-ichi Koda and Shinji Saito, JCTC (2024). doi.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. autoclass:: dmf.DirectMaxFlux

   .. autoclass:: dmf.DirectMaxFlux.History
   .. automethod:: get_positions
   .. automethod:: set_positions
   .. automethod:: get_forces
   .. automethod:: interpolate_energies
   .. automethod:: get_x
   .. automethod:: set_x
   .. automethod:: solve
   .. automethod:: objective
   .. automethod:: gradient
   .. automethod:: constraints
   .. automethod:: jacobian
   .. automethod:: intermediate

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
