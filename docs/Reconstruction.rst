Reconstruction Schemes
======================

Pydro provides two variants of reconstruction schemes. The first
are the standard reconstruction schemes including TVD and WENO-type
schemes, while the second are positivity-preserving adaptive-order schemes. The
two scheme classes have different interfaces and so must be used slightly
differently. The interface for each is documented in the sections below.

Standard reconstruction
-----------------------

The standard reconstruction schemes such as total variation
diminishing (TVD) schemes and different flavors of higher-order
essentially non-oscillatory or weighted compact schemes are
available through a common interface. The
:py:meth:`Reconstruction.reconstruct` allows reconstructing a
series of variables using one of the available schemes
(see :py:class:`Reconstruction.Scheme` for a list).


.. automodule:: Reconstruction
   :members:

Positivity-preserving reconstruction
------------------------------------

The PPAO schemes need to be wrapped in a function specific for each
evolution system so that the appropriate variables can have their
positivity preserved.

.. automodule:: ReconstructionPpao
   :members:
