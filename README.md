# Pydro - Simple hydro experiments

Pydro is a collection of reconstruction schemes and other components needed to
solve systems of equations that can develop discontinuities, such as Burgers
equation, and compressible Newtonian Euler. If Numba is available the code will
be JITed for better performance.

The implemented numerical schemes are:
- Time steppers:
  * Adams-Bashforth 1, 2, 3, or 4
- Reconstruction:
  * Minmod
  * WCNS3 (weighted compact nonlinear scheme)
    http://dx.doi.org/10.1016/j.compfluid.2012.09.001
  * WENO3 (weighted essentially non-oscillatory)
  * WCNS5 http://dx.doi.org/10.1016/j.compfluid.2012.09.001, and
    http://dx.doi.org/10.1016/j.compfluid.2015.08.023
  * WCNS5Z the WCNS5 scheme with the WENO5Z oscillation indicator
- Derivatives:
  * Midpoint-to-node (MD) 2nd order
  * MD4 (4th order)
  * Midpoint-to-node-to-node (MND) 4th order
    http://dx.doi.org/10.1016/j.compfluid.2012.09.001
  * MND 6th order http://dx.doi.org/10.1016/j.compfluid.2012.09.001
- For Euler system the primitive or conserved variables can be reconstructed.

Boundary conditions are assumed to be constant in time, which works fine for
the simple shock tests that are currently solved here.
