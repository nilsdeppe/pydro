Time Steppers
=============

Pydro offers a few different time steppers, including many
strong-stability-preserving (SSP) time steppers. The linear
Runge-Kutta (RK) time steppers, while simple, should be avoided
since they can only achieve the promised order of accuracy for
linear systems. The SSP RK3 and SSP RK4 are both robust choices
for nonlinear hydrodynamics simulations. The SSP RK4 has a maximum
step size roughly 50% larger than an Euler step and the SSP RK3,
making it a more efficient method than the SSP RK3 There are also
(mostly complete) Adams-Bashforth time steppers up to fourth order,
which still require a self-starting procedure in order to truly
reach the high-order accuracy.

.. automodule:: TimeStepper
   :members:
