### OpenCL Riemann solver for computational fluid dynamics.

A finite volume solver based on hyperbolic formalisms.

Works in 1D, 2D, and 3D.

Simulates Euler equations, SRHD, Maxwell, and ADM formalism (Bona-Masso) numerical relativity.  Almost got MHD working.

Euler is simulated via Burgers, HLL, HLLC, Roe.
SRHD via Roe.
Maxwell via Roe.
ADM Bona-Masso via Roe.
MHD, I'm working on a Burgers, HLLC, and Roe ... though none are working yet.

Slope limiters are working on all Burgers and Roe solvers.

Support for Periodic, Mirror, and Free-Flow boundary methods.

Self-gravity option for Euler and MHD equation solvers.

Might get around to CG or GMRes method of Backward Euler implicit integration.

Check out the Makefile for a list of its dependent projects.

### Bona-Masso ADM Numerical Relativity Results:

#### gravitational wave extrinsic curvature:

[![gravitational wave extrinsic curvature](http://img.youtube.com/vi/dDVYA4hPqf0/0.jpg)](http://www.youtube.com/watch?v=dDVYA4hPqf0 "gravitational wave extrinsic curvature")

#### slow and stable warp bubble

[![slow and stable warp bubble](http://img.youtube.com/vi/DZb5hh4M2jg/0.jpg)](http://www.youtube.com/watch?v=DZb5hh4M2jg "slow and stable warp bubble")

#### alcubierre warp bubble collapse

[![alcubierre warp bubble collapse](http://img.youtube.com/vi/ekKf21Cj4k0/0.jpg)](http://www.youtube.com/watch?v=ekKf21Cj4k0 "alcubierre warp bubble collapse")
