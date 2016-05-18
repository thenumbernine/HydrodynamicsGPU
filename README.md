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

### Bona-Masso ADM Numerical Relativity Results:

#### gravitational wave extrinsic curvature:

[![gravitational wave extrinsic curvature](http://img.youtube.com/vi/dDVYA4hPqf0/0.jpg)](http://www.youtube.com/watch?v=dDVYA4hPqf0 "gravitational wave extrinsic curvature")

#### slow and stable warp bubble

[![slow and stable warp bubble](http://img.youtube.com/vi/DZb5hh4M2jg/0.jpg)](http://www.youtube.com/watch?v=DZb5hh4M2jg "slow and stable warp bubble")

#### alcubierre warp bubble collapse

[![alcubierre warp bubble collapse](http://img.youtube.com/vi/ekKf21Cj4k0/0.jpg)](http://www.youtube.com/watch?v=ekKf21Cj4k0 "alcubierre warp bubble collapse")

Dependcies:
C++
- Common: https://github.com/thenumbernine/Common
- CLCommon: https://github.com/thenumbernine/CLCommon (depends on OpenCL)
- ImGuiCommon: https://github.com/thenumbernine/ImGuiCommon (depends on ImGui listed below)
- GLApp: https://github.com/thenumbernine/GLApp (depends on OpenGL, SDL2 listed below)
- Tensor: https://github.com/thenumbernine/Tensor
- Profiler: https://github.com/thenumbernine/Profiler
- Image: https://github.com/thenumbernine/Image (depends on LibPNG, listed below)
- Shader: https://github.com/thenumbernine/Shader
- LuaCxx: https://github.com/thenumbernine/LuaCxx (depends on Lua/LuaJIT, listed below)
Lua
- ext: https://github.com/thenumbernine/lua-ext
- symmath: https://github.com/thenumbernine/symmath-lua
external:
- ImGui v1.48: https://github.com/ocornut/imgui
- Lua (used by LuaCxx, check that for instructions on building against either), use either one:
	- 5.x: https://www.lua.org/
	- LuaJIT: http://luajit.org/
- SDL2 v2.0.3 (used by GLApp): https://www.libsdl.org/
- LibPNG v1.7.0-beta6 (used by Image): http://www.libpng.org/pub/png/libpng.html
- OpenCL v1.2.  If your dist doesn't include the cl.hpp header (*cough*Apple*cough*) then just put it in the CLCommon/include/OpenCL/ folder
- OpenGL
