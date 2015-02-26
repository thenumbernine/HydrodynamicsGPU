require 'util'	--holds helper functions
local configurations = require 'configurations'	--holds catalog of configurations


	-- solver variables


--solverName = 'EulerBurgers'
--solverName = 'EulerHLL'		-- needs slope limiter support
--solverName = 'EulerHLLC'		-- needs slope limiter support
solverName = 'EulerRoe'		-- fails on Colella-Woodward 2-wave problem, but works on all the configurations
--solverName = 'SRHDRoe'		-- in the works
--solverName = 'MHDBurgers'		-- works with Orszag-Tang and Brio-Wu, and some hydro problems too
--solverName = 'MHDHLLC'		-- needs 2nd order support, suffers same as EulerHLLC
--solverName = 'MHDRoe'			-- suffers from negative pressure with magnetic problems.  solves fluid-only problems fine.
--solverName = 'MaxwellRoe'		-- Roe solver based on Trangenstein's Maxwell equations hyperbolic formalism
--solverName = 'ADMRoe'			-- Bona-Masso basd on Alcubierre's paper rather than my attempt to follow his book which failed.
--solverName = 'ADM2DSpherical'	-- not yet
--solverName = 'ADM3DRoe'		-- not yet
--solverName = 'BSSNOKRoe'		-- BSSNOK Roe, not yet

--slopeLimiterName = 'DonorCell'
--slopeLimiterName = 'LaxWendroff'
--slopeLimiterName = 'BeamWarming'	-- not behaving correctly
--slopeLimiterName = 'Fromm'		-- not behaving correctly
--slopeLimiterName = 'CHARM'
--slopeLimiterName = 'HCUS'
--slopeLimiterName = 'HQUICK'
--slopeLimiterName = 'Koren'
--slopeLimiterName = 'MinMod'
--slopeLimiterName = 'Oshker'
--slopeLimiterName = 'Ospre'
--slopeLimiterName = 'Smart'
--slopeLimiterName = 'Sweby'
--slopeLimiterName = 'UMIST'
--slopeLimiterName = 'VanAlbada1'
--slopeLimiterName = 'VanAlbada2'
--slopeLimiterName = 'VanLeer'		-- not behaving correctly
--slopeLimiterName = 'MonotizedCentral'
slopeLimiterName = 'Superbee'
--slopeLimiterName = 'BarthJespersen'


--integratorName = 'ForwardEuler'
integratorName = 'RungeKutta4'


useGPU = true			-- = false means use OpenCL for CPU, which is shoddy for my intel card
maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false
fixedDT = .125
cfl = .5
displayMethod = 'DENSITY'
displayScale = 2
boundaryMethods = {'MIRROR', 'MIRROR', 'MIRROR'}

-- gravity is specific to the Euler fluid equation solver
useGravity = false
gravitationalConstant = 1	-- G = 6.67384e-11 m^3 kg^-1 s^-2 TODO meaningful units please
gaussSeidelMaxIter = 20

showVectorField = false
vectorFieldResolution = 64
vectorFieldScale = .125

-- Euler equations' constants:
gamma = 1.4

-- MHD constants:
vaccuumPermeability = 1	--4 * math.pi * 1e-7		-- mu0 = 4π*1e−7 V s A^-1 m^-1

-- Maxwell constants:
permittivity = 1
permeability = 1
conductivity = 1

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
size = {64, 64, 64}
vectorFieldResolution = 16
--]]
--[[ 2D
size = {128, 128}
--]]
-- [[ 1D
size = {1024}
displayScale = .25
--]]


-- Euler
configurations['Sod']()
--configurations['self-gravitation test 1']()

-- MHD
--configurations['Orszag-Tang']()

--[[ Maxwell 
displayMethod = 'ELECTRIC'
boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
configurations['Maxwell-1']()
--]]

--[[ ADM (1D)
configurations['ADM-1D']()
boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
displayMethod = 'ALPHA'
size = {1024}
displayScale = 128
--]]

