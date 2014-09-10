require 'util'	--holds helper functions
local configurations = require 'configurations'	--holds catalog of configurations


	-- solver variables


--solverName = 'EulerBurgers'
--solverName = 'EulerHLL' -- needs 2nd order support.
--solverName = 'EulerRoe'
--solverName = 'SRHDRoe' -- in the works
--solverName = 'MHDBurgers'
solverName = 'MHDRoe' -- in the works
--solverName = 'ADMRoe' -- exploding


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


integratorName = 'ForwardEuler'
--integratorName = 'RungeKutta4'


useGPU = true
maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false 
fixedDT = .01
cfl = .5
displayMethod = 'DENSITY'
displayScale = 2
boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}

-- gravity is specific to the Euler fluid equation solver
useGravity = false 
gravitationalConstant = 25	-- G = 6.67384e-11 m^3 kg^-1 s^-2 TODO meaningful units please
gaussSeidelMaxIter = 20

showVectorField = true
vectorFieldResolution = 64
vectorFieldScale = .125
-- Euler equations' constants:
gamma = 1.4

-- specific to MHD
vaccuumPermeability = 10	--4 * math.pi * 1e-7		-- mu0 = 4π*1e−7 V s A^-1 m^-1

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
size = {64, 64, 64}
vectorFieldResolution = 16
--]]
--[[ 2D
size = {512, 512}
--]]
-- [[ 1D
size = {1024}
displayScale = .25
--]]


-- see initState for a list of options
configurations['Brio-Wu']()

