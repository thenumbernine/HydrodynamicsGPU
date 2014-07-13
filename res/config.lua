require 'util'	--holds helper functions
local configurations = require 'configurations'	--holds catalog of configurations


	-- solver variables


--solverName = 'EulerBurgers' -- works for 1D, 2D, 3D
--solverName = 'EulerHLL' -- works for 1D, 2D, 3D.  needs 2nd order support.
--solverName = 'EulerRoe' -- works for 1D, 2D, compiler crashes for 3D
--solverName = 'SRHDRoe' -- in the works
solverName = 'MHDRoe' -- left eigenvectors not finished, and numeric inverse not working
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


useGPU = true
--maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false 
fixedDT = .01
cfl = .5
displayMethod = 'DENSITY'
displayScale = 2
boundaryMethods = {'MIRROR', 'MIRROR', 'MIRROR'}

-- gravity is specific to the Euler fluid equation solver
useGravity = false 
gravitationalConstant = 25	-- G = 6.67384e-11 m^3 kg^-1 s^-2 TODO meaningful units please
gaussSeidelMaxIter = 20

showVelocityField = true	-- for one test this was 72fps with, 76fps without, so 5% perf hit
velocityFieldResolution = 64
velocityFieldScale = .125
-- Euler equations' constants:
gamma = 1.4

-- specific to MHD
vaccuumPermeability = 1	--4 * math.pi * 1e-7		-- mu0 = 4π*1e−7 V s A^-1 m^-1

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
size = {64, 64, 64}
--]]
--[[ 2D
size = {256, 256}
--]]
-- [[ 1D
size = {1024}
displayScale = .25
--]]


-- see initState for a list of options
configurations['Brio-Wu']()

