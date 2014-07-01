require 'util'	--holds helper functions
local configurations = require 'configurations'	--holds catalog of configurations

	-- solver variables

--[[
options:
EulerBurgers	- works for 1D, 2D, 3D
EulerRoe - works for 1D, 2D, compiler crashes for 3D
EulerHLL - works for 1D, 2D, 3D.  self-gravity pulling towards the positive...
MHDRoe - left eigenvectors not finished
ADMRoe - compiler crashes
--]]
solverName = 'ADMRoe'

--[[
options:
DonorCell
LaxWendroff
BeamWarming	-- not behaving correctly
Fromm		-- not behaving correctly
CHARM
HCUS
HQUICK
Koren
MinMod
Oshker
Ospre
Smart
Sweby
UMIST
VanAlbada1
VanAlbada2
VanLeer		-- not behaving correctly
MonotizedCentral
Superbee
BarthJespersen
--]]
slopeLimiterName = 'Superbee'

useGPU = true
--maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false 
fixedDT = .01
cfl = .5
displayMethod = 'density'
displayScale = 2
boundaryMethods = {'mirror', 'mirror', 'mirror'}

-- gravity is specific to the Euler fluid equation solver
useGravity = false 

magneticFieldNoise = 0
gamma = 1.4

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
size = {64, 64, 64}
--]]
--[[ 2D
size = {512, 512}
--]]
-- [[ 1D
size = {1024}
displayScale = .25
--]]


-- see initState for a list of options
configurations['ADM-1D']()

