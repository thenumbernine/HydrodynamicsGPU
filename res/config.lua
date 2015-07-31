package.path = '?.lua;?/init.lua'

require 'util'	--holds helper functions
local configurations = require 'configurations'	--holds catalog of configurations


	-- solver variables


--solverName = 'EulerBurgers'
--solverName = 'EulerHLL'		-- needs slope limiter support
--solverName = 'EulerHLLC'		-- needs slope limiter support
--solverName = 'EulerRoe'		-- fails on Colella-Woodward 2-wave problem, but works on all the configurations
--solverName = 'SRHDRoe'		-- not yet
--solverName = 'MHDBurgers'		-- a mathematically-flawed version works with Orszag-Tang and Brio-Wu, and some hydro problems too.  fixing the math error causes it to break.
--solverName = 'MHDHLLC'		-- needs 2nd order support, suffers same as EulerHLLC
--solverName = 'MHDRoe'			-- suffers from negative pressure with magnetic problems.  solves fluid-only problems fine.
--solverName = 'MaxwellRoe'		-- Roe solver based on Trangenstein's Maxwell equations hyperbolic formalism
--solverName = 'ADM1DRoe'			-- Bona-Masso based on "The Appearance of Coordinate Shocks in Hyperbolic Formalisms of General Relativity" by Alcubierre, 1997 
--solverName = 'ADM2DSpherical'	-- not yet
--solverName = 'ADM3DRoe'		-- same as ADM1DRoe but for 3D 
--solverName = 'BSSNOKRoe'		-- not yet
-- TODO ImplicitIncompressibleNavierStokes	<- from my GPU fluid sim Lua+GLSL project
-- TODO ImplicitBSSNOK

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
--integratorName = 'BackwardEulerConjugateGradient'	-- not fully working, experimental only on EulerBurgers


useGPU = true			-- = false means use OpenCL for CPU, which is shoddy for my intel card
maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false
fixedDT = .125
cfl = .5

heatMapVariable = 'DENSITY'
heatMapColorScale = 2

boundaryMethods = {
	{min='FREEFLOW', max='FREEFLOW'},
	{min='FREEFLOW', max='FREEFLOW'},
	{min='FREEFLOW', max='FREEFLOW'},
}

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
size = {32, 32, 32}
vectorFieldResolution = 16
--]]
-- [[ 2D
size = {256, 256}
--]]
--[[ 1D
size = {1024}
heatMapColorScale = .25
--]]

camera = {}


--[[ Euler

-- override solids:

--[=[ cylinder
function calcSolid(x,y,z)
	local cx = .35 * xmin[1] + .65 * xmax[1]
	local cy = .35 * xmin[2] + .65 * xmax[2]
	local cz = .35 * xmin[3] + .65 * xmax[3]
	local dx = #size >= 1 and x - cx or 0
	local dy = #size >= 2 and y - cy or 0
	local dz = #size >= 3 and z - cz or 0
	local rSq = dx * dx + dy * dy + dz * dz
	return rSq < .1 * .1 and 1 or 0
end
--]=]

--[=[ arbitrary
-- hmm ... loading images from Lua ...
-- 1) provide a filename, but that means interjecting it into the resetState() converter code, which is a long way to carry it ... maybe not ...
-- 2) Lua image loading libraries.  the current one depends on FFI.  the LuaCxx binding based ones are having link location problems ...
-- 3) Lua image loading libraries in pure LuaJIT.  This forces us to only build this against LuaJIT though ...
function calcSolid(x,y,z)
	if x > -.275 and x < -.225 and y > -.4 and y < .4 then
		return 1
	end
end
--]=]
--solidFilename = 'test-solid.png'

configurations['Sod']()
--configurations['Square Cavity']()
--configurations['Kelvin-Hemholtz']()
--configurations['Shock Bubble Interaction']()
--configurations['Flow Around Cylinder']()
--configurations['Forward Facing Step']()
--configurations['Double Mach Reflection']()
--configurations['Spiral Implosion']()
--configurations['self-gravitation test 1']()
--]]

--[[ MHD
solverName = 'MHDRoe'
--configurations['Sod']()
configurations['Brio-Wu']()
--]]

--[[ Maxwell 
solverName = 'MaxwellRoe'
heatMapVariable = 'ELECTRIC'
boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
configurations['Maxwell-1']()
--]]

--[[ ADM (1D)
solverName = 'ADM1DRoe'
size = {1024}
heatMapColorScale = 128
configurations['ADM-1D']()
boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
heatMapVariable = 'ALPHA'
camera.zoom = 1/300
camera.pos = {150,150}
--]]

-- [[ ADM (3D)
solverName = 'ADM3DRoe'
--size = {1024} heatMapColorScale = 128
size = {256, 256} heatMapColorScale = 1
configurations['ADM-3D']()
boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
heatMapVariable = 'ALPHA'
--fixedDT = .125
--useFixedDT = true
camera.zoom = 1/300
camera.pos = {150,150}
camera.dist = 450
useGraph = false	--true
graphScale = 300
--]]


-- camera setup:

if #size == 1 then			-- 1D better be ortho
	camera.mode = 'ortho'
elseif #size == 2 then		-- 2D can handle either ortho or frustum
	camera.mode = 'ortho'
	--camera.mode = 'frustum'
else						-- 3D better be frustum
	camera.mode = 'frustum'
end

