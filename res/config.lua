package.path = '?.lua;?/?.lua'

require 'util'	--holds helper functions
local configurations = require 'configurations'	--holds catalog of configurations

-- solver variables

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
maxFrames = nil			-- enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	-- whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false
fixedDT = .001

heatMapVariable = 'DENSITY'
heatMapColorScale = 2

-- TODO AMD card has trouble with mirror and periodic boundaries ... probably all boundaries
boundaryMethods = {
	{min='FREEFLOW', max='FREEFLOW'},
	{min='FREEFLOW', max='FREEFLOW'},
	{min='FREEFLOW', max='FREEFLOW'},
}

-- gravity is specific to the Euler fluid equation solver
useGravity = false
gravitationalConstant = 1	-- G = 6.67384e-11 m^3 kg^-1 s^-2 TODO meaningful units please
-- used for gravitation Poisson solver
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

-- numerical relativity, speed of light, in grid units
speedOfLightInM = 299792458
speedOfLight = speedOfLightInM

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
size = {16, 16, 16}
vectorFieldResolution = 16
--]]
-- [[ 2D
size = {512, 512}
--]]
--[[ 1D
size = {1024}
heatMapColorScale = .25
--]]


camera = {}



-- [[ Euler

-- uncomment one:
solverName = 'EulerBurgers'
--solverName = 'EulerHLL'		-- needs slope limiter support
--solverName = 'EulerHLLC'		-- needs slope limiter support
--solverName = 'EulerRoe'		-- fails on Colella-Woodward 2-wave problem, but works on all the configurations
--solverName = 'SRHDRoe'		-- not yet

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
function calcSolid(x,y,z)
	if x > -.275 and x < -.225 and y > -.4 and y < .4 then
		return 1
	end
end
--]=]
--[=[ loading images from Lua ...
--solidFilename = 'test-solid.png'
--]=]

configurations['Sod']()
--configurations['Sphere']()
--configurations['Square Cavity']()
--configurations['Kelvin-Hemholtz']()
--configurations['Rayleigh-Taylor']()
--configurations['Shock Bubble Interaction']()
--configurations['Flow Around Cylinder']()
--configurations['Forward Facing Step']()
--configurations['Double Mach Reflection']()
--configurations['Spiral Implosion']()
--configurations['self-gravitation test 1']()
--configurations['Colella-Woodward']()
--configurations['Configuration 6']()
--]]

--[[ MHD

--solverName = 'MHDBurgers'		-- a mathematically-flawed version works with Orszag-Tang and Brio-Wu, and some hydro problems too.  fixing the math error causes it to break.
--solverName = 'MHDHLLC'		-- needs 2nd order support, suffers same as EulerHLLC
solverName = 'MHDRoe'			-- suffers from negative pressure with magnetic problems.  solves fluid-only problems fine.

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
--solverName = 'BSSNOKRoe'		-- not yet.  TODO copy from the gravitation wave sim project, but that BSSNOK+Roe solver isn't as accurate as it should be
-- TODO ImplicitIncompressibleNavierStokes	<- from my GPU fluid sim Lua+GLSL project
--solverName = 'BSSNOKFiniteDifference'	-- doing the bare minimum to consider this a solver.  I could use this to make a coefficient matrix (application function) and, from there, make the implicit solver.

size = {1024}
heatMapColorScale = 128
configurations['NR Gauge Shock Waves']{unitDomain=false}
boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
heatMapVariable = 'ALPHA'
camera.zoom = 1/300
camera.pos = {150,150}
--]]

--[[ ADM 2D Spherical
solverName = 'ADM2DSpherical'	-- not yet
-- no test cases yet?
-- I want to get rid of this one.  and the 1D ADM as well -- just one Bona-Masso ADM implementation is enough (I think) unless I should have separate ones for shift/less and mass/less
--]]

--[[ ADM (3D)
solverName = 'ADM3DRoe'
		
--size = {1024} heatMapColorScale = 128
size = {256, 256} heatMapColorScale = 1
--size = {16, 16, 16} heatMapColorScale = 1
--configurations['NR Gauge Shock Waves']{unitDomain=false}
--configurations['NR Gauge Shock Waves']{unitDomain=true}	-- for 2D,3D make sure unitDomain=true ... and now not working in 1D as well
configurations['NR Alcubierre Warp Bubble']()	-- ...needs shift vector support
--configurations['NR Schwarzschild Black Hole']()
--configurations['NR Stellar']()
--configurations['NR Stellar']{bodies={{pos = {0,0,0}, radius = .1, mass = .001, density=0, pressure=0}}}	-- planet plucked out of existence

--[=[
--[==[
earth radius = 6.37101e+6 m
domain: 10x radius = 6.37101e+7 m
earth mass = 5.9736e+24 kg = 5.9736e+24 * 6.6738480e-11 / 299792458^2 m
earth mass = Em * G / c^2 in meters
--]==]
local G = 6.6738480e-11	-- kg m^3/s^2
local c = 299792458	-- m/s
-- massInRadii is the order of 1e-9.  much more subtle than the default 1e-3 demo
local earth = {radiusInM = 6.37101e+6, massInKg = 5.9736e+24}
-- massInRadii is on the order of 1e-6
local sun = {radiusInM = 6.960e+8, massInKg = 1.9891e+30}

local planet = sun
planet.massInM = planet.massInKg * G / c^2
planet.massInRadii = planet.massInM / planet.radiusInM
planet.radiusInCoords = .1
planet.massInCoords = planet.massInRadii * planet.radiusInCoords
for k,v in pairs(planet) do print(k,v) end

local gridUnitsInM = planet.radiusInM / planet.radiusInCoords
speedOfLight = speedOfLightInM / gridUnitsInM
configurations['NR Stellar']{bodies={{pos = {0,0,0}, radius = planet.radiusInCoords, mass = planet.massInCoords}}}
--]=]

boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
--showHeatMap = false
heatMapVariable = 'ALPHA'
--fixedDT = .125
--useFixedDT = true
graphVariables = {'ALPHA', 'GAMMA', 'K'}	-- which variables to graph.  none = all.
graphStep = {4,4,4}
graphScale = 1
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

cfl = cfl or 1/#size
