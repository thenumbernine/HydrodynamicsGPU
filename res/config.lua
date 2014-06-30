

	-- solver variables

--[[
options:
EulerBurgers	- works for 1D, 2D, 3D
EulerRoe - works for 1D, compiler crashes for 2D and 3D
EulerHLL - works for 1D, compiler crashes for 2D and 3D
MHDRoe - left eigenvectors not finished
ADMRoe - compiler crashes
--]]
solverName = 'EulerRoe'

--[[
options:
DonorCell
Superbee
LaxWendroff
BeamWarming
Fromm
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
VanLeer
MonotizedCentral
Superbee
BarthJespersen
--]]
slopeLimiterName = 'Superbee'

useGPU = true
-- EulerBurgers is running 1024x1024 at 35fps, Roe is running 512x512 at 35fps
--maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false 
fixedDT = .01
cfl = .5
displayMethod = 'density'
displayScale = 2
boundaryMethods = {'periodic', 'periodic', 'periodic'}

-- gravity is specific to the Euler fluid equation solver
useGravity = false 

magneticFieldNoise = 0
gamma = 1.4

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
-- roe 3d is crashing on build.
-- burgers 3d with flux limiter is crashing on build, but without flux limiter works fine
size = {128, 128, 128}
--]]
--[[ 2D
-- max burgers size with 4 channels: 4096x4096
-- max roe size with 4 channels: 1024x1024
-- max burgers size with 8 channels: 2048x2048
-- roe with 8 channels: 512x512 
size = {512, 512}
--]]
-- [[ 1D
size = {1024}
displayScale = .25
--]]



local dim = #size

	-- helper functions


local function crand() return math.random() * 2 - 1 end

local function clamp(x,min,max) return math.max(min, math.min(max, x)) end

local function getSpecificEnergyKinetic(velocityX, velocityY, velocityZ)
	return .5  * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ)
end

local function getSpecificEnergyInternalForPressure(pressure, density)
	return pressure / ((gamma - 1) * density)
end

local MU0 = 1
local function getMagneticFieldEnergy(magneticFieldX, magneticFieldY, magneticFieldZ)
	return .5 * (magneticFieldX * magneticFieldX + magneticFieldY * magneticFieldY + magneticFieldZ * magneticFieldZ) / MU0
end

local function primsToState(density, velocityX, velocityY, velocityZ, energyTotal, magneticFieldX, magneticFieldY, magneticFieldZ)
	return density, velocityX * density, velocityY * density, velocityZ * density, energyTotal	--, magneticFieldX, magneticFieldY, magneticFieldZ
end

--[=[
table-driven so may be slower, but much more readable 
args:
	density (required)
	velocityX, velocityY (optional) velocity
	noise (optional) noise to add to velocity
	pressure				\_ one of these two
	specificEnergyInternal	/
--]=]
local function buildStateEuler(args)
	local noise = args.noise or 0
	local density = assert(args.density)
	local velocityX = args.velocityX or crand() * noise
	local velocityY = dim <= 1 and 0 or (args.velocityY or crand() * noise)
	local velocityZ = dim <= 2 and 0 or (args.velocityZ or crand() * noise)
	local magneticFieldX = args.magneticFieldX or crand() * magneticFieldNoise
	local magneticFieldY = args.magneticFieldY or crand() * magneticFieldNoise
	local magneticFieldZ = args.magneticFieldZ or crand() * magneticFieldNoise
	local specificEnergyKinetic = getSpecificEnergyKinetic(velocityX, velocityY, velocityZ)
	local specificEnergyInternal = args.specificEnergyInternal or getSpecificEnergyInternalForPressure(assert(args.pressure, "you need to provide either specificEnergyInternal or pressure"), density)
	local magneticFieldEnergy = getMagneticFieldEnergy(magneticFieldX, magneticFieldY, magneticFieldZ)
	local energyTotal = density * (specificEnergyKinetic + specificEnergyInternal) + magneticFieldEnergy
	return primsToState(density, velocityX, velocityY, velocityZ, energyTotal, magneticFieldX, magneticFieldY, magneticFieldZ)
end


	-- Euler equation initial states


--[[ 1D advect wave
function initState(x,y,z)
	local rSq = x * x + y * y + z * z
	return buildStateEuler{
		velocityX = 1,
		density = math.exp(-100*rSq) + 1,
		pressure = 1,
	}
end
--]]

--[[ circle -- http://www.cfd-online.com/Wiki/Explosion_test_in_2-D
function initState(x)
	local rSq = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
	local inside = rSq <= .2*.2
	return buildStateEuler{
		density = inside and 1 or .1,
		pressure = inside and 1 or .5,	--1 : .1 works for 2d but not 3d
	}
end
--]]	

-- [[ Sod test
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	local inside = x <= 0 and y <= 0 and z <= 0
	return buildStateEuler{
		density = inside and 1 or .1,
		specificEnergyInternal = 1,
	}
end
--]]

--[[ Brio Wu
-- http://www.astro.uni-bonn.de/~jmackey/jmac/node7.html
-- still in the works
gamma = 2
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	local lhs = x < 0
	return buildStateEuler{
		density = lhs and 1 or .125,
		pressure = lhs and 1 or .1,
		magneticFieldX = .75,
		magneticFieldY = lhs and 1 or -1,
	}
end
--]]

--[[ Colella-Woodward interacting blast wave problem
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	local pressure
	if x < -.4 then
		pressure = 1000
	elseif x < .4 then
		pressure = .01
	else
		pressure = 100
	end
	return buildStateEuler{
		density = 1,
		velocityX = 0, velocityY = 0, velocityZ = 0,
		pressure = pressure,
	}
end
--]]

--[[ Kelvin-Hemholtz
--solverName = 'Roe'	--EulerBurgers is having trouble... hmm...
function initState(x,y,z)
	local inside = y > -.25 and y < .25
	local theta = (x - xmin[1]) / (xmax[1] - xmin[1]) * 2 * math.pi
	if dim >= 3 then 
		theta = theta * (z - xmin[3]) / (xmax[3] - xmin[3]) 
	end
	local noise = size[1] * 2e-5
	return buildStateEuler{
		density = inside and 2 or 1,
		velocityX = math.cos(theta) * noise + (inside and -.5 or .5),
		velocityY = math.sin(theta) * noise,
		velocityZ = math.sin(theta) * noise,
		pressure = 2.5,
	}
end
--]]

--[[ gravity potential test - equilibrium - some Rayleigh-Taylor
useGravity = true
boundaryMethods = {'freeflow', 'freeflow', 'freeflow'}
local sources = {
-- [=[ single source
	{0, 0, 0, radius = .2},
--]=]
--[=[ two
	{-.25, 0, 0, radius = .1},
	{.25, 0, 0, radius = .1},
--]=]
--[=[ multiple sources
	{.25, .25, 0, radius = .1},
	{-.25, .25, 0, radius = .1},
	{.25, -.25, 0, radius = .1},
	{-.25, -.25, 0, radius = .1},
--]=]
}
function initState(x,y,z)
	local minDistSq = math.huge
	local minSource
	local inside = false
	for _,source in ipairs(sources) do
		local sx, sy, sz = unpack(source)
		local dx = sx - x
		local dy = sy - y
		local dz = sz - z
		distSq = dx * dx + dy * dy + dz * dz
		if distSq < minDistSq then
			minDistSq = distSq
			minSource = source
			if distSq < source.radius * source.radius then
				inside = true
				break
			end
		end
	end
	local dx = x - minSource[1]
	local dy = y - minSource[2]
	local dz = z - minSource[3]
	local noise = math.exp(-100 * (dx * dx + dy * dy + dz * dz))
	return buildStateEuler{
		density = inside and 1 or .1,
		pressure = 1,
		velocityX = .01 * noise * crand(),
		velocityY = .01 * noise * crand(),
		velocityZ = .01 * noise * crand(),
	}
end
--]]


		-- 1D ADM equation initial state

--[[
xmin = {-30, -30, -30}
xmax = {30, 30, 30}
local xmid = (xmax[1] + xmin[1]) * .5
adm_BonaMasso_f = 1
function initState(x,y,z)
	x = (x - xmid) / ((xmax[1] - xmid) / 3)
	local h = math.exp(-x*x); 
	local dh_dx = -2 * x * h;
	local d2h_dx2 = 2 * h * (2 * x * x - 1);
	local g = 1 - dh_dx * dh_dx;
	local D_g = -2 * dh_dx * d2h_dx2 / g;
	local KTilde = -d2h_dx2 / g;
	local f = adm_BonaMasso_f;
	local D_alpha = math.sqrt(f) * KTilde;
	return D_alpha, D_g, KTilde, 0, 0, 0, 0, 0	
end
--]]

