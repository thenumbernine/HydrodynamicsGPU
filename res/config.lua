

	-- solver variables


solverName = 'Burgers'
useGPU = true
-- Burgers is running 1024x1024 at 35fps, Roe is running 512x512 at 35fps
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
useGravity = false 
noise = 0	--.01
magneticFieldNoise = 0
gamma = 1.4

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
-- roe 3d is crashing on build.
-- burgers 3d with flux limiter is crashing on build, but without flux limiter works fine
size = {64, 64, 64}
--]]
-- [[ 2D
-- max burgers size with 4 channels: 4096x4096
-- max roe size with 4 channels: 1024x1024
-- max burgers size with 8 channels: 2048x2048
-- roe with 8 channels: 512x512 
size = {1024, 1024}
--]]
--[[ 1D
size = {1024}
displayScale = .25
--]]


--[[ testing HLL
solverName = 'HLL'
useFixedDT = true
fixedDT = .01
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

local function primsToState(density, velocityX, velocityY, velocityZ, magneticFieldX, magneticFieldY, magneticFieldZ, energyTotal)
	return {density, velocityX * density, velocityY * density, velocityZ * density, energyTotal, magneticFieldX, magneticFieldY, magneticFieldZ}
end

--[=[
table-driven so may be slower, but much more readable 
args:
	density (required)
	velocityX, velocityY (optional) velocity
	pressure				\_ one of these two
	specificEnergyInternal	/
--]=]
local function buildState(args)
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
	return {
		density = density,
		velocity = {velocityX, velocityY, velocityZ},
		magneticField = {magneticFieldX, magneticFieldY, magneticFieldZ},
		energyTotal = energyTotal,
	}
	--return primsToState(density, velocityX, velocityY, velocityZ, magneticFieldX, magneticFieldY, magneticFieldZ, energyTotal)
end


	-- initial state descriptions


--[[ 1D advect wave
function initState(x)
	local rSq = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
	return buildState{
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
	return buildState{
		density = inside and 1 or .1,
		pressure = inside and 1 or .5,	--1 : .1 works for 2d but not 3d
	}
end
--]]	

-- [[ Sod test
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x)
	local inside = x[1] <= 0 and x[2] <= 0 and x[3] <= 0
	return buildState{
		density = inside and 1 or .1,
		specificEnergyInternal = 1,
	}
end
--]]

--[[ Brio Wu
-- http://www.astro.uni-bonn.de/~jmackey/jmac/node7.html
gamma = 2
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x)
	local lhs = x[1] < 0
	return buildState{
		density = lhs and 1 or .125,
		pressure = lhs and 1 or .1,
		magneticFieldX = .75,
		magneticFieldY = lhs and 1 or -1,
	}
end
--]]

--[[ Colella-Woodward interacting blast wave problem
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x)
	local pressure
	if x[1] < -.4 then
		pressure = 1000
	elseif x[1] < .4 then
		pressure = .01
	else
		pressure = 100
	end
	return buildState{
		density = 1,
		velocityX = 0, velocityY = 0, velocityZ = 0,
		pressure = pressure,
	}
end
--]]

--[[ Kelvin-Hemholtz
noise = size[1] * 2e-5
--solverName = 'Roe'	--Burgers is having trouble... hmm...
function initState(x)
	local inside = x[2] > -.25 and x[2] < .25
	local theta = (x[1] - xmin[1]) / (xmax[1] - xmin[1]) * 2 * math.pi
	if dim >= 3 then 
		theta = theta * (x[3] - xmin[3]) / (xmax[3] - xmin[3]) 
	end
	return buildState{
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
noise = 0	--noise must be 0 at borders with freeflow or we'll get waves at the edges
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
function initState(pos)
	local x, y, z = unpack(pos)
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
	--noise must be 0 at borders with freeflow or we'll get waves at the edges
	local dx = x - minSource[1]
	local dy = y - minSource[2]
	local dz = z - minSource[3]
	local noise = math.exp(-100 * (dx * dx + dy * dy + dz * dz))
	return buildState{
		density = inside and 1 or .1,
		pressure = 1,
		velocityX = .01 * noise * crand(),
		velocityY = .01 * noise * crand(),
		velocityZ = .01 * noise * crand(),
	}
end
--]]

