

	-- solver variables


solverName = 'Roe'
useGPU = true
-- Burgers is running 1024x1024 at 35fps, Roe is running 512x512 at 35fps
--maxFrames = 1			--enable to automatically pause the solver after this many frames.  useful for comparing solutions.  push 'u' to toggle update pause/play.
showTimestep = false	--whether to print timestep.  useful for debugging.  push 't' to toggle.
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false
cfl = .5
displayMethod = 'density'
displayScale = 2
boundaryMethods = {'periodic', 'periodic', 'periodic'}
useGravity = false 
noise = 0	--.01
gamma = 1.4

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
-- roe 3d is crashing on build.
-- burgers 3d with flux limiter is crashing on build, but without flux limiter works fine
size = {64, 64, 64}
--]]
--[[ 2D
-- max burgers size with 4 channels: 4096x4096
-- max roe size with 4 channels: 1024x1024
-- max burgers size with 8 channels: 2048x2048
-- roe with 8 channels is crashing on build
size = {256, 256}
--]]
-- [[ 1D
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

local function energyKineticForVelocity(vx, vy, vz)
	return .5  * (vx * vx + vy * vy + vz * vz)
end

local function energyInternalForPressure(pressure, density)
	return pressure / ((gamma - 1) * density)
end

local function primsToState(density, vx, vy, vz, energyTotal)
	return {density, vx * density, vy * density, vz * density, energyTotal * density}
end

--[=[
table-driven so may be slower, but much more readable 
args:
	density (required)
	vx, vy (optional) velocity
	pressure		\_ one of these two
	energyInternal	/
--]=]
local function buildState(args)
	local density = assert(args.density)
	local vx = args.vx or crand() * noise
	local vy = dim <= 1 and 0 or (args.vy or crand() * noise)
	local vz = dim <= 2 and 0 or (args.vz or crand() * noise)
	local energyKinetic = energyKineticForVelocity(vx, vy, vz)
	local energyInternal = args.energyInternal or energyInternalForPressure(assert(args.pressure, "you need to provide either energyInternal or pressure"), density)
	local energyTotal = energyKinetic + energyInternal 
	return primsToState(density, vx, vy, vz, energyTotal)
end


	-- initial state descriptions


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
		energyInternal = 1,
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
		vx = 0, vy = 0, vz = 0,
		pressure = pressure,
	}
end
--]]

--[[ Kelvin-Hemholtz
noise = size[1] * 2e-5
solverName = 'Roe'	--Burgers is having trouble... hmm...
function initState(x)
	local inside = x[2] > -.25 and x[2] < .25
	local theta = (x[1] - xmin[1]) / (xmax[1] - xmin[1]) * 2 * math.pi
	if dim >= 3 then 
		theta = theta * (x[3] - xmin[3]) / (xmax[3] - xmin[3]) 
	end
	return buildState{
		density = inside and 2 or 1,
		vx = math.cos(theta) * noise + (inside and -.5 or .5),
		vy = math.sin(theta) * noise,
		vz = math.sin(theta) * noise,
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
		vx = .01 * noise * crand(),
		vy = .01 * noise * crand(),
		vz = .01 * noise * crand(),
	}
end
--]]

