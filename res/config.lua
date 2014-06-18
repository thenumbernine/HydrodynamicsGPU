

	-- solver variables


solverName = 'Burgers'
dim = 3
useGPU = true
-- Burgers is running 1024x1024 at 35fps, Roe is running 512x512 at 35fps
--maxFrames = 1		--enable to automatically pause the solver after this many frames.  useful for comparing solutions
xmin = {-.5, -.5, -.5}
xmax = {.5, .5, .5}
useFixedDT = false
cfl = .5
displayMethod = 'density'
displayScale = 2
boundaryMethods = {'periodic', 'periodic', 'periodic'}
useGravity = false
noise = .01
gamma = 1.4

-- size according to dim and solver
if dim == 3 then
	size = {64, 64, 64}
elseif dim == 2 then
	if solverName == 'Burgers' then
		size = {1024, 1024}
	elseif solverName == 'Roe' then
		size = {512, 512}
	else
		error('unknown solver name '..solverName)
	end
else
	error('unknown dim '..dim)
end


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
	local vy = args.vy or crand() * noise
	local vz = args.vz or crand() * noise
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

--[[ square shock wave / 2D Sod test
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x)
	local inside = x[1] <= 0 and x[2] <= 0 and x[3] <= 0
	return buildState{
		density = inside and 1 or .1,
		energyInternal = 1,
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

-- [[ gravity potential test - equilibrium - some Rayleigh-Taylor
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
