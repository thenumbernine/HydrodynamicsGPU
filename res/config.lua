solverName = 'Burgers'
useGPU = true
-- Burgers is running 1024x1024 at 35fps, Roe is running 512x512 at 35fps
sizeX, sizeY = 1024, 1024
--maxFrames = 1		--enable to automatically pause the solver after this many frames.  useful for comparing solutions
xmin = -.5
xmax = .5
ymin = -.5
ymax = .5
useFixedDT = false
cfl = .5
displayMethod = displayMethods.density
displayScale = 2
boundaryMethod = boundaryMethods.periodic
useGravity = false
noise = 0
gamma = 1.4

-- some helper functions

local function crand() return math.random() * 2 - 1 end

local function energyKineticForVelocity(vx, vy)
	return .5  * (vx * vx + vy * vy)
end

local function energyInternalForPressure(pressure, density)
	return pressure / ((gamma - 1) * density)
end

local function primsToState(density, vx, vy, energyTotal)
	return density, vx * density, vy * density, energyTotal * density
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
	local energyKinetic = energyKineticForVelocity(vx, vy)
	local energyInternal = args.energyInternal or energyInternalForPressure(assert(args.pressure, "you need to provide either energyInternal or pressure"), density)
	local energyTotal = energyKinetic + energyInternal 
	return primsToState(density, vx, vy, energyTotal)
end

--[[ circle -- http://www.cfd-online.com/Wiki/Explosion_test_in_2-D
function initState(x,y)
	local rSq = x * x + y * y
	local inside = rSq <= .2*.2
	return buildState{
		density = inside and 1 or .1,
		pressure = inside and 1 or .1
	}
end
--]]	

-- [[ square shock wave
boundaryMethod = boundaryMethods.mirror
function initState(x,y)
	local inside = x < -.2 and y < -.2
	return buildState{
		density = inside and 1 or .1,
		energyInternal = 1,
	}
end
--]]

--[[ gravity potential test - equilibrium
useGravity = true
boundaryMethod = boundaryMethods.freeflow
local sources = {{0,0}}
function initState(x,y)
	local minDistSq = math.huge
	for _,source in ipairs(sources) do
		local sx, sy = unpack(source)
		local dx = sx - x
		local dy = sy - y
		distSq = dx * dx + dy * dy
		if distSq < minDistSq then minDistSq = distSq end
	end
	local minDist = math.sqrt(minDistSq)
	local inside = minDist < .2
	return buildState{
		density = inside and 1 or .1,
		pressure = 1,
	}
end
--]]

