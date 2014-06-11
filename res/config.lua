useGPU = true
sizeX = 512
sizeY = 512
--maxFrames = 1		--enable to automatically pause the solver after this many frames.  useful for comparing solutions
xmin = -.5
xmax = .5
ymin = -.5
ymax = .5
useFixedDT = false
cfl = .5
displayMethod = 0	--density
displayScale = 2
boundaryMethod = 0	--mirror
useGravity = false
noise = 0
gamma = 1.4

-- some helper functions

local function primsToState(density, vx, vy, energyTotal)
	return density, vx * density, vy * density, energyTotal * density
end

local function crand() return math.random() * 2 - 1 end

local function initVelocity()
	return crand() * noise, crand() * noise
end

local function energyKineticForVelocity(vx, vy)
	return .5  * (vx * vx + vy * vy)
end

local function energyInternalForPressure(pressure, density)
	return pressure / ((gamma - 1) * density)
end

--[[ circle -- http://www.cfd-online.com/Wiki/Explosion_test_in_2-D
function initState(x,y)
	local rSq = x * x + y * y
	local inside = rSq <= .2*.2
	local density = inside and 1 or .125
	local vx, vy = initVelocity()
	local pressure = inside and 1 or .1
	local energyKinetic = energyKineticForVelocity(vx, vy)
	local energyInternal = energyInternalForPressure(pressure, density)
	local energyTotal = energyKinetic + energyInternal
	return primsToState(density, vx, vy, energyTotal)
end
--]]	

--[[ square shock wave
boundaryMethod = 1
function initState(x,y)
	local inside = x < -.2 and y < -.2
	local density = inside and 1 or .1
	local vx, vy = initVelocity()
	local energyKinetic = energyKineticForVelocity(vx, vy)
	local energyInternal = 1
	local energyTotal = energyKinetic + energyInternal
	return primsToState(density, vx, vy, energyTotal)
end
--]]

-- [[ gravity potential test - equilibrium
useGravity = true
boundaryMethod = 2
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
	local density = inside and 1 or .1
	local vx, vy = initVelocity()
	local energyKinetic = energyKineticForVelocity(vx,vy)
	local pressure = 1
	local energyInternal = energyInternalForPressure(pressure, density)
	local energyTotal = energyKinetic + energyInternal
	return primsToState(density, vx, vy, energyTotal)
end
--]]

