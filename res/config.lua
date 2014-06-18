

	-- solver variables


solverName = 'Burgers'
useGPU = true
dim = 2
-- Burgers is running 1024x1024 at 35fps, Roe is running 512x512 at 35fps
size = {512, 512, 512} --{1024, 1024, 1024}
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


	-- helper functions


local function crand() return math.random() * 2 - 1 end

local function clamp(x,min,max) return math.max(min, math.min(max, x)) end

local function energyKineticForVelocity(vx, vy)
	return .5  * (vx * vx + vy * vy)
end

local function energyInternalForPressure(pressure, density)
	return pressure / ((gamma - 1) * density)
end

local function primsToState(density, vx, vy, energyTotal)
	return {density, vx * density, vy * density, energyTotal * density}
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


	-- initial state descriptions


--[[ circle -- http://www.cfd-online.com/Wiki/Explosion_test_in_2-D
function initState(pos)
	local x, y = unpack(pos)
	local rSq = x * x + y * y
	local inside = rSq <= .2*.2
	return buildState{
		density = inside and 1 or .1,
		pressure = inside and 1 or .1
	}
end
--]]	

--[[ square shock wave / 2D Sod test
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(pos)
	local x, y = unpack(pos)
	local inside = x < 0 and y < 0
	return buildState{
		density = inside and 1 or .1,
		energyInternal = 1,
	}
end
--]]

--[[ Kelvin-Hemholtz
noise = sizeX*2e-5
solverName = 'Roe'	--Burgers is having trouble... hmm...
function initState(pos)
	local x, y = unpack(pos)
	local inside = y > -.25 and y < .25
	local theta = (x - xmin) / (xmax - xmin) * 2 * math.pi
	return buildState{
		density = inside and 2 or 1,
		vx = math.cos(theta) * noise + (inside and -.5 or .5),
		vy = math.sin(theta) * noise,
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
	{0,0, radius=.2},
--]=]
--[=[ two
	{-.25, 0, radius=.1},
	{.25, 0, radius=.1},
--]=]
--[=[ multiple sources
	{.25,.25, radius=.1},
	{-.25,.25, radius=.1},
	{.25,-.25, radius=.1},
	{-.25,-.25, radius=.1},
--]=]
}
function initState(pos)
	local x, y = unpack(pos)
	local minDistSq = math.huge
	local minSource
	local inside = false
	for _,source in ipairs(sources) do
		local sx, sy = unpack(source)
		local dx = sx - x
		local dy = sy - y
		distSq = dx * dx + dy * dy
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
	local noise = math.exp(-100 * (x * x + y * y))
	return buildState{
		density = inside and 1 or .1,
		pressure = 1,
		vx = .01 * noise * crand(),
		vy = .01 * noise * crand(),
	}
end
--]]

