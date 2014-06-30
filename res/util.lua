
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
function buildStateEuler(args)
	local dim = #size
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

function buildStateEulerQuadrant(x,y,z,args)
	if y > 0 then
		if x > 0 then
			return buildStateEuler(args.q1)
		else
			return buildStateEuler(args.q2)
		end
	else
		if x < 0 then
			return buildStateEuler(args.q3)
		else
			return buildStateEuler(args.q4)
		end
	end
end

