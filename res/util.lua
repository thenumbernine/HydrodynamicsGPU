
function crand() return math.random() * 2 - 1 end

function clamp(x,min,max) return math.max(min, math.min(max, x)) end

local function getSpecificEnergyKinetic(velocityX, velocityY, velocityZ)
	return .5  * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ)
end

local function getSpecificEnergyInternalForPressure(pressure, density)
	return pressure / ((gamma - 1) * density)
end

local function getMagneticFieldEnergy(magneticFieldX, magneticFieldY, magneticFieldZ)
	return .5 * (magneticFieldX * magneticFieldX + magneticFieldY * magneticFieldY + magneticFieldZ * magneticFieldZ) / vaccuumPermeability
end

local function primsToState(density, velocityX, velocityY, velocityZ, energyTotal, magneticFieldX, magneticFieldY, magneticFieldZ, potentialEnergy)
	return 
		-- density
		density,
		-- momentum
		velocityX * density,
		velocityY * density,
		velocityZ * density,
		-- magnetic field
		magneticFieldX,
		magneticFieldY,
		magneticFieldZ,
		-- total energy
		energyTotal,
		-- potential energy
		potentialEnergy
end

--[=[
table-driven so may be slower, but much more readable 
args:
	density (required)
	velocityX, velocityY (optional) velocity
	noise (optional) noise to add to velocity
	magneticFieldX, magneticFieldY, magneticFieldZ (optional) magnetic field
	magneticFieldNoise (optional) noise to add to magnetic field
	pressure				\_ one of these two
	specificEnergyInternal	/
	potentialEnergy (optional)
--]=]
function buildStateEuler(args)
	local dim = #size
	local noise = args.noise or 0
	local magneticFieldNoise = args.magneticFieldNoise or 0
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
	local potentialEnergy = args.potentialEnergy or 0
	-- dont' add potential energy to total energy.  
	-- it is added to total energy after self-gravity optionally calculates it (if enabled)
	local energyTotal = density * (specificEnergyKinetic + specificEnergyInternal) + magneticFieldEnergy
	return primsToState(density, velocityX, velocityY, velocityZ, energyTotal, magneticFieldX, magneticFieldY, magneticFieldZ, potentialEnergy)
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

function buildSelfGravitationState(x,y,z,args)
	local minDistSq = math.huge
	local minSource
	local inside = false
	for _,source in ipairs(args.sources) do
		local sx, sy, sz = unpack(source.center)
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
	local dx = x - minSource.center[1]
	local dy = y - minSource.center[2]
	local dz = z - minSource.center[3]
	local noise = math.exp(-100 * (dx * dx + dy * dy + dz * dz))
	if inside and minSource.inside then
		return minSource.inside(dx, dy, dz)
	end
	return buildStateEuler{
		density = inside and (
				(minSource and minSource.density) or 1
			) or .1,
		pressure = 1,
		velocityX = .01 * noise * crand(),
		velocityY = .01 * noise * crand(),
		velocityZ = .01 * noise * crand(),
	}
end

