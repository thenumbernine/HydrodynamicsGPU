local table = require 'ext.table'
local range = require 'ext.range'
local symmath = require 'symmath'
local mat33 = require 'mat33'

function crand() return math.random() * 2 - 1 end

function clamp(x,min,max) return math.max(min, math.min(max, x)) end

local function getSpecificEnergyKinetic(velocityX, velocityY, velocityZ)
	return .5  * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ)
end

local function getSpecificEnergyInternalForPressure(pressure, density)
	return pressure / ((defs.idealGas_heatCapacityRatio - 1) * density)
end

local function getMagneticFieldEnergy(magneticFieldX, magneticFieldY, magneticFieldZ)
	return .5 * (magneticFieldX * magneticFieldX + magneticFieldY * magneticFieldY + magneticFieldZ * magneticFieldZ) / defs.mhd_vacuumPermeability
end

--[=[
table-driven so may be slower, but much more readable
args:
	x,y,z,
	density (required)
	velocityX, velocityY (optional) velocity
	noise (optional) noise to add to velocity
	magneticFieldX, magneticFieldY, magneticFieldZ (optional) magnetic field
	magneticFieldNoise (optional) noise to add to magnetic field
	pressure				\_ one of these two
	specificEnergyInternal	/
	potentialEnergy (optional)
	solid
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
	local solid = args.solid or 0	-- 0 or 1

	if calcSolid then
		local x = assert(args.x)
		local y = assert(args.y)
		local z = assert(args.z)
		solid = calcSolid(x,y,z)
	end

	return
		density,
		-- momentum
		density * velocityX,
		density * velocityY,
		density * velocityZ,
		-- magnetic field
		-- TODO put this last, so I can keep tacking crap on the end without readjusting indexes
		magneticFieldX,
		magneticFieldY,
		magneticFieldZ,
		-- energy
		energyTotal,
		potentialEnergy,
		-- solid or not
		-- TODO get rid of this? solid sucks at the moment.
		solid
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
		local sx, sy, sz = table.unpack(source.center)
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

--[[
args:
	args = {x,y,z} spatial basis variables
	alpha = lapse expression
	(beta isn't used yet)
	gamma = 3-metric
	K = extrinsic curvature

	gamma & K are stored as {xx,xy,xz,yy,yz,zz}

	density = lua function
--]]
function initNumRel(args)
	local vars = assert(args.vars)

	local exprs = table{
		alpha = assert(args.alpha),
		gamma = {table.unpack(args.gamma)},
		K = {table.unpack(args.K)}
	}
	assert(#exprs.gamma == 6)
	assert(#exprs.K == 6)

	local function toExpr(expr, name)
		if type(expr) == 'number' then expr = symmath.Constant(expr) end
		if type(expr) == 'table' then
			if not expr.isa then
				expr = table.map(expr, toExpr)
			end
			-- simplify?
			if symmath.Expression:isa(expr) then
				expr = expr()
			end
		end
		return expr, name
	end
	print('converting everything to expressions...')
	exprs = table.map(exprs, toExpr)
	print('...done converting everything to expressions')

	local function buildCalc(expr, name)
		assert(type(expr) == 'table')
		if symmath.Expression:isa(expr) then
			expr = expr()
			print('compiling '..expr)
			return expr:compile(vars), name
		end
		return table.map(expr, buildCalc), name
	end

	-- ADM
	if solverName == 'ADM1DRoe' then
		exprs.gamma = exprs.gamma[1]	-- only need g_xx
		exprs.A = (exprs.alpha:diff(vars[1]) / exprs.alpha)()	-- only need a_x
		exprs.D = (exprs.gamma:diff(vars[1])/exprs.gamma)()	-- only need D_g = ln (g_xx),x = 2 D_xxx / g_xx
		exprs.K = exprs.K[1]	-- only need K_xx
		print('compiling expressions...')
		local calc = table.map(exprs, buildCalc)
		print('...done compiling expressions')
		initState = function(x,y,z)
			local alpha = calc.alpha(x,y,z)
			local gamma = calc.gamma(x,y,z)
			local A = calc.A(x,y,z)
			local D = calc.D(x,y,z)
			local K = calc.K(x,y,z)
			return alpha, gamma, A, D, K
		end
	elseif solverName == 'ADM3DRoe' then

		-- for complex computations it might be handy to extract the determinant first ...
		-- or even just perform a numerical inverse ...
		if not args.useNumericInverse then
			print('inverting spatial metric...')
			exprs.gammaU = {mat33.inv(exprs.gamma:unpack())}
			print('...done inverting spatial metric')
		end

		-- this takes forever.  why is that?  differentiation?
		print('building metric partials...')
		exprs.D = table.map(vars, function(x_k)
			return table.map(exprs.gamma, function(g_ij)
				print('differentiating '..g_ij)
				return (g_ij:diff(x_k)/2)()
			end)
		end)
		print('...done building metric partials')

		print('building lapse partials...')
		exprs.A = table.map(vars, function(var)
			return (exprs.alpha:diff(var) / exprs.alpha)()
		end)
		print('...done building lapse partials')

		print('compiling expressions...')
		local calc = table.map(exprs, buildCalc)
		print('...done compiling expressions')

		local densityFunc = args.density or 0
		if symmath.Expression:isa(densityFunc) then
			densityFunc = densityFunc():compile(vars)
		end

		local pressureFunc = args.pressure or 0
		if symmath.Expression:isa(pressureFunc) then
			pressureFunc = pressureFunc():compile(vars)
		end

		initState = function(x,y,z)

			local alpha = calc.alpha(x,y,z)
			local gamma = calc.gamma:map(function(g_ij) return g_ij(x,y,z) end)
			local A = calc.A:map(function(A_i) return A_i(x,y,z) end)
			local D = calc.D:map(function(D_i) return D_i:map(function(D_ijk) return D_ijk(x,y,z) end) end)
			local gammaU = args.useNumericInverse and table{mat33.inv(gamma:unpack())} or calc.gammaU:map(function(gammaUij) return gammaUij(x,y,z) end)

			local function sym3x3(m,i,j)
				local m_xx, m_xy, m_xz, m_yy, m_yz, m_zz = m:unpack()
				if i==1 then
					if j==1 then return m_xx end
					if j==2 then return m_xy end
					if j==3 then return m_xz end
				elseif i==2 then
					if j==1 then return m_xy end
					if j==2 then return m_yy end
					if j==3 then return m_yz end
				elseif i==3 then
					if j==1 then return m_xz end
					if j==2 then return m_yz end
					if j==3 then return m_zz end
				end
				error'here'
			end
			local V = range(3):map(function(i)
				local s = 0
				for j=1,3 do
					for k=1,3 do
						local D_ijk = sym3x3(D[i],j,k)
						local D_kji = sym3x3(D[k],j,i)
						local gammaUjk = sym3x3(gammaU,j,k)
						local dg = (D_ijk - D_kji) * gammaUjk
						s = s + dg
					end
				end
				return s
			end)
			local K = {}
			for i=1,6 do
				K[i] = calc.K[i](x,y,z)
			end

			local density = 0
			if type(densityFunc) == 'number' then
				density = densityFunc
			elseif type(densityFunc) == 'function' then
				density = densityFunc(x,y,z)
			elseif densityFunc ~= nil then
				error("don't know how to handle density")
			end

			local pressure = 0
			if type(pressureFunc) == 'number' then
				pressure = pressureFunc
			elseif type(pressureFunc) == 'function' then
				pressure = pressureFunc(x,y,z)
			elseif pressureFunc ~= nil then
				error("don't know how to handle pressure")
			end

			local velocityX, velocityY, velocityZ
			if args.velocity then velocityX, velocityY, velocityZ = args.velocity(x,y,z) end
			velocityX = velocityX or 0
			velocityY = velocityY or 0
			velocityZ = velocityZ or 0

			return
				alpha,
				gamma[1], gamma[2], gamma[3], gamma[4], gamma[5], gamma[6],
				A[1], A[2], A[3],
				D[1][1], D[1][2], D[1][3], D[1][4], D[1][5], D[1][6],
				D[2][1], D[2][2], D[2][3], D[2][4], D[2][5], D[2][6],
				D[3][1], D[3][2], D[3][3], D[3][4], D[3][5], D[3][6],
				K[1], K[2], K[3], K[4], K[5], K[6],
				V[1], V[2], V[3],
				density,
				velocityX, velocityY, velocityZ,
				pressure
		end
	end
end
