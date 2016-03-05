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
	return pressure / ((gamma - 1) * density)
end

local function getMagneticFieldEnergy(magneticFieldX, magneticFieldY, magneticFieldZ)
	return .5 * (magneticFieldX * magneticFieldX + magneticFieldY * magneticFieldY + magneticFieldZ * magneticFieldZ) / vaccuumPermeability
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
		magneticFieldX,
		magneticFieldY,
		magneticFieldZ, 
		-- energy
		energyTotal,
		potentialEnergy,
		-- solid or not
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
	g = 3-metric
	K = extrinsic curvature

	g & K are stored as {xx,xy,xz,yy,yz,zz}
--]]
function initNumRel(args)	
	local vars = assert(args.vars)
	
	local exprs = table{
		alpha = assert(args.alpha),
		g = {table.unpack(args.g)},
		K = {table.unpack(args.K)}
	}
	assert(#exprs.g == 6)
	assert(#exprs.K == 6)

	local function toExpr(expr, name)
		if type(expr) == 'number' then expr = symmath.Constant(expr) end
		if type(expr) == 'table' then
			if not expr.isa then
				expr = table.map(expr, toExpr)
			end
		end
		return expr, name
	end
	exprs = table.map(exprs, toExpr)
	
	local function buildCalc(expr, name)
		assert(type(expr) == 'table')
		if symmath.Expression.is(expr) then 
			return expr:simplify():compile(vars), name
		end
		return table.map(expr, buildCalc), name
	end
	
	-- ADM
	if solverName == 'ADM1DRoe' then
		exprs.g = exprs.g[1]	-- only need g_xx
		exprs.A = (exprs.alpha:diff(vars[1]) / exprs.alpha):simplify()	-- only need a_x
		exprs.D = (exprs.g:diff(vars[1])/2):simplify()	-- only need D_xxx
		exprs.K = exprs.K[1]	-- only need K_xx
		local calc = table.map(exprs, buildCalc)
		initState = function(x,y,z)
			local alpha = calc.alpha(x,y,z)
			local g = calc.g(x,y,z)
			local A = calc.A(x,y,z)
			local D = calc.D(x,y,z)
			local K = calc.K(x,y,z)
			return alpha, g, A, D, K
		end
	elseif solverName == 'ADM3DRoe' then
		-- for complex computations it might be handy to extract the determinant first ...
		local gUxx, gUxy, gUxz, gUyy, gUyz, gUzz = mat33.inv(exprs.g:unpack())
		exprs.gU = table{gUxx, gUxy, gUxz, gUyy, gUyz, gUzz}

		exprs.D = table.map(vars, function(x_k)
			return table.map(exprs.g, function(g_ij)
				return (g_ij:diff(x_k)/2):simplify()
			end)
		end)
		
		exprs.A = table.map(vars, function(var)
			return (exprs.alpha:diff(var) / exprs.alpha):simplify()
		end)
		
		local calc = table.map(exprs, buildCalc)
		
		initState = function(x,y,z)
		
			local alpha = calc.alpha(x,y,z)
			local A = calc.A:map(function(A_i) return A_i(x,y,z) end)
			local g = calc.g:map(function(g_ij) return g_ij(x,y,z) end)
			local D = calc.D:map(function(D_i) return D_i:map(function(D_ijk) return D_ijk(x,y,z) end) end)
			local gU = calc.gU:map(function(gUij) return gUij(x,y,z) end)
			
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
						local gUjk = sym3x3(gU,j,k)
						local dg = (D_ijk - D_kji) * gUjk
						s = s + dg
					end
				end
				return s
			end)
			local K = {}
			for i=1,6 do
				K[i] = calc.K[i](x,y,z)
			end

			return
				alpha,
				g[1], g[2], g[3], g[4], g[5], g[6],
				A[1], A[2], A[3],
				D[1][1], D[1][2], D[1][3], D[1][4], D[1][5], D[1][6],
				D[2][1], D[2][2], D[2][3], D[2][4], D[2][5], D[2][6],
				D[3][1], D[3][2], D[3][3], D[3][4], D[3][5], D[3][6],
				K[1], K[2], K[3], K[4], K[5], K[6],
				V[1], V[2], V[3]
		end
	end
end

