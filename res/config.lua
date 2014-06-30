require 'util'	--holds helper functions


	-- solver variables

--[[
options:
EulerBurgers	- works for 1D, 2D, 3D
EulerRoe - works for 1D, compiler crashes for 2D and 3D
EulerHLL - works for 1D, 2D, compiler crashes 3D
MHDRoe - left eigenvectors not finished
ADMRoe - compiler crashes
--]]
solverName = 'EulerRoe'

--[[
options:
DonorCell
LaxWendroff
BeamWarming
Fromm		-- not behaving correctly
CHARM
HCUS
HQUICK
Koren
MinMod
Oshker
Ospre
Smart
Sweby
UMIST
VanAlbada1
VanAlbada2
VanLeer		-- not behaving correctly
MonotizedCentral
Superbee
BarthJespersen
--]]
slopeLimiterName = 'Superbee'

useGPU = true
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

-- gravity is specific to the Euler fluid equation solver
useGravity = false 

magneticFieldNoise = 0
gamma = 1.4

-- the number of non-1-sized elements in 'size' determine the dimension
--  (if an element is not provided or nil then it defaults to 1)
--[[ 3D
size = {64, 64, 64}
--]]
-- [[ 2D
size = {256, 256}
--]]
--[[ 1D
size = {2048}
displayScale = .25
--]]




	
	-- Euler equation initial states


--[[ 1D advect wave
function initState(x,y,z)
	local rSq = x * x + y * y + z * z
	return buildStateEuler{
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
	return buildStateEuler{
		density = inside and 1 or .1,
		pressure = inside and 1 or .5,	--1 : .1 works for 2d but not 3d
	}
end
--]]	

--[[ Sod test
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	local inside = x <= 0 and y <= 0 and z <= 0
	return buildStateEuler{
		density = inside and 1 or .1,
		specificEnergyInternal = 1,
	}
end
--]]

-- 2D tests described in Alexander Kurganov, Eitan Tadmor, Solution of Two-Dimensional Riemann Problems for Gas Dynamics without Riemann Problem Solvers
--  which says it is compared with  C. W. Schulz-Rinne, J. P. Collins, and H. M. Glaz, Numerical solution of the Riemann problem for two-dimensional gas dynamics
-- and I can't find that paper right now
--[[ configuration 1
cfl = .475
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	return buildStateEulerQuadrant(x,y,z,{
		q1 = {density=1, pressure=1, velocityX=0, velocityY=0},
		q2 = {density=.5197, pressure=.4, velocityX=-.7259, velocityY=0},
		q3 = {density=.1072, pressure=.0439, velocityX=-.7259, velocityY=-1.4045},
		q4 = {density=.2579, pressure=.15, velocityX=0, velocityY=-1.4045},
	})
end
--]]
--[[ configuration 2
cfl = .475
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	return buildStateEulerQuadrant(x,y,z,{
		q1 = {density=1, pressure=1, velocityX=0, velocityY=0},
		q2 = {density=.5197, pressure=.4, velocityX=-.7259, velocityY=0},
		q3 = {density=1, pressure=1, velocityX=-.7259, velocityY=-.7259},
		q4 = {density=.5197, pressure=.4, velocityX=0, velocityY=-.7259},
	})
end
--]]
--[[ configuration 3
-- using Superbee flux limiter, working with HLL (2nd order not implemented), breaking with Burgers and Roe
cfl = .475
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	return buildStateEulerQuadrant(x,y,z,{
		q1 = {density=1.5, pressure=1.5, velocityX=0, velocityY=0},
		q2 = {density=.5323, pressure=.3, velocityX=1.206, velocityY=0},
		q3 = {density=.138, pressure=.029, velocityX=1.206, velocityY=1.206},
		q4 = {density=.5323, pressure=.3, velocityX=0, velocityY=1.206},
	})
end
--]]

--[[ Sedov shock wave
-- looks good for HLL, Roe not so much (wave moves faster along axii)
local xmid = {
	(xmax[1] + xmin[1]) * .5,
	(xmax[2] + xmin[2]) * .5,
	(xmax[3] + xmin[3]) * .5,
}
local dx = {
	(xmax[1] - xmin[1]) / size[1],
	(xmax[2] - xmin[2]) / (size[2] or 1),
	(xmax[3] - xmin[3]) / (size[3] or 1),
}
function initState(x,y,z)
	local x = x - xmid[1]
	local y = y - xmid[2]
	local z = z - xmid[3]
	local state = {buildStateEuler{
		density = 1,
		pressure = 1e-5,
	}}
	if math.abs(x) < dx[1]
	and math.abs(y) < dx[2]
	and math.abs(z) < dx[3]
	then
		state[5] = 1e+5
	end
	return unpack(state)
end
--]]

--[[ Brio Wu
-- http://www.astro.uni-bonn.de/~jmackey/jmac/node7.html
-- still in the works
gamma = 2
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	local lhs = x < 0
	return buildStateEuler{
		density = lhs and 1 or .125,
		pressure = lhs and 1 or .1,
		magneticFieldX = .75,
		magneticFieldY = lhs and 1 or -1,
	}
end
--]]

--[[ Colella-Woodward interacting blast wave problem
boundaryMethods = {'mirror', 'mirror', 'mirror'}
function initState(x,y,z)
	local pressure
	if x < -.4 then
		pressure = 1000
	elseif x < .4 then
		pressure = .01
	else
		pressure = 100
	end
	return buildStateEuler{
		density = 1,
		velocityX = 0, velocityY = 0, velocityZ = 0,
		pressure = pressure,
	}
end
--]]

-- [[ Kelvin-Hemholtz
--solverName = 'Roe'	--EulerBurgers is having trouble... hmm...
function initState(x,y,z)
	local dim = #size
	local inside = y > -.25 and y < .25
	local theta = (x - xmin[1]) / (xmax[1] - xmin[1]) * 2 * math.pi
	if dim >= 3 then 
		theta = theta * (z - xmin[3]) / (xmax[3] - xmin[3]) 
	end
	local noise = size[1] * 2e-5
	return buildStateEuler{
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
function initState(x,y,z)
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
	local dx = x - minSource[1]
	local dy = y - minSource[2]
	local dz = z - minSource[3]
	local noise = math.exp(-100 * (dx * dx + dy * dy + dz * dz))
	return buildStateEuler{
		density = inside and 1 or .1,
		pressure = 1,
		velocityX = .01 * noise * crand(),
		velocityY = .01 * noise * crand(),
		velocityZ = .01 * noise * crand(),
	}
end
--]]


		-- 1D ADM equation initial state

--[[
xmin = {-30, -30, -30}
xmax = {30, 30, 30}
local xmid = (xmax[1] + xmin[1]) * .5
adm_BonaMasso_f = 1
function initState(x,y,z)
	x = (x - xmid) / ((xmax[1] - xmid) / 3)
	local h = math.exp(-x*x); 
	local dh_dx = -2 * x * h;
	local d2h_dx2 = 2 * h * (2 * x * x - 1);
	local g = 1 - dh_dx * dh_dx;
	local D_g = -2 * dh_dx * d2h_dx2 / g;
	local KTilde = -d2h_dx2 / g;
	local f = adm_BonaMasso_f;
	local D_alpha = math.sqrt(f) * KTilde;
	return D_alpha, D_g, KTilde, 0, 0, 0, 0, 0	
end
--]]

