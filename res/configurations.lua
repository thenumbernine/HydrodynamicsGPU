return {
	
	
	-- Euler equation initial states

	--debugging
	['Constant'] = function()
		initState = function(x,y,z)
			return buildStateEuler{
				density = 1,
				pressure = 1,
				magneticFieldX = 1,
				magneticFieldY = 1,
				magneticFieldZ = 1,
			}
		end
	end,
	
	['1D Advect Wave'] = function()
		initState = function(x,y,z)
			local rSq = x * x + y * y + z * z
			return buildStateEuler{
				velocityX = 1,
				density = math.exp(-100*rSq) + 1,
				pressure = 1,
			}
		end
	end,

	-- http://www.cfd-online.com/Wiki/Explosion_test_in_2-D
	['Sphere'] = function()
		initState = function(x,y,z)
			local rSq = x * x + y * y + z * z
			local inside = rSq <= .2*.2
			return buildStateEuler{
				density = inside and 1 or .1,
				pressure = inside and 1 or .1,	--1 : .1 works for 2d but not 3d
			}
		end
	end,

	['Sod'] = function()
		--boundaryMethods = {'MIRROR', 'MIRROR', 'MIRROR'}
		initState = function(x,y,z)
			local inside = x <= 0 and y <= 0 and z <= 0
			return buildStateEuler{
				density = inside and 1 or .1,
				specificEnergyInternal = 1,
--debugging
--magneticFieldX = 1,
--magneticFieldY = 1,
--magneticFieldZ = 1,
			}
		end
	end,


	-- 2D tests described in Alexander Kurganov, Eitan Tadmor, Solution of Two-Dimensional Riemann Problems for Gas Dynamics without Riemann Problem Solvers
	--  which says it is compared with  C. W. Schulz-Rinne, J. P. Collins, and H. M. Glaz, Numerical solution of the Riemann problem for two-dimensional gas dynamics
	-- and I can't find that paper right now

	['Configuration 1'] = function()
		cfl = .475
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildStateEulerQuadrant(x,y,z,{
				q1 = {density=1, pressure=1, velocityX=0, velocityY=0},
				q2 = {density=.5197, pressure=.4, velocityX=-.7259, velocityY=0},
				q3 = {density=.1072, pressure=.0439, velocityX=-.7259, velocityY=-1.4045},
				q4 = {density=.2579, pressure=.15, velocityX=0, velocityY=-1.4045},
			})
		end
	end,
	
	['Configuration 2'] = function()
		cfl = .475
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildStateEulerQuadrant(x,y,z,{
				q1 = {density=1, pressure=1, velocityX=0, velocityY=0},
				q2 = {density=.5197, pressure=.4, velocityX=-.7259, velocityY=0},
				q3 = {density=1, pressure=1, velocityX=-.7259, velocityY=-.7259},
				q4 = {density=.5197, pressure=.4, velocityX=0, velocityY=-.7259},
			})
		end
	end,

	-- HLL looks good
	-- Roe gets noise along -x axis, shows antisymmetry between axii, then blows up near the noise
	['Configuration 3'] = function()
		cfl = .475
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildStateEulerQuadrant(x,y,z,{
				q1 = {density=1.5, pressure=1.5, velocityX=0, velocityY=0},
				q2 = {density=.5323, pressure=.3, velocityX=1.206, velocityY=0},
				q3 = {density=.138, pressure=.029, velocityX=1.206, velocityY=1.206},
				q4 = {density=.5323, pressure=.3, velocityX=0, velocityY=1.206},
			})
		end
	end,

	['Configuration 4'] = function()
		cfl = .475
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildStateEulerQuadrant(x,y,z,{
				q1 = {density=1.1, pressure=1.1, velocityX=0, velocityY=0},
				q2 = {density=.5065, pressure=.35, velocityX=.8939, velocityY=0},
				q3 = {density=1.1, pressure=1.1, velocityX=.8939, velocityY=.8939},
				q4 = {density=.5065, pressure=.35, velocityX=0, velocityY=.8939},
			})
		end
	end,

	['Configuration 5'] = function()
		cfl = .475
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildStateEulerQuadrant(x,y,z,{
				q1 = {density=1, pressure=1, velocityX=-.75, velocityY=-.5},
				q2 = {density=2, pressure=1, velocityX=-.75, velocityY=.5},
				q3 = {density=1, pressure=1, velocityX=.75, velocityY=.5},
				q4 = {density=3, pressure=1, velocityX=.75, velocityY=-.5},
			})
		end
	end,

	['Configuration 6'] = function()
		cfl = .475
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildStateEulerQuadrant(x,y,z,{
				q1 = {density=1, pressure=1, velocityX=.75, velocityY=-.5},
				q2 = {density=2, pressure=1, velocityX=.75, velocityY=.5},
				q3 = {density=1, pressure=1, velocityX=-.75, velocityY=.5},
				q4 = {density=3, pressure=1, velocityX=-.75, velocityY=-.5},
			})
		end
	end,

	-- looks good for HLL
	-- Roe not so much: wave moves faster when aligned with axii
	['Sedov'] = function()
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
		initState = function(x,y,z)
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
	end,

	-- http://www.astro.uni-bonn.de/~jmackey/jmac/node7.html
	-- http://www.astro.princeton.edu/~jstone/Athena/tests/brio-wu/Brio-Wu.html
	['Brio-Wu'] = function()
		gamma = 2
		initState = function(x,y,z)
			local lhs = x < 0
			return buildStateEuler{
				density = lhs and 1 or .125,
				pressure = lhs and 1 or .1,
				magneticFieldX = .75,
				magneticFieldY = lhs and 1 or -1,
				magneticFieldZ = 0,
			}
		end
	end,

	-- Colella-Woodward interacting blast wave problem
	['Colella-Woodward'] = function()
		boundaryMethods = {'MIRROR', 'MIRROR', 'MIRROR'}
		initState = function(x,y,z)
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
	end,

	--EulerBurgers is having trouble
	--EulerHLL works fine
	--EulerRoe at high resolutions after a long time shows some waves and then blows up
	['Kelvin-Hemholtz'] = function()
		boundaryMethods = {'PERIODIC', 'PERIODIC', 'PERIODIC'}
		initState = function(x,y,z)
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
	end,

	-- http://www.astro.virginia.edu/VITA/ATHENA/dmr.html
	['Double Mach Reflection'] = function()
		xmin[1] = 0
		xmax[1] = 4
		xmin[2] = 0
		xmax[2] = 1
		boundaryMethods[1] = 'FREEFLOW'
		boundaryMethods[2] = 'MIRROR'
		local x0 = 1/6
		initState = function(x,y,z)
			local lhs = x < x0 + y * (1/3)^(1/2)
			return buildStateEuler{
				density = lhs and 8 or 1.4,
				velocityX = lhs and 8.25 * math.cos(math.rad(30)) or 0,
				velocityY = lhs and -8.25 * math.cos(math.rad(30)) or 0,
				pressure = lhs and 116.5 or 1,
			}
		end
	end,

	-- gravity potential test - equilibrium - Rayleigh-Taylor

	['self-gravitation test 1'] = function()
		useGravity = true
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildSelfGravitationState(x,y,z,{
				sources={
					{center={0, 0, 0}, radius = .1},
				},
			})
		end
	end,

	['self-gravitation test 1 spinning'] = function()
		useGravity = true
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildSelfGravitationState(x,y,z,{
				sources={
					{
						center={0, 0, 0}, 
						radius = .2,
						inside = function(dx,dy,dz)
							return buildStateEuler{
								velocityX = -10 * dy,
								velocityY = 10 * dx,
								pressure = 1,
								density = 1,
							}
						end},
				},
			})
		end
	end,

	['self-gravitation test 2'] = function()
		useGravity = true
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			local rho,mx,my,mz,eTotal,bx,by,bz = buildSelfGravitationState(x,y,z,{
				sources={
					{
						center = {-.25, 0, 0},
						radius = .1,
						inside = function(dx,dy,dz)
							return buildStateEuler{
								pressure = 1,
								density = 1,
							}
						end,
					},
					{
						center = {.25, 0, 0},
						radius = .1,
						inside = function(dx,dy,dz)
							return buildStateEuler{
								pressure = 1,
								density = 1,
							}
						end,
					},
				},
			})
			mx = -5 * rho * y
			my = 5 * rho * x
			return rho,mx,my,mz,eTotal,bx,by,bz
		end
	end,

	['self-gravitation test 4'] = function()
		useGravity = true
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
			return buildSelfGravitationState(x,y,z,{
				sources={
					{center={.25, .25, 0}, radius = .1},
					{center={-.25, .25, 0}, radius = .1},
					{center={.25, -.25, 0}, radius = .1},
					{center={-.25, -.25, 0}, radius = .1},
				},
			})
		end
	end,


		-- 1D ADM equation initial state


	['ADM-1D'] = function()
		xmin = {-30, -30, -30}
		xmax = {30, 30, 30}
		local xmid = (xmax[1] + xmin[1]) * .5
		adm_BonaMasso_f = 1
		boundaryMethods = {'FREEFLOW', 'FREEFLOW', 'FREEFLOW'}
		initState = function(x,y,z)
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
	end,
}


