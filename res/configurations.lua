return {
	
	
	-- Euler equation initial states

	--debugging
	['Constant'] = function()
		initState = function(x,y,z)
			return buildStateEuler{
				density = 1,
				pressure = 1,
				velocityX = 1,
				velocityY = 1,
				velocityZ = 1,
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
	--   only when using the arbitrary-normal method.  when rotating into the x-axis it works fine
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
			local lhs = x <= 0 and y <= 0 and z <= 0
			return buildStateEuler{
				density = lhs and 1 or .125,
				pressure = lhs and 1 or .1,
				magneticFieldX = .75,
				magneticFieldY = lhs and 1 or -1,
				magneticFieldZ = 0,
			}
		end
	end,

	-- http://www.astro.virginia.edu/VITA/ATHENA/ot.html
	-- http://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
	['Orszag-Tang'] = function()
		gamma = 5/3
		local B0 = 1/math.sqrt(4 * math.pi)
		-- assumes coordinate space to be [-.5,.5]^2
		initState = function(x,y,z)
			return buildStateEuler{
				density = 25/(36*math.pi),
				velocityX = -math.sin(2*math.pi*(y+.5)),
				velocityY = math.sin(2*math.pi*(x+.5)),
				velocityZ = 0,
				pressure = 5/(12*math.pi),	-- is this hydro pressure or total pressure?
				magneticFieldX = -B0 * math.sin(2 * math.pi * (y+.5)),
				magneticFieldY = B0 * math.sin(4 * math.pi * (x+.5)),
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
					{center={0, 0, 0}, radius = .2},
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
		xmin = {0, 0, 0}
		xmax = {300, 300, 300}
		local xmid = (xmax[1] + xmin[1]) * .5
		local sigma = 10
		adm_BonaMasso_f = '1.f + 1.f / (alpha * alpha)'
		initState = function(x,y,z)
			local h = 5 * math.exp(-((x - xmid) / sigma)^2)
			local dx_h = -2 * (x - xmid) / sigma^2 * h
			local d2x_h = (-2 / sigma^2 + 4 * (x - xmid)^2 / sigma^4) * h
			local alpha = 1
			local g = 1 - dx_h^2
			local A = 0	-- dx ln alpha
			local D = -dx_h * d2x_h
			local K = -d2x_h / math.sqrt(g)
			return alpha, g, A, D, K
		end
	end,

	['ADM-3D'] = function()
		xmin = {0, 0, 0}
		xmax = {300, 300, 300}
		local xmid = (xmax[1] + xmin[1]) * .5
		local ymid = (xmax[2] + xmin[2]) * .5
		local zmid = (xmax[3] + xmin[3]) * .5
		local sigma = 10
		local sigma2 = sigma * sigma
		local sigma4 = sigma2 * sigma2
		adm_BonaMasso_f = '1.f + 1.f / (alpha * alpha)'	-- TODO ... provide code in some more reliable way 
		initState = function(x,y,z)
			local dx = x - xmid
			local dx2 = dx * dx
			--[[ 1D
			local ds2 = dx2
			local h = 5 * math.exp(-ds2 / sigma2)
			local hx = -2 * dx / sigma2 * h
			local hy = 0
			local hz = 0
			local hxx = (-2 / sigma2 + 4 * dx2 / sigma4) * h
			local hxy = 0
			local hxz = 0
			local hyy = 0
			local hyz = 0
			local hzz = 0
			--]]		
			-- [[ 2D
			local dy = y - ymid
			local dy2 = dy * dy
			local ds2 = dx2 + dy2
			local h = 5 * math.exp(-ds2 / sigma2)
			local hx = -2 * dx / sigma2 * h
			local hy = -2 * dy / sigma2 * h
			local hz = 0
			local hxx = (-2 / sigma2 + 4 * dx2 / sigma4) * h
			local hxy = 4 * dx * dy / sigma4 * h
			local hxz = 0
			local hyy = (-2 / sigma2 + 4 * dy2 / sigma4) * h
			local hyz = 0
			local hzz = 0
			--]]
			--[[ 3D
			local dy = 0--y - ymid
			local dz = 0--z - zmid
			local dx2 = dx * dx
			local dy2 = dy * dy
			local dz2 = dz * dz
			local ds2 = dx2 + dy2 + dz2
			local h = 5 * math.exp(-ds2 / sigma2)
			local hx = -2 * dx / sigma2 * h
			local hy = -2 * dy / sigma2 * h
			local hz = -2 * dz / sigma2 * h
			local hxx = (-2 / sigma2 + 4 * dx2 / sigma4) * h
			local hxy = 4 * dx * dy / sigma4 * h
			local hxz = 4 * dx * dz / sigma4 * h
			local hyy = (-2 / sigma2 + 4 * dy2 / sigma4) * h
			local hyz = 4 * dy * dz / sigma4 * h
			local hzz = (-2 / sigma2 + 4 * dz2 / sigma4) * h
			--]]
			local g_xx = 1 - hx * hx
			local g_xy = -hx * hy
			local g_xz = -hx * hz
			local g_yy = 1 - hy * hy
			local g_yz = -hy * hz
			local g_zz = 1 - hz * hz
			local det_g = g_xx * g_yy * g_zz + g_xy * g_yz * g_xz + g_xz * g_xy * g_yz - g_xz * g_yy * g_xz - g_yz * g_yz * g_xx - g_zz * g_xy * g_xy
			local gUxx = (g_yy * g_zz - g_yz * g_yz) / det_g
			local gUxy = (g_xz * g_yz - g_xy * g_zz) / det_g
			local gUxz = (g_xy * g_yz - g_xz * g_yy) / det_g
			local gUyy = (g_xx * g_zz - g_xz * g_xz) / det_g
			local gUyz = (g_xz * g_xy - g_xx * g_yz) / det_g
			local gUzz = (g_xx * g_yy - g_xy * g_xy) / det_g
			local D_xxx = -hx * hxx
			local D_xxy = -.5 * (hxx * hy + hx * hxy)
			local D_xxz = -.5 * (hxx * hz + hx * hxz)
			local D_xyy = -hy * hxy
			local D_xyz = -.5 * (hxy * hz + hy * hxz)
			local D_xzz = -hz * hxz
			local D_yxx = -hx * hxy
			local D_yxy = -.5 * (hxy * hy + hx * hyy)
			local D_yxz = -.5 * (hxy * hz + hx * hyz)
			local D_yyy = -hy * hyy
			local D_yyz = -.5 * (hyy * hz + hy * hyz)
			local D_yzz = -hz * hyz
			local D_zxx = -hx * hxz
			local D_zxy = -.5 * (hxz * hy + hx * hyz)
			local D_zxz = -.5 * (hxz * hz + hx * hzz)
			local D_zyy = -hy * hyz
			local D_zyz = -.5 * (hyz * hz + hy * hzz)
			local D_zzz = -hz * hzz
			local sqrt_det_g = math.sqrt(det_g)	
			local K_xx = -hxx / sqrt_det_g
			local K_xy = -hxy / sqrt_det_g
			local K_xz = -hxz / sqrt_det_g
			local K_yy = -hyy / sqrt_det_g
			local K_yz = -hyz / sqrt_det_g
			local K_zz = -hzz / sqrt_det_g
			local V_x = (D_xxy - D_yxx) * gUxy + (D_xxz - D_zxx) * gUxz + (D_xyy - D_yxy) * gUyy + (D_xyz - D_yxz) * gUyz + (D_xyz - D_zxy) * gUyz + (D_xzz - D_zxz) * gUzz
			local V_y = (D_yxx - D_xxy) * gUxx + (D_yxy - D_xyy) * gUxy + (D_yxz - D_xyz) * gUxz + (D_yxz - D_zxy) * gUxz + (D_yyz - D_zyy) * gUyz + (D_yzz - D_zyz) * gUzz
			local V_z = (D_zxx - D_xxz) * gUxx + (D_zxy - D_xyz) * gUxy + (D_zxy - D_yxz) * gUxy + (D_zxz - D_xzz) * gUxz + (D_zyy - D_yyz) * gUyy + (D_zyz - D_yzz) * gUyz
			local alpha = 1
			local A_x = 0	-- dx ln alpha
			local A_y = 0	-- dy ln alpha
			local A_z = 0	-- dz ln alpha
			return 
				alpha,
				g_xx, g_xy, g_xz, g_yy, g_yz, g_zz,
				A_x, A_y, A_z,
				D_xxx, D_xxy, D_xxz, D_xyy, D_xyz, D_xzz,
				D_yxx, D_yxy, D_yxz, D_yyy, D_yyz, D_yzz,
				D_zxx, D_zxy, D_zxz, D_zyy, D_zyz, D_zzz,
				K_xx, K_xy, K_xz, K_yy, K_yz, K_zz,
				V_x, V_y, V_z
		end
	end,


		-- Maxwell equations initial state


	['Maxwell-1'] = function()
		initState = function(x,y,z)
			local inside = x <= 0 and y <= 0 and z <= 0
			local ex = 0 
			local ey = 0
			local ez = 1
			local bx = 0
			local by = 0
			local bz = inside and 1 or -1
			return ex * permittivity, ey * permittivity, ez * permittivity, bx, by, bz
		end
	end,
}


