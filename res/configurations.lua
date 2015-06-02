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
		--boundaryMethods = {{min='MIRROR', max='MIRROR'}, {min='MIRROR', max='MIRROR'}, {min='MIRROR', max='MIRROR'}}
		initState = function(x,y,z)
			local inside = x <= 0 and y <= 0 and z <= 0	
			return buildStateEuler{
				x=x, y=y, z=z,
				density = inside and 1 or .1,
				specificEnergyInternal = 1,
			}
		end
	end,


	-- 2D tests described in Alexander Kurganov, Eitan Tadmor, Solution of Two-Dimensional Riemann Problems for Gas Dynamics without Riemann Problem Solvers
	--  which says it is compared with  C. W. Schulz-Rinne, J. P. Collins, and H. M. Glaz, Numerical solution of the Riemann problem for two-dimensional gas dynamics
	-- and I can't find that paper right now

	['Configuration 1'] = function()
		cfl = .475
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
			if math.abs(x) < 1.5 * dx[1]
			and math.abs(y) < 1.5 * dx[2]
			and math.abs(z) < 1.5 * dx[3]
			then
				return buildStateEuler{
					density = 1,
					pressure = 1e+5,
				}
			else
				return buildStateEuler{
					density = 1,
					pressure = 1e-5,
				}
			end
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='PERIODIC', max='PERIODIC'}, {min='PERIODIC', max='PERIODIC'}, {min='PERIODIC', max='PERIODIC'}}
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
		boundaryMethods[1] = {min='FREEFLOW', max='FREEFLOW'}
		boundaryMethods[2] = {min='MIRROR', max='MIRROR'}
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

	-- http://www.cfd-online.com/Wiki/2-D_laminar/turbulent_driven_square_cavity_flow
	['Square Cavity'] = function()
		boundaryMethods = {
			{min='MIRROR', max='MIRROR'},
			{min='MIRROR', max='NONE'},
			{min='MIRROR', max='MIRROR'},
		}
		initState = function(x,y,z)
			return buildStateEuler{
				x=x, y=y, z=z,
				density = 1,
				velocityX = y > .45 and 1 or 0,
				pressure = 1,
			}
		end
	end,

	['Shock Bubble Interaction'] = function()
		boundaryMethods = {
			{min='PERIODIC', max='FREEFLOW'},
			{min='FREEFLOW', max='FREEFLOW'},
			{min='FREEFLOW', max='FREEFLOW'},
		}
		local bubbleX = 0
		local bubbleY = 0 
		local bubbleZ = 0 
		local bubbleRadius = .2
		local pressureWaveX = -.225
		initState = function(x,y,z)
			local bubbleRSq = (x-bubbleX)^2 + (y-bubbleY)^2 + (z-bubbleZ)^2
			return buildStateEuler{
				x=x, y=y, z=z,
				density = bubbleRSq < bubbleRadius*bubbleRadius and .1 or 1,
				pressure = (x < pressureWaveX) and 1.9 or .1,
			}
		end
	end,

	['Flow Around Cylinder'] = function()
		boundaryMethods = {
			{min='NONE', max='FREEFLOW'},
			{min='FREEFLOW', max='FREEFLOW'},
			{min='FREEFLOW', max='FREEFLOW'},
		}
		local cylinderRadius = .25
		local cylinderRadiusSq = cylinderRadius * cylinderRadius 
		initState = function(x,y,z)
			local r2 = x*x + y*y + z*z
			return buildStateEuler{
				x=x, y=y, z=z,
				density = 1,
				velocityX = .2,
				pressure = 1,
				solid = r2 < cylinderRadiusSq and 1 or 0
			}
		end
	end,

	-- gravity potential test - equilibrium - Rayleigh-Taylor

	['self-gravitation test 1'] = function()
		useGravity = true
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		boundaryMethods = {{min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}, {min='FREEFLOW', max='FREEFLOW'}}
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
		print('deriving and compiling...')
		local function det3x3sym(msym)
			local xx, xy, xz, yy, yz, zz = unpack(msym)
			return xx * yy * zz + xy * yz * xz + xz * xy * yz - xz * yy * xz - yz * yz * xx - zz * xy * xy
		end
		local function inv3x3sym(msym, det)
			local xx, xy, xz, yy, yz, zz = unpack(msym)
			if not det then det = det3x3sym(msym) end
			return  (yy * zz - yz * yz) / det, (xz * yz - xy * zz) / det, (xy * yz - xz * yy) / det, (xx * zz - xz * xz) / det, (xz * xy - xx * yz) / det, (xx * yy - xy * xy) / det
		end	
		local function sym3x3unpack(m) 
			return {m[1][1], m[1][2], m[1][3], m[2][2], m[2][3], m[3][3]} 
		end
		local function sym3x3pack(xx,xy,xz,yy,yz,zz) 
			return {{xx,xy,xz},{xy,yy,yz},{xz,yz,zz}} 
		end
		local symmath = require 'symmath'	-- this is failing ...
		xmin = {0, 0, 0}
		xmax = {300, 300, 300}
		local xc = (xmax[1] + xmin[1]) * .5
		local yc = (xmax[2] + xmin[2]) * .5
		local zc = (xmax[3] + xmin[3]) * .5
		local sigma = 10
		local x = symmath.var'x'
		local y = symmath.var'y'
		local z = symmath.var'z'
		local xs = table{x,y,z}
		local function delta(i,j) return i == j and 1 or 0 end
		local h = 5 * symmath.exp(-((x - xc)^2 + (y - yc)^2) / sigma^2)
		local dh = xs:map(function(xi) return h:diff(xi):simplify() end)
		local d2h = dh:map(function(dhi) return xs:map(function(xj) return dhi:diff(xj):simplify() end) end)
		local g = xs:map(function(xi,i) return xs:map(function(xj,j) return (delta(xi,xj) - dh[i] * dh[j]):simplify() end) end)
		local det_g = det3x3sym(sym3x3unpack(g)):simplify()
		local gU = sym3x3pack(inv3x3sym(sym3x3unpack(g), det_g))
		local K = xs:map(function(xi,i) return xs:map(function(xj,j) return (-d2h[i][j] / det_g^.5):simplify() end) end)
		local D = xs:map(function(xk,k) return xs:map(function(xi,i) return xs:map(function(xj,j) return (g[i][j]:diff(xk)/2):simplify() end) end) end)
		local alpha = symmath.Constant(1)
		local A = xs:map(function(xi) return (alpha:diff(xi) / alpha):simplify() end)
		local V = xs:map(function(xi,i)
			local s = 0
			for j=1,3 do
				for k=1,3 do
					s = s + (D[i][j][k] - D[k][j][i]) * gU[j][k]
				end
			end
			return s:simplify()
		end)
		local exprs = table{
			alpha = alpha,
			A = A,
			g = sym3x3unpack(g),
			gU = sym3x3unpack(gU),
			D = xs:map(function(xi,i) return sym3x3unpack(D[i]) end),
			K = sym3x3unpack(K),
			V = V,
		}
		local function buildCalc(expr, name)
			assert(type(expr) == 'table')
			if expr.isa and expr:isa(symmath.Expression) then 
				return expr:simplify():compile(xs), name
			end
			return table.map(expr, buildCalc), name
		end
		local calc = exprs:map(buildCalc)
		adm_BonaMasso_f = '1.f + 1.f / (alpha * alpha)'	-- TODO OpenCL exporter with lua symmath
		print('...done deriving and compiling.')
		initState = function(x,y,z)
			local alpha = calc.alpha(x,y,z)
			local A_x = calc.A:map(function(A_k) return A_k(x,y,z) end):unpack()
			local g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = calc.g:map(function(g_ij) return g_ij(x,y,z) end):unpack()
			local gUxx, gUxy, gUxz, gUyy, gUyz, gUzz = calc.gU:map(function(gUij) return gUij(x,y,z) end):unpack()
			local D_xxx, D_xxy, D_xxz, D_xyy, D_xyz, D_xzz = calc.D[1]:map(function(D_xjk) return D_xjk(x,y,z) end):unpack()
			local D_yxx, D_yxy, D_yxz, D_yyy, D_yyz, D_yzz = calc.D[2]:map(function(D_yjk) return D_yjk(x,y,z) end):unpack()
			local D_zxx, D_zxy, D_zxz, D_zyy, D_zyz, D_zzz = calc.D[3]:map(function(D_zjk) return D_zjk(x,y,z) end):unpack()
			local K_xx, K_xy, K_xz, K_yy, K_yz, K_zz = calc.K:map(function(K_ij) return K_ij(x,y,z) end):unpack()
			local V_x, V_y, V_z = calc.V:map(function(V_k) return V_k(x,y,z) end):unpack()
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

	['Alcubierre'] = function()
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


