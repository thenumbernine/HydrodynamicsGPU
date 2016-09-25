local mat33
mat33 = {
	det = function(xx, xy, xz, yy, yz, zz)
		return xx * yy * zz
			+ xy * yz * xz
			+ xz * xy * yz
			- xz * yy * xz
			- yz * yz * xx
			- zz * xy * xy
	end,
	inv = function(xx, xy, xz, yy, yz, zz, d)
		if not d then d = mat33.det(xx, xy, xz, yy, yz, zz) end
		return 
			(yy * zz - yz * yz) / d,	-- xx
			(xz * yz - xy * zz) / d,	-- xy
			(xy * yz - xz * yy) / d,	-- xz
			(xx * zz - xz * xz) / d,	-- yy
			(xz * xy - xx * yz) / d,	-- yz
			(xx * yy - xy * xy) / d		-- zz
	end,
}
return mat33
