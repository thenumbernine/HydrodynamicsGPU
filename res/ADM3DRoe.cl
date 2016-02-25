/*
The components of the Roe solver specific to the ADM equations
paritcularly the spectral decomposition

This is the 3D version

looks like it will require refining from the Euler Roe
in which the Euler Roe eigenbasis operates on all state variables
whereas the ADM Roe only operates on certain ones ...

...according to "Introduciton to Numerical Relativity"
partial_i operates on A_i, K_jk, D_ijk
therefore instead of eigenvectors across all 37 states, we only need it across 25 states

A_k = partial_k ln alpha = partial_k alpha / alpha <=> A_k alpha = partial_k alpha
D_kij = 1/2 partial_k g_ij <=> 2 D_kij = partial_k g_ij
V_k = D_km^m - D^m_mk

lambda^k_ij = D^k_ij + 1/2 delta^k_i (A_j + 2 V_j - D_jm^m) + 1/2 delta^k_j (A_i + 2 V_i - D_im^m)
S_ij = -R4_ij + tr K K_ij - 2 K_ik K^k_j + 4 D_kmi D^km_j + Gamma^k_km Gamma^m_ij - Gamma_ikm Gamma_j^km
	+ (A^k - 2 D_m^km) (D_ijk + D_jik) + A_i (V_j - 1/2 D_jk^k) + A_j (V_i - 1/2 D_ik^k)


partial_g_mn g^ij 
= derivative of an inverse wrt its matrix
= -g^im g^nj

partial_k g^ij
= partial_g_mn g^ij partial_k g_mn
= -g^im g^nj partial_k g_mn

partial_k tr K
= partial_k (K_ij g^ij)
= g^ij partial_k K_ij + K_ij partial_k g^ij
= g^ij partial_k K_ij + K_ij partial_g_mn g^ij partial_k g_mn
= g^ij partial_k K_ij - K_ij g^im g^nj partial_k g_mn

partial_k lambda^k_ij
= partial_k (D^k_ij + 1/2 delta^k_i (A_j + 2 V_j - D_jm^m) + 1/2 delta^k_j (A_i + 2 V_i - D_im^m)
= partial_k D^k_ij + 1/2 delta^k_i (partial_k A_j + 2 partial_k V_j - partial_k D_jm^m) + 1/2 delta^k_j (partial_k A_i + 2 partial_k V_i - partial_k D_im^m)
= partial_k (g^km D_mij) + 1/2 delta^k_i (partial_k A_j + 2 partial_k V_j - partial_k (D_jmn g^mn)) + 1/2 delta^k_j (partial_k A_i + 2 partial_k V_i - partial_k (D_imn g^mn))
= D_mij partial_k g^km + g^km partial_k D_mij 
	+ 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i)
	+ delta^k_i partial_k V_j + delta^k_j partial_k V_i
= -D_mij g^kr g^sm partial_k g_rs + g^km partial_k D_mij
	+ 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i)
	+ delta^k_i partial_k V_j + delta^k_j partial_k V_i
= -2 D_mij D_k^km + g^km partial_k D_mij
	+ 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i)
	+ delta^k_i partial_k V_j + delta^k_j partial_k V_i

full system:

partial_t alpha = -alpha^2 f tr K
partial_t g_ij = -2 alpha K_ij
partial_t A_k + partial_k (alpha f tr K) = 0
... partial_t A_k + tr K(f + alpha f') partial_k alpha + alpha f partial_k tr K = 0
... partial_t A_k + alpha f g^ij partial_k K_ij = -alpha tr K (f + alpha f') A_k + 2 alpha f K^ij D_kij
partial_t D_kij + partial_k (alpha K_ij) = 0
... partial_t D_kij + K_ij partial_k alpha + alpha partial_k K_ij = 0
... partial_t D_kij + alpha partial_k K_ij = -alpha A_k K_ij
partial_t K_ij + partial_k (alpha lambda^k_ij) = alpha S_ij
... partial_t K_ij + lambda^k_ij partial_k alpha + alpha partial_k lambda^k_ij = alpha S_ij
... partial_t K_ij + alpha partial_k lambda^k_ij = alpha S_ij - alpha lambda^k_ij A_k
... partial_t K_ij + alpha (-2 D_mij D_k^km + g^km partial_k D_mij
	+ 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i)
	+ delta^k_i partial_k V_j + delta^k_j partial_k V_i) 
	= alpha S_ij - alpha lambda^k_ij A_k
... partial_t K_ij + alpha (g^km partial_k D_mij
	+ 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i)
	+ delta^k_i partial_k V_j + delta^k_j partial_k V_i) 
	= alpha S_ij - alpha lambda^k_ij A_k + 2 alpha D_mij D_k^km
partial_t V_k = alpha P_k

final result:

	flux-less terms:
partial_t alpha = -alpha^2 f tr K
partial_t g_ij = -2 alpha K_ij
partial_t V_k = alpha P_k

	flux terms:
partial_t A_k + alpha f g^ij partial_k K_ij = -alpha tr K (f + alpha f') A_k + 2 alpha f K^ij D_kij
partial_t D_kij + alpha partial_k K_ij = -alpha A_k K_ij
partial_t K_ij + alpha (g^km partial_k D_mij + 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i) + delta^k_i partial_k V_j + delta^k_j partial_k V_i) = alpha S_ij - alpha lambda^k_ij A_k + 2 alpha D_mij D_k^km

full full system

d/dx:	A_x,A_y,A_z,D_xxx,D_xxy,D_xxz,D_xyy,D_xyz,D_xzz,D_yxx,D_yxy,D_yxz,D_yyy,D_yyz,D_yzz,D_zxx,D_zxy,D_zxz,D_zyy,D_zyz,D_zzz,K_xx,K_xy,K_xz,K_yy,K_yz,K_zz,V_x,V_y,V_z
A_x:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha*f*gUxx,alpha*f*gUxy,alpha*f*gUxz,alpha*f*gUyy,alpha*f*gUyz,alpha*f*gUzz,0,0,0],
A_y:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
A_z:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0,0,0],
D_xxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0,0],
D_xxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0],
D_xyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0],
D_xyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0],
D_xzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0],
D_yxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
K_xx:	[1,0,0,alpha*gUxx,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUxz,0,0,0,0,0,0,0,0,0,0,0,2,0,0],
K_xy:	[0,1/2,0,0,alpha*gUxx,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUxz,0,0,0,0,0,0,0,0,0,0,0,1,0],
K_xz:	[0,0,1/2,0,0,alpha*gUxx,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUxz,0,0,0,0,0,0,0,0,0,0,0,1],
K_yy:	[0,0,0,0,0,0,alpha*gUxx,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUxz,0,0,0,0,0,0,0,0,0,0,0],
K_yz:	[0,0,0,0,0,0,0,alpha*gUxx,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUxz,0,0,0,0,0,0,0,0,0,0],
K_zz:	[0,0,0,0,0,0,0,0,alpha*gUxx,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUxz,0,0,0,0,0,0,0,0,0],
V_x:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
V_y:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
V_z:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

d/dy:	A_x,A_y,A_z,D_xxx,D_xxy,D_xxz,D_xyy,D_xyz,D_xzz,D_yxx,D_yxy,D_yxz,D_yyy,D_yyz,D_yzz,D_zxx,D_zxy,D_zxz,D_zyy,D_zyz,D_zzz,K_xx,K_xy,K_xz,K_yy,K_yz,K_zz,V_x,V_y,V_z
A_x:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
A_y:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha*f*gUxx,alpha*f*gUxy,alpha*f*gUxz,alpha*f*gUyy,alpha*f*gUyz,alpha*f*gUzz,0,0,0],
A_z:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0,0,0],
D_yxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0,0],
D_yxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0],
D_yyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0],
D_yyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0],
D_yzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0],
D_zxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
K_xx:	[0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUyy,0,0,0,0,0,alpha*gUyz,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
K_xy:	[1/2,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUyy,0,0,0,0,0,alpha*gUyz,0,0,0,0,0,0,0,0,0,0,1,0,0],
K_xz:	[0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUyy,0,0,0,0,0,alpha*gUyz,0,0,0,0,0,0,0,0,0,0,0,0],
K_yy:	[0,1,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUyy,0,0,0,0,0,alpha*gUyz,0,0,0,0,0,0,0,0,0,2,0],
K_yz:	[0,0,1/2,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUyy,0,0,0,0,0,alpha*gUyz,0,0,0,0,0,0,0,0,0,1],
K_zz:	[0,0,0,0,0,0,0,0,alpha*gUxy,0,0,0,0,0,alpha*gUyy,0,0,0,0,0,alpha*gUyz,0,0,0,0,0,0,0,0,0],
V_x:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
V_y:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
V_z:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

d/dz:	A_x,	A_y,	A_z,	D_xxx,	D_xxy,	D_xxz,	D_xyy,	D_xyz,	D_xzz,	D_yxx,	D_yxy,	D_yxz,	D_yyy,	D_yyz,	D_yzz,	D_zxx,	D_zxy,	D_zxz,	D_zyy,	D_zyz,	D_zzz,	K_xx,	K_xy,	K_xz,	K_yy,	K_yz,	K_zz,	V_x,	V_y,	V_z
A_x:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
A_y:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
A_z:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha*f*gUxx,alpha*f*gUxy,alpha*f*gUxz,alpha*f*gUyy,alpha*f*gUyz,alpha*f*gUzz,0,0,0],
D_xxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_xzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_yzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
D_zxx:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0,0,0],
D_zxy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0,0],
D_zxz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0,0],
D_zyy:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0,0],
D_zyz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0,0],
D_zzz:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,alpha,0,0,0],
K_xx:	[0,0,0,alpha*gUzx,0,0,0,0,0,alpha*gUzy,0,0,0,0,0,alpha*gUzz,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
K_xy:	[0,0,0,0,alpha*gUzx,0,0,0,0,0,alpha*gUzy,0,0,0,0,0,alpha*gUzz,0,0,0,0,0,0,0,0,0,0,0,0,0],
K_xz:	[1/2,0,0,0,0,alpha*gUzx,0,0,0,0,0,alpha*gUzy,0,0,0,0,0,alpha*gUzz,0,0,0,0,0,0,0,0,0,1,0,0],
K_yy:	[0,0,0,0,0,0,alpha*gUzx,0,0,0,0,0,alpha*gUzy,0,0,0,0,0,alpha*gUzz,0,0,0,0,0,0,0,0,0,0,0],
K_yz:	[0,1/2,0,0,0,0,0,alpha*gUzx,0,0,0,0,0,alpha*gUzy,0,0,0,0,0,alpha*gUzz,0,0,0,0,0,0,0,0,1,0],
K_zz:	[0,0,1,0,0,0,0,0,alpha*gUzx,0,0,0,0,0,alpha*gUzy,0,0,0,0,0,alpha*gUzz,0,0,0,0,0,0,0,0,0],
V_x:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
V_y:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
V_z:	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

hmm, looks like the paper includes alpha and g_ij too ...
which means maybe I shouldn't have simplified some of those partial_k alpha = A_k alpha, and partial_k g_ij = 2 D_kij

attempt to find what the correct simplification is using the eigenfields ...
rewrite our few inner products of the Roe.cl to call external functions, and have ADM3D just provide 37 calculations instead of 37x37 matrix 

check out symmath/tests/adm_bona_masso_... for my derivation
*/

#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/ADM3D.h"

__kernel void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenfieldsBuffer,
	const __global real* stateBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) return;
	int index = INDEXV(i);

	int indexPrev = index - stepsize[side];

	int interfaceIndex = index;
	
	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenfield = eigenfieldsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;

	//store the intermediate state in the eigenfield and reconstruct it upon eigenfield[Inverse]Transform()
	for (int i = 0; i < NUM_STATES; ++i) {
		eigenfield[i] = .5f * (stateL[i] + stateR[i]);
	}

	real alpha = eigenfield[0];
	real g_xx = eigenfield[1], g_xy = eigenfield[2], g_xz = eigenfield[3], g_yy = eigenfield[4], g_yz = eigenfield[5], g_zz = eigenfield[6];
	//real A_x = eigenfield[7], A_y = eigenfield[8], A_z = eigenfield[9];
	//real D_xxx = eigenfield[10], D_xxy = eigenfield[11], D_xxz = eigenfield[12], D_xyy = eigenfield[13], D_xyz = eigenfield[14], D_xzz = eigenfield[15];
	//real D_yxx = eigenfield[16], D_yxy = eigenfield[17], D_yxz = eigenfield[18], D_yyy = eigenfield[19], D_yyz = eigenfield[20], D_yzz = eigenfield[21];
	//real D_zxx = eigenfield[22], D_zxy = eigenfield[23], D_zxz = eigenfield[24], D_zyy = eigenfield[25], D_zyz = eigenfield[26], D_zzz = eigenfield[27];
	//real K_xx = eigenfield[28], K_xy = eigenfield[29], K_xz = eigenfield[30], K_yy = eigenfield[31], K_yz = eigenfield[32], K_zz = eigenfield[33];
	//real V_x = eigenfield[34], V_y = eigenfield[35], V_z = eigenfield[36];
	real g = det3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz);
	real8 gInv = inv3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz, g);
	real gUxx = gInv[0], gUxy = gInv[1], gUxz = gInv[2], gUyy = gInv[3], gUyz = gInv[4], gUzz = gInv[5];
	real f = ADM_BONA_MASSO_F;	//could be based on alpha...

	//store only what information is needed for the function of applying the eigenvectors/inverses 
	eigenfield[37] = gUxx;
	eigenfield[38] = gUxy;
	eigenfield[39] = gUxz;
	eigenfield[40] = gUyy;
	eigenfield[41] = gUyz;
	eigenfield[42] = gUzz;
	eigenfield[43] = g;
	eigenfield[44] = f;

	//eigenvalues

	real lambdaLight;
	switch (side) {
	case 0:
		lambdaLight = alpha * sqrt(gUxx); 
		break;
	case 1:
		lambdaLight = alpha * sqrt(gUyy); 
		break;
	case 2:
		lambdaLight = alpha * sqrt(gUzz); 
		break;
	}
	real lambdaGauge = lambdaLight * sqrt(f);
	eigenvalues[0] = -lambdaGauge;
	eigenvalues[1] = -lambdaLight;
	eigenvalues[2] = -lambdaLight;
	eigenvalues[3] = -lambdaLight;
	eigenvalues[4] = -lambdaLight;
	eigenvalues[5] = -lambdaLight;
	eigenvalues[6] = 0.f;
	eigenvalues[7] = 0.f;
	eigenvalues[8] = 0.f;
	eigenvalues[9] = 0.f;
	eigenvalues[10] = 0.f;
	eigenvalues[11] = 0.f;
	eigenvalues[12] = 0.f;
	eigenvalues[13] = 0.f;
	eigenvalues[14] = 0.f;
	eigenvalues[15] = 0.f;
	eigenvalues[16] = 0.f;
	eigenvalues[17] = 0.f;
	eigenvalues[18] = 0.f;
	eigenvalues[19] = 0.f;
	eigenvalues[20] = 0.f;
	eigenvalues[21] = 0.f;
	eigenvalues[22] = 0.f;
	eigenvalues[23] = 0.f;
	eigenvalues[24] = 0.f;
	eigenvalues[25] = 0.f;
	eigenvalues[26] = 0.f;
	eigenvalues[27] = 0.f;
	eigenvalues[28] = 0.f;
	eigenvalues[29] = 0.f;
	eigenvalues[30] = 0.f;
	eigenvalues[31] = lambdaLight;
	eigenvalues[32] = lambdaLight;
	eigenvalues[33] = lambdaLight;
	eigenvalues[34] = lambdaLight;
	eigenvalues[35] = lambdaLight;
	eigenvalues[36] = lambdaGauge;
}

void eigenfieldTransform(
	real* results,
	const __global real* eigenfield,
	const real* input,
	int side)
{
	//real alpha = eigenfield[0];
	//real g_xx = eigenfield[1], g_xy = eigenfield[2], g_xz = eigenfield[3], g_yy = eigenfield[4], g_yz = eigenfield[5], g_zz = eigenfield[6];
	//real A_x = eigenfield[7], A_y = eigenfield[8], A_z = eigenfield[9];
	//real D_xxx = eigenfield[10], D_xxy = eigenfield[11], D_xxz = eigenfield[12], D_xyy = eigenfield[13], D_xyz = eigenfield[14], D_xzz = eigenfield[15];
	//real D_yxx = eigenfield[16], D_yxy = eigenfield[17], D_yxz = eigenfield[18], D_yyy = eigenfield[19], D_yyz = eigenfield[20], D_yzz = eigenfield[21];
	//real D_zxx = eigenfield[22], D_zxy = eigenfield[23], D_zxz = eigenfield[24], D_zyy = eigenfield[25], D_zyz = eigenfield[26], D_zzz = eigenfield[27];
	//real K_xx = eigenfield[28], K_xy = eigenfield[29], K_xz = eigenfield[30], K_yy = eigenfield[31], K_yz = eigenfield[32], K_zz = eigenfield[33];
	//real V_x = eigenfield[34], V_y = eigenfield[35], V_z = eigenfield[36];
	real gUxx = eigenfield[37], gUxy = eigenfield[38], gUxz = eigenfield[39], gUyy = eigenfield[40], gUyz = eigenfield[41], gUzz = eigenfield[42];
	//real g = eigenfield[43];
	real f = eigenfield[44];

	real sqrt_f = sqrt(f);
	if (side == 0) {
		real sqrt_gUxx = sqrt(gUxx);
		real gUxx_toThe_3_2 = sqrt_gUxx * gUxx;

		results[0] = ((((-(2.f * gUxz * input[36])) - (gUxx * input[7])) + (sqrt_f * gUxx_toThe_3_2 * input[28]) + (2.f * sqrt_f * gUxy * input[29] * sqrt_gUxx) + (2.f * sqrt_f * gUxz * input[30] * sqrt_gUxx) + (sqrt_f * gUyy * input[31] * sqrt_gUxx) + (2.f * sqrt_f * gUyz * input[32] * sqrt_gUxx) + (((sqrt_f * gUzz * input[33] * sqrt_gUxx) - (2.f * gUxx * input[34])) - (2.f * gUxy * input[35]))) / sqrt_gUxx);
		results[1] = (((-(gUxx * input[11])) + ((input[29] * sqrt_gUxx) - input[35])) / sqrt_gUxx);
		results[2] = (((-(gUxx * input[12])) + ((input[30] * sqrt_gUxx) - input[36])) / sqrt_gUxx);
		results[3] = ((-(sqrt_gUxx * input[13])) + input[31]);
		results[4] = ((-(sqrt_gUxx * input[14])) + input[32]);
		results[5] = ((-(sqrt_gUxx * input[15])) + input[33]);
		results[6] = input[0];
		results[7] = input[1];
		results[8] = input[2];
		results[9] = input[3];
		results[10] = input[4];
		results[11] = input[5];
		results[12] = input[6];
		results[13] = input[8];
		results[14] = input[9];
		results[15] = input[16];
		results[16] = input[17];
		results[17] = input[18];
		results[18] = input[19];
		results[19] = input[20];
		results[20] = input[21];
		results[21] = input[22];
		results[22] = input[23];
		results[23] = input[24];
		results[24] = input[25];
		results[25] = input[26];
		results[26] = input[27];
		results[27] = input[34];
		results[28] = input[35];
		results[29] = input[36];
		results[30] = ((((((input[7] - (f * gUxx * input[10])) - (2.f * f * gUxy * input[11])) - (2.f * f * gUxz * input[12])) - (f * gUyy * input[13])) - (2.f * f * gUyz * input[14])) - (f * gUzz * input[15]));
		results[31] = (((gUxx * input[11]) + (input[29] * sqrt_gUxx) + input[35]) / sqrt_gUxx);
		results[32] = (((gUxx * input[12]) + (input[30] * sqrt_gUxx) + input[36]) / sqrt_gUxx);
		results[33] = ((sqrt_gUxx * input[13]) + input[31]);
		results[34] = ((sqrt_gUxx * input[14]) + input[32]);
		results[35] = ((sqrt_gUxx * input[15]) + input[33]);
		results[36] = (((2.f * gUxz * input[36]) + (gUxx * input[7]) + (sqrt_f * gUxx_toThe_3_2 * input[28]) + (2.f * sqrt_f * gUxy * input[29] * sqrt_gUxx) + (2.f * sqrt_f * gUxz * input[30] * sqrt_gUxx) + (sqrt_f * gUyy * input[31] * sqrt_gUxx) + (2.f * sqrt_f * gUyz * input[32] * sqrt_gUxx) + (sqrt_f * gUzz * input[33] * sqrt_gUxx) + (2.f * gUxx * input[34]) + (2.f * gUxy * input[35])) / sqrt_gUxx);
	} else if (side == 1) {
		real sqrt_gUyy = sqrt(gUyy);
		real gUyy_toThe_3_2 = sqrt_gUyy * gUyy;

		results[0] = (((((-(2.f * gUyz * input[36])) - (2.f * gUyy * input[35])) - (gUyy * input[8])) + (sqrt_f * gUxx * input[28] * sqrt_gUyy) + (2.f * sqrt_f * gUxy * input[29] * sqrt_gUyy) + (2.f * sqrt_f * gUxz * input[30] * sqrt_gUyy) + (sqrt_f * gUyy_toThe_3_2 * input[31]) + (2.f * sqrt_f * gUyz * input[32] * sqrt_gUyy) + ((sqrt_f * gUzz * input[33] * sqrt_gUyy) - (2.f * gUxy * input[34]))) / sqrt_gUyy);
		results[1] = ((-(sqrt_gUyy * input[16])) + input[28]);
		results[2] = (((-(gUyy * input[17])) + ((input[29] * sqrt_gUyy) - input[34])) / sqrt_gUyy);
		results[3] = ((-(sqrt_gUyy * input[18])) + input[30]);
		results[4] = (((-(gUyy * input[20])) + ((input[32] * sqrt_gUyy) - input[36])) / sqrt_gUyy);
		results[5] = ((-(sqrt_gUyy * input[21])) + input[33]);
		results[6] = input[0];
		results[7] = input[1];
		results[8] = input[2];
		results[9] = input[3];
		results[10] = input[4];
		results[11] = input[5];
		results[12] = input[6];
		results[13] = input[7];
		results[14] = input[9];
		results[15] = input[10];
		results[16] = input[11];
		results[17] = input[12];
		results[18] = input[13];
		results[19] = input[14];
		results[20] = input[15];
		results[21] = input[22];
		results[22] = input[23];
		results[23] = input[24];
		results[24] = input[25];
		results[25] = input[26];
		results[26] = input[27];
		results[27] = input[34];
		results[28] = input[35];
		results[29] = input[36];
		results[30] = ((((((input[8] - (f * gUxx * input[16])) - (2.f * f * gUxy * input[17])) - (2.f * f * gUxz * input[18])) - (f * gUyy * input[19])) - (2.f * f * gUyz * input[20])) - (f * gUzz * input[21]));
		results[31] = ((sqrt_gUyy * input[16]) + input[28]);
		results[32] = (((gUyy * input[17]) + (input[29] * sqrt_gUyy) + input[34]) / sqrt_gUyy);
		results[33] = ((sqrt_gUyy * input[18]) + input[30]);
		results[34] = (((gUyy * input[20]) + (input[32] * sqrt_gUyy) + input[36]) / sqrt_gUyy);
		results[35] = ((sqrt_gUyy * input[21]) + input[33]);
		results[36] = (((2.f * gUyz * input[36]) + (2.f * gUyy * input[35]) + (gUyy * input[8]) + (sqrt_f * gUxx * input[28] * sqrt_gUyy) + (2.f * sqrt_f * gUxy * input[29] * sqrt_gUyy) + (2.f * sqrt_f * gUxz * input[30] * sqrt_gUyy) + (sqrt_f * gUyy_toThe_3_2 * input[31]) + (2.f * sqrt_f * gUyz * input[32] * sqrt_gUyy) + (sqrt_f * gUzz * input[33] * sqrt_gUyy) + (2.f * gUxy * input[34])) / sqrt_gUyy);
	} else if (side == 2) {
		real sqrt_gUzz = sqrt(gUzz);
		real gUzz_toThe_3_2 = sqrt_gUzz * gUzz;
	
		results[0] = (((((-(2.f * gUzz * input[36])) - (2.f * gUyz * input[35])) - (gUzz * input[9])) + (sqrt_f * gUxx * input[28] * sqrt_gUzz) + (2.f * sqrt_f * gUxy * input[29] * sqrt_gUzz) + (2.f * sqrt_f * gUxz * input[30] * sqrt_gUzz) + (sqrt_f * gUyy * input[31] * sqrt_gUzz) + (2.f * sqrt_f * gUyz * input[32] * sqrt_gUzz) + ((sqrt_f * gUzz_toThe_3_2 * input[33]) - (2.f * gUxz * input[34]))) / sqrt_gUzz);
		results[1] = ((-(sqrt_gUzz * input[22])) + input[28]);
		results[2] = ((-(sqrt_gUzz * input[23])) + input[29]);
		results[3] = (((-(gUzz * input[24])) + ((input[30] * sqrt_gUzz) - input[34])) / sqrt_gUzz);
		results[4] = ((-(sqrt_gUzz * input[25])) + input[31]);
		results[5] = (((-(gUzz * input[26])) + ((input[32] * sqrt_gUzz) - input[35])) / sqrt_gUzz);
		results[6] = input[0];
		results[7] = input[1];
		results[8] = input[2];
		results[9] = input[3];
		results[10] = input[4];
		results[11] = input[5];
		results[12] = input[6];
		results[13] = input[7];
		results[14] = input[8];
		results[15] = input[10];
		results[16] = input[11];
		results[17] = input[12];
		results[18] = input[13];
		results[19] = input[14];
		results[20] = input[15];
		results[21] = input[16];
		results[22] = input[17];
		results[23] = input[18];
		results[24] = input[19];
		results[25] = input[20];
		results[26] = input[21];
		results[27] = input[34];
		results[28] = input[35];
		results[29] = input[36];
		results[30] = ((((((input[9] - (f * gUxx * input[22])) - (2.f * f * gUxy * input[23])) - (2.f * f * gUxz * input[24])) - (f * gUyy * input[25])) - (2.f * f * gUyz * input[26])) - (f * gUzz * input[27]));
		results[31] = ((sqrt_gUzz * input[22]) + input[28]);
		results[32] = ((sqrt_gUzz * input[23]) + input[29]);
		results[33] = (((gUzz * input[24]) + (input[30] * sqrt_gUzz) + input[34]) / sqrt_gUzz);
		results[34] = ((sqrt_gUzz * input[25]) + input[31]);
		results[35] = (((gUzz * input[26]) + (input[32] * sqrt_gUzz) + input[35]) / sqrt_gUzz);
		results[36] = (((2.f * gUzz * input[36]) + (2.f * gUyz * input[35]) + (gUzz * input[9]) + (sqrt_f * gUxx * input[28] * sqrt_gUzz) + (2.f * sqrt_f * gUxy * input[29] * sqrt_gUzz) + (2.f * sqrt_f * gUxz * input[30] * sqrt_gUzz) + (sqrt_f * gUyy * input[31] * sqrt_gUzz) + (2.f * sqrt_f * gUyz * input[32] * sqrt_gUzz) + (sqrt_f * gUzz_toThe_3_2 * input[33]) + (2.f * gUxz * input[34])) / sqrt_gUzz);
	}
}

void eigenfieldInverseTransform(
	__global real* results,
	const __global real* eigenfield,
	const real* input,
	int side)
{
	//real alpha = eigenfield[0];
	//real g_xx = eigenfield[1], g_xy = eigenfield[2], g_xz = eigenfield[3], g_yy = eigenfield[4], g_yz = eigenfield[5], g_zz = eigenfield[6];
	//real A_x = eigenfield[7], A_y = eigenfield[8], A_z = eigenfield[9];
	//real D_xxx = eigenfield[10], D_xxy = eigenfield[11], D_xxz = eigenfield[12], D_xyy = eigenfield[13], D_xyz = eigenfield[14], D_xzz = eigenfield[15];
	//real D_yxx = eigenfield[16], D_yxy = eigenfield[17], D_yxz = eigenfield[18], D_yyy = eigenfield[19], D_yyz = eigenfield[20], D_yzz = eigenfield[21];
	//real D_zxx = eigenfield[22], D_zxy = eigenfield[23], D_zxz = eigenfield[24], D_zyy = eigenfield[25], D_zyz = eigenfield[26], D_zzz = eigenfield[27];
	//real K_xx = eigenfield[28], K_xy = eigenfield[29], K_xz = eigenfield[30], K_yy = eigenfield[31], K_yz = eigenfield[32], K_zz = eigenfield[33];
	//real V_x = eigenfield[34], V_y = eigenfield[35], V_z = eigenfield[36];
	real gUxx = eigenfield[37], gUxy = eigenfield[38], gUxz = eigenfield[39], gUyy = eigenfield[40], gUyz = eigenfield[41], gUzz = eigenfield[42];
	//real g = eigenfield[43];
	real f = eigenfield[44];

	real sqrt_f = sqrt(f);

	if (side == 0) {
		real sqrt_gUxx = sqrt(gUxx);
		real gUxx_toThe_3_2 = sqrt_gUxx * gUxx;
	
		results[0] = input[6];
		results[1] = input[7];
		results[2] = input[8];
		results[3] = input[9];
		results[4] = input[10];
		results[5] = input[11];
		results[6] = input[12];
		results[7] = (((-input[36]) + (4.f * gUxz * input[29] * (1.f / sqrt_gUxx)) + (4.f * gUxy * input[28] * (1.f / sqrt_gUxx)) + (4.f * input[27] * sqrt_gUxx) + input[0]) / (-(2.f * sqrt_gUxx)));
		results[8] = input[13];
		results[9] = input[14];
		results[10] = (((-input[36]) + (gUzz * input[35] * f) + (2.f * gUyz * input[34] * f) + (gUyy * input[33] * f) + (2.f * gUxz * input[32] * f) + (2.f * gUxy * input[31] * f) + (2.f * input[30] * sqrt_gUxx) + ((4.f * gUxz * input[29] * (1.f / sqrt_gUxx)) - (4.f * gUxz * f * input[29] * (1.f / sqrt_gUxx))) + ((4.f * gUxy * input[28] * (1.f / sqrt_gUxx)) - (4.f * gUxy * f * input[28] * (1.f / sqrt_gUxx))) + ((((((4.f * input[27] * sqrt_gUxx) - (gUzz * input[5] * f)) - (2.f * gUyz * input[4] * f)) - (gUyy * input[3] * f)) - (2.f * gUxz * input[2] * f)) - (2.f * gUxy * input[1] * f)) + input[0]) / (-(2.f * gUxx_toThe_3_2 * f)));
		results[11] = (((-input[31]) + (2.f * input[28] * (1.f / sqrt_gUxx)) + input[1]) / (-(2.f * sqrt_gUxx)));
		results[12] = (((-input[32]) + (2.f * input[29] * (1.f / sqrt_gUxx)) + input[2]) / (-(2.f * sqrt_gUxx)));
		results[13] = (((-input[33]) + input[3]) / (-(2.f * sqrt_gUxx)));
		results[14] = (((-input[34]) + input[4]) / (-(2.f * sqrt_gUxx)));
		results[15] = (((-input[35]) + input[5]) / (-(2.f * sqrt_gUxx)));
		results[16] = input[15];
		results[17] = input[16];
		results[18] = input[17];
		results[19] = input[18];
		results[20] = input[19];
		results[21] = input[20];
		results[22] = input[21];
		results[23] = input[22];
		results[24] = input[23];
		results[25] = input[24];
		results[26] = input[25];
		results[27] = input[26];
		results[28] = ((((((((((((input[36] - (gUzz * input[35] * sqrt_f)) - (2.f * gUyz * input[34] * sqrt_f)) - (gUyy * input[33] * sqrt_f)) - (2.f * gUxz * input[32] * sqrt_f)) - (2.f * gUxy * input[31] * sqrt_f)) - (gUzz * input[5] * sqrt_f)) - (2.f * gUyz * input[4] * sqrt_f)) - (gUyy * input[3] * sqrt_f)) - (2.f * gUxz * input[2] * sqrt_f)) - (2.f * gUxy * input[1] * sqrt_f)) + input[0]) / (2.f * sqrt_f * gUxx));
		results[29] = ((input[31] + input[1]) / 2.f);
		results[30] = ((input[32] + input[2]) / 2.f);
		results[31] = ((input[33] + input[3]) / 2.f);
		results[32] = ((input[34] + input[4]) / 2.f);
		results[33] = ((input[35] + input[5]) / 2.f);
		results[34] = input[27];
		results[35] = input[28];
		results[36] = input[29];
	} else if (side == 1) {
		real sqrt_gUyy = sqrt(gUyy);
		real gUyy_toThe_3_2 = sqrt_gUyy * gUyy;

		results[0] = input[6];
		results[1] = input[7];
		results[2] = input[8];
		results[3] = input[9];
		results[4] = input[10];
		results[5] = input[11];
		results[6] = input[12];
		results[7] = input[13];
		results[8] = (((-input[36]) + (4.f * gUyz * input[29] * (1.f / sqrt_gUyy)) + (4.f * input[28] * sqrt_gUyy) + (4.f * gUxy * input[27] * (1.f / sqrt_gUyy)) + input[0]) / (-(2.f * sqrt_gUyy)));
		results[9] = input[14];
		results[10] = input[15];
		results[11] = input[16];
		results[12] = input[17];
		results[13] = input[18];
		results[14] = input[19];
		results[15] = input[20];
		results[16] = (((-input[31]) + input[1]) / (-(2.f * sqrt_gUyy)));
		results[17] = (((-input[32]) + (2.f * input[27] * (1.f / sqrt_gUyy)) + input[2]) / (-(2.f * sqrt_gUyy)));
		results[18] = (((-input[33]) + input[3]) / (-(2.f * sqrt_gUyy)));
		results[19] = (((-input[36]) + (gUzz * input[35] * f) + (2.f * gUyz * input[34] * f) + (2.f * gUxz * input[33] * f) + (2.f * gUxy * input[32] * f) + (gUxx * input[31] * f) + (2.f * input[30] * sqrt_gUyy) + ((4.f * gUyz * input[29] * (1.f / sqrt_gUyy)) - (4.f * gUyz * f * input[29] * (1.f / sqrt_gUyy))) + (4.f * input[28] * sqrt_gUyy) + (((((((4.f * gUxy * input[27] * (1.f / sqrt_gUyy)) - (4.f * gUxy * f * input[27] * (1.f / sqrt_gUyy))) - (gUzz * input[5] * f)) - (2.f * gUyz * input[4] * f)) - (2.f * gUxz * input[3] * f)) - (2.f * gUxy * input[2] * f)) - (gUxx * input[1] * f)) + input[0]) / (-(2.f * gUyy_toThe_3_2 * f)));
		results[20] = (((-input[34]) + (2.f * input[29] * (1.f / sqrt_gUyy)) + input[4]) / (-(2.f * sqrt_gUyy)));
		results[21] = (((-input[35]) + input[5]) / (-(2.f * sqrt_gUyy)));
		results[22] = input[21];
		results[23] = input[22];
		results[24] = input[23];
		results[25] = input[24];
		results[26] = input[25];
		results[27] = input[26];
		results[28] = ((input[31] + input[1]) / 2.f);
		results[29] = ((input[32] + input[2]) / 2.f);
		results[30] = ((input[33] + input[3]) / 2.f);
		results[31] = ((((((((((((input[36] - (gUzz * input[35] * sqrt_f)) - (2.f * gUyz * input[34] * sqrt_f)) - (2.f * gUxz * input[33] * sqrt_f)) - (2.f * gUxy * input[32] * sqrt_f)) - (gUxx * input[31] * sqrt_f)) - (gUzz * input[5] * sqrt_f)) - (2.f * gUyz * input[4] * sqrt_f)) - (2.f * gUxz * input[3] * sqrt_f)) - (2.f * gUxy * input[2] * sqrt_f)) - (gUxx * input[1] * sqrt_f)) + input[0]) / (2.f * sqrt_f * gUyy));
		results[32] = ((input[34] + input[4]) / 2.f);
		results[33] = ((input[35] + input[5]) / 2.f);
		results[34] = input[27];
		results[35] = input[28];
		results[36] = input[29];
	} else if (side == 2) {
		real sqrt_gUzz = sqrt(gUzz);
		real gUzz_toThe_3_2 = sqrt_gUzz * gUzz;
	
		results[0] = input[6];
		results[1] = input[7];
		results[2] = input[8];
		results[3] = input[9];
		results[4] = input[10];
		results[5] = input[11];
		results[6] = input[12];
		results[7] = input[13];
		results[8] = input[14];
		results[9] = (((-input[36]) + (4.f * input[29] * sqrt_gUzz) + (4.f * gUyz * input[28] * (1.f / sqrt_gUzz)) + (4.f * gUxz * input[27] * (1.f / sqrt_gUzz)) + input[0]) / (-(2.f * sqrt_gUzz)));
		results[10] = input[15];
		results[11] = input[16];
		results[12] = input[17];
		results[13] = input[18];
		results[14] = input[19];
		results[15] = input[20];
		results[16] = input[21];
		results[17] = input[22];
		results[18] = input[23];
		results[19] = input[24];
		results[20] = input[25];
		results[21] = input[26];
		results[22] = (((-input[31]) + input[1]) / (-(2.f * sqrt_gUzz)));
		results[23] = (((-input[32]) + input[2]) / (-(2.f * sqrt_gUzz)));
		results[24] = (((-input[33]) + (2.f * input[27] * (1.f / sqrt_gUzz)) + input[3]) / (-(2.f * sqrt_gUzz)));
		results[25] = (((-input[34]) + input[4]) / (-(2.f * sqrt_gUzz)));
		results[26] = (((-input[35]) + (2.f * input[28] * (1.f / sqrt_gUzz)) + input[5]) / (-(2.f * sqrt_gUzz)));
		results[27] = (((-input[36]) + (2.f * gUyz * input[35] * f) + (gUyy * input[34] * f) + (2.f * gUxz * input[33] * f) + (2.f * gUxy * input[32] * f) + (gUxx * input[31] * f) + (2.f * input[30] * sqrt_gUzz) + (4.f * input[29] * sqrt_gUzz) + ((4.f * gUyz * input[28] * (1.f / sqrt_gUzz)) - (4.f * gUyz * f * input[28] * (1.f / sqrt_gUzz))) + (((((((4.f * gUxz * input[27] * (1.f / sqrt_gUzz)) - (4.f * gUxz * f * input[27] * (1.f / sqrt_gUzz))) - (2.f * gUyz * input[5] * f)) - (gUyy * input[4] * f)) - (2.f * gUxz * input[3] * f)) - (2.f * gUxy * input[2] * f)) - (gUxx * input[1] * f)) + input[0]) / (-(2.f * gUzz_toThe_3_2 * f)));
		results[28] = ((input[31] + input[1]) / 2.f);
		results[29] = ((input[32] + input[2]) / 2.f);
		results[30] = ((input[33] + input[3]) / 2.f);
		results[31] = ((input[34] + input[4]) / 2.f);
		results[32] = ((input[35] + input[5]) / 2.f);
		results[33] = ((((((((((((input[36] - (2.f * gUyz * input[35] * sqrt_f)) - (gUyy * input[34] * sqrt_f)) - (2.f * gUxz * input[33] * sqrt_f)) - (2.f * gUxy * input[32] * sqrt_f)) - (gUxx * input[31] * sqrt_f)) - (2.f * gUyz * input[5] * sqrt_f)) - (gUyy * input[4] * sqrt_f)) - (2.f * gUxz * input[3] * sqrt_f)) - (2.f * gUxy * input[2] * sqrt_f)) - (gUxx * input[1] * sqrt_f)) + input[0]) / (2.f * sqrt_f * gUzz));
		results[34] = input[27];
		results[35] = input[28];
		results[36] = input[29];
	}
}

__kernel void addSource(
	__global real* derivBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	__global real* deriv = derivBuffer + NUM_STATES * index;
	const __global real* state = stateBuffer + NUM_STATES * index;

	real alpha = state[0];
	real g_xx = state[1], g_xy = state[2], g_xz = state[3], g_yy = state[4], g_yz = state[5], g_zz = state[6];
	//real A_x = state[7], A_y = state[8], A_z = state[9];
	//real D_xxx = state[10], D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	//real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18], D_yyy = state[19], D_yyz = state[20], D_yzz = state[21];
	//real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26], D_zzz = state[27];
	real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	//real V_x = state[34], V_y = state[35], V_z = state[36];
	real g = det3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz);
	real8 gInv = inv3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz, g);
	real gUxx = gInv[0], gUxy = gInv[1], gUxz = gInv[2], gUyy = gInv[3], gUyz = gInv[4], gUzz = gInv[5];
	real f = ADM_BONA_MASSO_F;	//could be based on alpha...
	real tr_K =  K_xx * gUxx + K_yy * gUyy + K_zz * gUzz + 2.f * K_xy * gUxy + 2.f * K_yz * gUyz + 2.f * K_xz * gUxz;
	
	deriv[0] += -alpha * alpha * f * tr_K;
	deriv[1] += -2.f * alpha * K_xx;
	deriv[2] += -2.f * alpha * K_xy;
	deriv[3] += -2.f * alpha * K_xz;
	deriv[4] += -2.f * alpha * K_yy;
	deriv[5] += -2.f * alpha * K_yz;
	deriv[6] += -2.f * alpha * K_zz;
	// TODO dq_dts[i][28..33] == partial_t K_ij += alpha * S_ij
	// partial_t V_k = alpha * P_k
}

__kernel void constrain(
	__global real* stateBuffer)
{
#if 0
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	__global real* state = stateBuffer + NUM_STATES * index;

	//real alpha = state[0];
	real g_xx = state[1], g_xy = state[2], g_xz = state[3], g_yy = state[4], g_yz = state[5], g_zz = state[6];
	//real A_x = state[7], A_y = state[8], A_z = state[9];
	real /*D_xxx = state[10],*/ D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18], /*D_yyy = state[19],*/ D_yyz = state[20], D_yzz = state[21];
	real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26]/*, D_zzz = state[27]*/;
	//real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	//real V_x = state[34], V_y = state[35], V_z = state[36];
	real g = det3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz);
	real8 gInv = inv3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz, g);
	real gUxx = gInv[0], gUxy = gInv[1], gUxz = gInv[2], gUyy = gInv[3], gUyz = gInv[4], gUzz = gInv[5];

	state[34] = 
		2. * (D_xxy - D_yxx) * gUxy
		+ (D_xyy - D_yxy) * gUyy
		+ (D_xyz - D_yxz) * gUyz
		+ 2. * (D_xxz - D_zxx) * gUxz
		+ (D_xyz - D_zxy) * gUyz
		+ (D_xzz - D_zxz) * gUzz;
	state[35] = 
		(D_yxx - D_xxy) * gUxx
		+ (D_yxy - D_xyy) * gUxy
		+ (D_yxz - D_xyz) * gUxz
		+ (D_yxz - D_zxy) * gUxz
		+ (D_yyz - D_zyy) * gUyz
		+ (D_yzz - D_zyz) * gUzz;
	state[36] = 
		(D_zxx - D_xxz) * gUxx
		+ (D_zxy - D_xyz) * gUxy
		+ (D_zxy - D_yxz) * gUxy
		+ (D_zxz - D_xzz) * gUxz
		+ (D_zyy - D_yyz) * gUyy
		+ (D_zyz - D_yzz) * gUyz;
#endif
}
