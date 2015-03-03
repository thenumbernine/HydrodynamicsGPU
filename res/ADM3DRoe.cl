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

partial_t alpha = -alpha^2 f tr K
partial_t g_ij = -2 alpha K_ij
partial_t A_k + alpha f g^ij partial_k K_ij = -alpha tr K (f + alpha f') A_k + 2 alpha f K^ij D_kij
partial_t D_kij + alpha partial_k K_ij = -alpha A_k K_ij
partial_t K_ij + alpha (g^km partial_k D_mij + 1/2 (delta^k_i partial_k A_j + delta^k_j partial_k A_i) + delta^k_i partial_k V_j + delta^k_j partial_k V_i) = alpha S_ij - alpha lambda^k_ij A_k + 2 alpha D_mij D_k^km
partial_t V_k = alpha P_k

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

*/

#include "HydroGPU/Shared/Common.h"

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
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

	for (int side = 0; side < DIM; ++side) {
		int indexPrev = index - stepsize[side];

		int interfaceIndex = side + DIM * index;
		
		const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
		const __global real* stateR = stateBuffer + NUM_STATES * index;
		
		__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
		__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
		__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

		//q0 = d/dx ln alpha
		//q1 = d/dx ln g = d/dx ln g_xx
	
		real alpha = .5f * (stateL[STATE_ALPHA] + stateR[STATE_ALPHA]);
		real g = .5f * (stateL[STATE_G] + stateR[STATE_G]);
		real A = .5f * (stateL[STATE_A] + stateR[STATE_A]);
		//real D = .5f * (stateL[STATE_D] + stateR[STATE_D]);
		real K = .5f * (stateL[STATE_K] + stateR[STATE_K]);
		
		const real f = ADM_BONA_MASSO_F;
		
		//eigenvalues

		real eigenvalue = alpha * sqrt(f/g); 
		eigenvalues[0] = -eigenvalue;
		eigenvalues[1] = 0.f;
		eigenvalues[2] = 0.f;
		eigenvalues[3] = 0.f;
		eigenvalues[4] = eigenvalue;

		//eigenvectors

		//col
		eigenvectors[0 + NUM_STATES * 0] = 0.f;
		eigenvectors[1 + NUM_STATES * 0] = 0.f; 
		eigenvectors[2 + NUM_STATES * 0] = f/g; 
		eigenvectors[3 + NUM_STATES * 0] = 1.f; 
		eigenvectors[4 + NUM_STATES * 0] = -sqrt(f/g); 
		//col
		eigenvectors[0 + NUM_STATES * 1] = alpha;
		eigenvectors[1 + NUM_STATES * 1] = 0.f;
		eigenvectors[2 + NUM_STATES * 1] = -A;
		eigenvectors[3 + NUM_STATES * 1] = 0.f;
		eigenvectors[4 + NUM_STATES * 1] = -K;
		//col
		eigenvectors[0 + NUM_STATES * 2] = 0.f;
		eigenvectors[1 + NUM_STATES * 2] = 0.f;
		eigenvectors[2 + NUM_STATES * 2] = 0.f;
		eigenvectors[3 + NUM_STATES * 2] = 1.f;
		eigenvectors[4 + NUM_STATES * 2] = 0.f;
		//col
		eigenvectors[0 + NUM_STATES * 3] = 0.f;
		eigenvectors[1 + NUM_STATES * 3] = 1.f;
		eigenvectors[2 + NUM_STATES * 3] = 0.f;
		eigenvectors[3 + NUM_STATES * 3] = 0.f;
		eigenvectors[4 + NUM_STATES * 3] = 0.f;
		//col
		eigenvectors[0 + NUM_STATES * 4] = 0.f;
		eigenvectors[1 + NUM_STATES * 4] = 0.f;
		eigenvectors[2 + NUM_STATES * 4] = f/g;
		eigenvectors[3 + NUM_STATES * 4] = 1.f;
		eigenvectors[4 + NUM_STATES * 4] = sqrt(f/g);

		//calculate eigenvector inverses ... 
		//min 
		eigenvectorsInverse[0 + NUM_STATES * 0] = (g * A / f - K * sqrt(g / f)) / (2.f * alpha); 
		eigenvectorsInverse[0 + NUM_STATES * 1] = 0.f; 
		eigenvectorsInverse[0 + NUM_STATES * 2] = g / (2.f * f); 
		eigenvectorsInverse[0 + NUM_STATES * 3] = 0.f; 
		eigenvectorsInverse[0 + NUM_STATES * 4] = -.5f * sqrt(g / f); 
		//row
		eigenvectorsInverse[1 + NUM_STATES * 0] = 1.f / alpha;
		eigenvectorsInverse[1 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 4] = 0.f;
		//row
		eigenvectorsInverse[2 + NUM_STATES * 0] = -(g * A) / (alpha * f); 
		eigenvectorsInverse[2 + NUM_STATES * 1] = 0.f; 
		eigenvectorsInverse[2 + NUM_STATES * 2] = -g / f; 
		eigenvectorsInverse[2 + NUM_STATES * 3] = 1.f; 
		eigenvectorsInverse[2 + NUM_STATES * 4] = 0.f; 
		//row
		eigenvectorsInverse[3 + NUM_STATES * 0] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 1] = 1.f;
		eigenvectorsInverse[3 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 4] = 0.f;
		//row
		eigenvectorsInverse[4 + NUM_STATES * 0] = (g * A / f + K * sqrt(g / f)) / (2.f * alpha); 
		eigenvectorsInverse[4 + NUM_STATES * 1] = 0.f; 
		eigenvectorsInverse[4 + NUM_STATES * 2] = g / (2.f * f); 
		eigenvectorsInverse[4 + NUM_STATES * 3] = 0.f; 
		eigenvectorsInverse[4 + NUM_STATES * 4] = .5f * sqrt(g / f); 
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
	real g = state[1];
	real A = state[2];
	real D = state[3];
	real K = state[4];
	real f = ADM_BONA_MASSO_F;
	deriv[STATE_ALPHA] += -alpha * alpha * f * K / g;
	deriv[STATE_G] += -2.f * alpha * K;
	deriv[STATE_K] += + alpha * (A * D - K * K) / g;
}

