/*
The components of the Roe solver specific to the BSSNOK equations

taken from p.175 of "Introduction to 3+1 Numerical Relativity" by Alcubierre

hypersurface spatial dimensionality:
	n = 3

conservative hyperbolic BSSNOK variables:
	a_i = partial_i ln alpha
	Phi_i = partial_i phi = 1/2n Gamma^m_im
	dTilde_ijk = 1/2 partial_i gammaTilde_jk

BSSNOK variables:
	conformal trace-free extrinsic curvature:
		ATilde_ij = exp(-4 phi) A_ij = exp(-4 phi) (K_ij - 1/n gamma_ij K) = exp(-4 phi) (K_ij - 1/n exp(4 phi) gammaTilde_ij K) = exp(-4 phi) K_ij - 1/n gammaTilde_ij K
	extrinsic curvature trace:
		K = K^i_i
	contraction of Christoffel symbol of conformal metric
		GammaTilde^i = gammaTilde^jk GammaTilde^i_jk = 1/2 gammaTilde^jk (partial_k gammaTilde_ij + partial_j gammaTilde_ik - partial_i gammaTilde_jk)
	conformal trace-free metric:
		gammaTilde_ij = exp(-4 phi) gamma_ij = exp(-1/n ln det gamma_kl) gamma_ij = gamma_ij (det gamma_kl)^(-1/n),
			such that det gammaTilde_ij = det (gamma_ij (det gamma_kl)^(-1/n)) 
			... factor out conformal factor from determinant, raise to the power of the matrix dimension ...
			= (det gamma_ij) / (det gamma_kl)^(1/n)^n = (det gamma_ij) / (det gamma_kl) = 1
	log of metric determinant:
		exp(-4 phi) = exp(-1/n ln gamma)
		phi = 1/4n ln gamma = 1/4n ln det gamma_ij

fundamental variables:
	gamma_ij = spatial metric
	alpha = lapse
	beta^i = shift


state variable evolution equations (up to principle part):
	
	partial_0 a_i = -alpha partial_i Q
	partial_0 Phi_i = -1/6 alpha partial_i K
	partial_0 dTilde_ijk = -alpha partial_i ATilde_jk
	partial_0 K = -alpha exp(-4 phi) gammaTilde^mn partial_m a_n
	partial_0 GammaTilde^i = alpha partial_k ((xi - 2) ATilde^ik - 2/3 xi gammaTilde^ik K)

... for partial_0 = partial_t - beta^i partial_i


1D:
Bona-Masso slicing condition: Q = f K
treating f as a constant
n = 1 for 1-dimensional hypersurface 
equations:
	phi = -1/(4*n) ln gamma_xx
	gammaTilde_xx = exp(-4 phi) gamma_xx = exp(1/n ln gamma_xx) gamma_xx = gamma_xx / gamma_xx^1 = 1
	dTilde_xxx = 1/2 partial_x gammaTilde_xx = 1/2 partial_x 1 = 0	<- state variable eliminated
	GammaTilde^x = gammaTilde^xx GammaTilde^x_xx = 1/2 (partial_k gammaTilde_ij + partial_j gammaTilde_ik - partial_i gammaTilde_jk) = 0

3 state variables:
	a_x = partial_x ln alpha
	Phi_x = partial_x phi = -1/4n partial_x ln gamma_xx
	K = K^i_i

evolution equations:
partial_t a_x - beta^x partial_x a_x = -alpha partial_x (f K) = -alpha f partial_x K
partial_t Phi_x - beta^x partial_x Phi_x = -1/6 alpha partial_x K
partial_t K - beta^x partial_x K = -alpha exp(-4 phi) gammaTilde^xx partial_x a_x

solve for partial_t u_a + M_ab partial_x u_b = S_a for state indexes a,b:
partial_t a_x - beta^x partial_x a_x + alpha f partial_x K = 0
partial_t Phi_x - beta^x partial_x Phi + 1/6 alpha partial_x K = 0
partial_t K - beta^x partial_x K + alpha exp(-4 phi) partial_x a_x = 0

no shift: beta^x = 0
partial_t a_x + alpha f partial_x K = 0
partial_t Phi_x + 1/6 alpha partial_x K = 0
partial_t K + alpha exp(-4 phi) partial_x a_x = 0


matrix:
          [ a_x ]   [0, 				0, 	alpha f		] 			[ a_x ]   [0]
partial_t [Phi_x] - [0,					0,	1/6 alpha	] partial_x [Phi_x] = [0]
          [  K  ]   [alpha exp(-4 phi), 0, 	0			]			[  K  ]   [0]

...gives us eigenvectors
[-lambda, 0, alpha f]
[0, -lambda, alpha/6]
[alpha exp(-4 phi), 0, -lambda] = 0
-lambda (lambda^2) + (alpha f)(lambda alpha exp(-4 phi)) = 0
lambda (-lambda^2 + f alpha^2 exp(-4 phi)) = 0
lambda = 0, lambda = +/- alpha exp(-2 phi) sqrt(f)

for lambda = 0:
[0, 0, alpha f]					a_x = 0	<=> alpha = constant
[0, 0, alpha/6]					Phi_x = a	<=> phi = a x + b for constants a, b
alpha exp(-4 phi), 0, 0] <=>	K = 0
... has eigenvector [0, 1, 0]

for lambda = +/- alpha exp(-4 phi) sqrt(f)
[-/+ alpha exp(-2 phi) sqrt(f),		0,					alpha f	]
[0,					-/+ alpha exp(-2 phi) sqrt(f), 		alpha/6	]
[alpha exp(-4 phi), 	0,		-/+ alpha exp(-2 phi) sqrt(f)	]
reduces ...
[alpha exp(-4 phi),		0,			-/+ exp(-2 phi) alpha^2 / sqrt(f)]
[0,				-/+ alpha exp(-2 phi) sqrt(f), 		alpha/6	]
[0,					 	0,		-/+ alpha exp(-2 phi) (sqrt(f) - alpha / sqrt(f))	]

2D: (13 states)
	a_x
	a_y
	Phi_x
	Phi_y
	dTilde_xxx
	dTilde_xxy
	dTilde_yxx
	dTilde_yxy
	K
	ATilde_xx
	ATilde_xy
	GammaTilde^x
	GammaTilde^y
Notes:
	dTilde is symmetric, so dTilde_iyx = dTilde_ixy
	dTilde is trace-free, so dTilde_iyy can be derived from dTilde_ixx and dTilde_ixy
	ATilde is symmetric, so ATilde_yx = ATilde_xy
	ATilde is trace-free, so ATilde_yy can be derived from ATilde_xx and ATilde_xy

3D: (30 states)
	a_x
	a_y
	a_z
	Phi_x
	Phi_y
	Phi_z
	dTilde_xxx
	dTilde_xxy
	dTilde_xxz
	dTilde_xyy
	dTilde_xyz
	dTilde_yxx
	dTilde_yxy
	dTilde_yxz
	dTilde_yyy
	dTilde_yyz
	dTilde_zxx
	dTilde_zxy
	dTilde_zxz
	dTilde_zyy
	dTilde_zyz
	K
	ATilde_xx
	ATilde_xy
	ATilde_xz
	ATilde_yy
	ATilde_yz
	GammaTilde^x
	GammaTilde^y
	GammaTilde^z
Notes:
	dTilde_ijk is trace-free along jk, so dTilde_izz can be derived from the other components of dTilde_ijk
	dTilde_ijk is symmetric along jk, so dTilde_ikj = dTilde_ijk
	ATilde_jk is trace-free along jk, so ATilde_zz can be derived from the other components of ATilde_jk
	ATilde_jk is symmetric along jk, so ATilde_kj = ATilde_jk


30 total eigen fields and modes (vectors and values):

let p,q!=x

18 timelike fields:
lambda_timelike = -beta^x
fields = 
	a_q					<- n-1
	Phi_q				<- n-1
	dTilde_qij			<- (n-1)*(n*(n+1)/2 - 1)
	a_x - 6 f Phi_x		<- 1
	GammaTilde^i + (xi - 2) dTilde_m^mi - 4 xi gammaTilde^ik Phi_k	<- n
for n=3 we get 2 + 2 + 2*(3*4/2-1) + 1 + 3 = 18, check

2 gauge fields:
lambda_gauge_+/- = -beta^x +/- alpha exp(-2 phi) sqrt(f gammaTilde^xx) 
fields = exp(-2 phi) sqrt(f gammaTilde^xx) K -/+ a^x

4 longitudinal fields:
lambda_long_+/- = -beta^x +/- alpha exp(-2 phi) sqrt(gammaTilde^xx xi/2)
fields = exp(2 phi) sqrt(gammaTilde^xx xi/2) ATilde^x_q -/+ LambdaTilde^xx_q

4 transverse-traceless light-cone fields: (6 indexes, minus two from the identity of ATilde_pq)
lambda_light_+/- = -beta^x +/- alpha exp(-2 phi) sqrt(gammaTilde^xx)
fields = exp(2 phi) sqrt(gammaTilde^xx) (ATilde_pq + gammaTilde_pq / (2 gammaTilde^xx) ATilde^xx) 
			-/+ (LambdaTilde^x_pq + gammaTilde_pq / (2 gammaTilde^xx) LambdaTilde^xxx)
... with ATilde_pq + gammaTilde_pq / (2 gammaTilde^xx) ATilde^xx = exp(-4 phi) (K_pq - h_pq/2 KTilde)

2 trace of transverse components:
lambda_trace = -beta^x + alpha exp(-2 phi) sqrt(gammaTilde^xx (2 xi - 1) / 3)
fields = exp(2 phi) sqrt(gammaTilde^xx (2 xi - 1) / 3) (ATilde^xx - 2/3 gammaTilde^xx K) -/+ (LambdaTilde^xxx - 2/3 gammaTilde^xx aTilde^x)
... for aTilde^x = gammaTilde^xm a_m
... is only real for 2 xi - 1 > 0 <=> xi > 1/2.

standard BSSNOK is for xi = 2


1D, xi=2:
.. are the fields represented as basis vectors?
for states [a_x, Phi_x, K, GammaTilde^x] 

2 timelike fields:
lambda_timelike = -beta^x
fields:
a_x - 6 f Phi_x
GammaTilde^x - 8 gammaTilde^xx Phi_x

value: -beta^x
vector: [1, -6*f, 0, 0]

value: -beta^x
vector: [0, -8 gammaTilde^xx, 0, 1]
 ... should gammaTilde^xx be converted into Phi_x?  if so, how does a product between the two convert to a vector basis?

2 gauge fields:
lambda_gauge_+/- = -beta^x +/- alpha exp(-2 phi) sqrt(f gammaTilde^xx) 
fields = exp(-2 phi) sqrt(f gammaTilde^xx) K -/+ a^x

value: -beta^x - alpha exp(-2 phi) sqrt(f gammaTilde^xx)
vector: [-1, 0, exp(-2 phi) sqrt(f gammaTilde^xx), 0]

value: -beta^x + alpha exp(-2 phi) sqrt(f gammaTilde^xx)
vector: [1, 0, exp(-2 phi) sqrt(f gammaTilde^xx), 0]

longitudinal fields reduce to zero

transverse-traceless light-cone fields reduce to zero

2 trace of transverse components:
lambda_trace = -beta^x + alpha exp(-2 phi) sqrt(gammaTilde^xx)
fields = 2/3 (gammaTilde^xx)^(3/2) (-exp(2 phi) K +/- sqrt(gammaTilde^xx) a_x)



*/

#include "HydroGPU/Shared/Common.h"

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);

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
		int indexPrev2 = indexPrev - stepsize[side];
		int indexNext = index + stepsize[side];

		int interfaceIndex = side + DIM * index;
		
		const __global real* stateL2 = stateBuffer + NUM_STATES * indexPrev2;
		const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
		const __global real* stateR = stateBuffer + NUM_STATES * index;
		const __global real* stateR2 = stateBuffer + NUM_STATES * indexNext;
		
		__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
		__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
		__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

		//q0 = d/dx ln alpha
		//q1 = d/dx ln g = d/dx ln g_xx
		
		real ln_alphaL = (stateR[STATE_DX_LN_ALPHA] - stateL2[STATE_DX_LN_ALPHA]) / (2.f * dx[side]);
		real alphaL = exp(ln_alphaL);
		real ln_gL = (stateR[STATE_DX_LN_G] - stateL2[STATE_DX_LN_G]) / (2.f * dx[side]);
		real gL = exp(ln_gL);
		real weightL = .5f;
		
		real ln_alphaR = (stateR2[STATE_DX_LN_ALPHA] - stateL[STATE_DX_LN_ALPHA]) / (2.f * dx[side]);
		real alphaR = exp(ln_alphaR);
		real ln_gR = (stateR2[STATE_DX_LN_G] - stateL[STATE_DX_LN_G]) / (2.f * dx[side]);
		real gR = exp(ln_gR);
		real weightR = .5f;
	
		real invDenom = weightL + weightR;
		real alpha = (alphaL * weightL + alphaR * weightR) * invDenom;
		real g = (gL * weightL + gR * weightR) * invDenom;

		const real f = adm_BonaMasso_f;
		
		real sqrtF = sqrt(f);
		real oneOverF = 1.f / f;
		real oneOverSqrtF = sqrt(oneOverF);
		
		real sqrtG = sqrt(g);
		real oneOverSqrtG = 1.f / sqrtG;	
		
		//eigenvalues

		real eigenvalue = alpha * sqrtF * oneOverSqrtG;
		eigenvalues[0] = -eigenvalue;
		eigenvalues[1] = 0.f;
		eigenvalues[2] = eigenvalue;

		//eigenvectors

		//col
		
		eigenvectors[0 + NUM_STATES * 0] = f;
		eigenvectors[1 + NUM_STATES * 0] = 2.f; 
		eigenvectors[2 + NUM_STATES * 0] = -sqrtF;
		//col
		eigenvectors[0 + NUM_STATES * 1] = 0.f;
		eigenvectors[1 + NUM_STATES * 1] = 1.f;
		eigenvectors[2 + NUM_STATES * 1] = 0.f;
		//col
		eigenvectors[0 + NUM_STATES * 2] = f;
		eigenvectors[1 + NUM_STATES * 2] = 2.f;
		eigenvectors[2 + NUM_STATES * 2] = sqrtF;

		//calculate eigenvector inverses ... 
		//min row
		eigenvectorsInverse[0 + NUM_STATES * 0] = .5f * oneOverF;
		eigenvectorsInverse[0 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[0 + NUM_STATES * 2] = -.5f * oneOverSqrtF;
		//mid normal row
		eigenvectorsInverse[1 + NUM_STATES * 0] = -2.f * oneOverF;
		eigenvectorsInverse[1 + NUM_STATES * 1] = 1.f;
		eigenvectorsInverse[1 + NUM_STATES * 2] = 0.f;
		//mid tangent A row
		eigenvectorsInverse[2 + NUM_STATES * 0] = .5f * oneOverF;
		eigenvectorsInverse[2 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 2] = .5f * oneOverSqrtF;
	}
}
