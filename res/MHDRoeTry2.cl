/*
using the following:
"A Higher-Order Godunov Method for the Equations of Ideal Magnetohydrodynamics" by Zachary and Collela
*/

#include "HydroGPU/Shared/Common.h"

//debugging
//#define DEBUG_OUTPUT
#define DEBUG_INDEX		7

//#define USE_FLUX_FIX

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;

	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;

	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* eigenvectors = eigenvectorsInverse + NUM_STATES * NUM_STATES;

	const real gammaMinusOne = gamma - 1.f;

	real densityL = stateL[STATE_DENSITY];
	real invDensityL = 1.f / densityL;
	real4 velocityL = VELOCITY(stateL);
	real4 magneticFieldL = (real4)(stateL[STATE_MAGNETIC_FIELD_X], stateL[STATE_MAGNETIC_FIELD_Y], stateL[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticEnergyDensityL = .5f * dot(magneticFieldL, magneticFieldL) / vaccuumPermeability;
	real totalPlasmaEnergyDensityL = stateL[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityL = totalPlasmaEnergyDensityL - magneticEnergyDensityL;
	real kineticEnergyDensityL = .5f * densityL * dot(velocityL, velocityL);
	real potentialEnergyL = potentialBuffer[indexPrev];
	real potentialEnergyDensityL = densityL * potentialEnergyL;
	real internalEnergyDensityL = totalHydroEnergyDensityL - kineticEnergyDensityL - potentialEnergyDensityL;
internalEnergyDensityL = max(0.f, internalEnergyDensityL);	//magnetic energy is exceeding total energy ...
	real pressureL = gammaMinusOne * internalEnergyDensityL;
	//real enthalpyTotalL = (totalHydroEnergyDensityL + pressureL) / densityL;
	real roeWeightL = 1.f;//sqrt(densityL);

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1.f / densityR;
	real4 velocityR = VELOCITY(stateR);
	real4 magneticFieldR = (real4)(stateR[STATE_MAGNETIC_FIELD_X], stateR[STATE_MAGNETIC_FIELD_Y], stateR[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticEnergyDensityR = .5f * dot(magneticFieldR, magneticFieldR) / vaccuumPermeability;
	real totalPlasmaEnergyDensityR = stateR[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityR = totalPlasmaEnergyDensityR - magneticEnergyDensityR;
	real kineticEnergyDensityR = .5f * densityR * dot(velocityR, velocityR);
	real potentialEnergyR = potentialBuffer[index];
	real potentialEnergyDensityR = densityR * potentialEnergyR;
	real internalEnergyDensityR = totalHydroEnergyDensityR - kineticEnergyDensityR - potentialEnergyDensityR;
internalEnergyDensityR = max(0.f, internalEnergyDensityR);	//magnetic energy is exceeding total energy ...
	real pressureR = gammaMinusOne * internalEnergyDensityR;
	//real enthalpyTotalR = (totalHydroEnergyDensityR + pressureR) / densityR;
	real roeWeightR = 1.f;//sqrt(densityR);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (velocityL * roeWeightL + velocityR * roeWeightR) * roeWeightNormalization;
	real4 magneticField = (magneticFieldL * roeWeightL + magneticFieldR * roeWeightR) * roeWeightNormalization;
	real pressure = (pressureL * roeWeightL + pressureR * roeWeightR) * roeWeightNormalization;
	real density = .5f * (densityL + densityR); 
	real densitySq = density * density; 

#if DIM > 1
	if (side == 1) {
		velocity.xy = velocity.yx;
		magneticField.xy = magneticField.yx;
	}
#if DIM > 2
	else if (side == 2) {
		velocity.xz = velocity.zx;
		magneticField.xz = magneticField.zx;
	}
#endif
#endif

#ifdef USE_FLUX_FIX
	if (pressure <= 0.f || densitySq <= 0.f) {
		
		real speedOfSound = sqrt(gamma * pressure / density);

		//solve the HLL flux
		//TODO provide a mechanism to write to the flux
		// ... and therefore bail this out of the subsequent flux computation
		//   that is based on these eigenvectors

		real speedOfSoundL = sqrt(gamma * pressureL * invDensityL);
		real speedOfSoundR = sqrt(gamma * pressureR * invDensityR);

		real eigenvaluesMinL = velocityL.x - speedOfSoundL;
		real eigenvaluesMaxR = velocityR.x + speedOfSoundR;

		for (int i = 0; i < NUM_STATES; ++i) {
			eigenvalues[i] = velocity.x;
		}
		eigenvalues[0] -= speedOfSound;
		real eigenvaluesMin = eigenvalues[0];
		eigenvalues[NUM_STATES-1] += speedOfSound;
		real eigenvaluesMax = eigenvalues[NUM_STATES-1];

		//flux

#if 0	//Davis direct
		real sl = eigenvaluesMinL;
		real sr = eigenvaluesMaxR;
#endif
#if 1	//Davis direct bounded
		real sl = min(eigenvaluesMinL, eigenvaluesMin);
		real sr = max(eigenvaluesMaxR, eigenvaluesMax);
#endif

		//for the HLL MHD not sure if I should use the original flux or the symmetrized flux ...
		real fluxL[NUM_STATES];
		fluxL[STATE_DENSITY] = densityL * velocityL.x;
		fluxL[STATE_MOMENTUM_X] = densityL * velocityL.x * velocityL.x +  pressureL - magneticFieldL.x * magneticFieldL.x / vaccuumPermeability;
		fluxL[STATE_MOMENTUM_Y] = densityL * velocityL.y * velocityL.x - magneticFieldL.x * magneticFieldL.y / vaccuumPermeability;
		fluxL[STATE_MOMENTUM_Z] = densityL * velocityL.z * velocityL.x - magneticFieldL.x * magneticFieldL.z / vaccuumPermeability;
		fluxL[STATE_MAGNETIC_FIELD_X] = 0.f;
		fluxL[STATE_MAGNETIC_FIELD_Y] = velocityL.x * magneticFieldL.y - magneticFieldL.x * velocityL.y;
		fluxL[STATE_MAGNETIC_FIELD_Z] = velocityL.x * magneticFieldL.z - magneticFieldL.x * velocityL.z;
		fluxL[STATE_ENERGY_TOTAL] = (totalPlasmaEnergyDensityL + pressureL) * velocityL.x + dot(velocityL, magneticFieldL) * magneticFieldL.x / vaccuumPermeability;

		real fluxR[NUM_STATES];
		fluxR[STATE_DENSITY] = densityR * velocityR.x;
		fluxR[STATE_MOMENTUM_X] = densityR * velocityR.x * velocityR.x + pressureR - magneticFieldR.x * magneticFieldR.x / vaccuumPermeability;
		fluxR[STATE_MOMENTUM_Y] = densityR * velocityR.y * velocityR.x - magneticFieldR.x * magneticFieldR.y / vaccuumPermeability;
		fluxR[STATE_MOMENTUM_Z] = densityR * velocityR.z * velocityR.x - magneticFieldR.x * magneticFieldR.z / vaccuumPermeability;
		fluxR[STATE_MAGNETIC_FIELD_X] = 0.f;
		fluxR[STATE_MAGNETIC_FIELD_Y] = velocityR.x * magneticFieldR.y - magneticFieldR.x * velocityR.y;
		fluxR[STATE_MAGNETIC_FIELD_Z] = velocityR.x * magneticFieldR.z - magneticFieldR.x * velocityR.z;
		fluxR[STATE_ENERGY_TOTAL] = (totalPlasmaEnergyDensityR + pressureR) * velocityR.x + dot(velocityR, magneticFieldR) * magneticFieldR.x / vaccuumPermeability;

		//HLL-specific
		__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;
		__global char* fluxFlag = fluxFlagBuffer + interfaceIndex;

		//1 means we've got something
		*fluxFlag = 1;

		if (0.f <= sl) {
			for (int i = 0; i < NUM_STATES; ++i) {
				flux[i] = fluxL[i];
			}
		} else if (sl <= 0.f && 0.f <= sr) {
			//(sr * fluxL[j] - sl * fluxR[j] + sl * sr * (stateR[j] - stateL[j])) / (sr - sl)
			real invDenom = 1.f / (sr - sl);
			for (int i = 0; i < NUM_STATES; ++i) {
				flux[i] = (sr * fluxL[i] - sl * fluxR[i] + sl * sr * (stateR[i] - stateL[i])) * invDenom;
			}
		} else if (sr <= 0.f) {
			for (int i = 0; i < NUM_STATES; ++i) {
				flux[i] = fluxR[i];
			}
		}

		//rotate back to side
		{
			real tmp = flux[STATE_MOMENTUM_X];
			flux[STATE_MOMENTUM_X] = flux[STATE_MOMENTUM_X + side];
			flux[STATE_MOMENTUM_X + side] = tmp;
		}

	} else
#endif	//USE_FLUX_FIX
	{
		/*
		W = (tau, ux, uy, uz, By, Bz, p)'
		U = (rho, rho*ux, rho*uy, rho*uz, By, Bz, rho*E)'
		tau = 1 / rho
		rho * E = 1/2 * rho * uSq + p / (%gamma - 1) + BTSq / (8 * pi)		<- NOTICE that E is defined in terms of BTSq = By^2 + Bz^2, which means we have to refactor out the normal magnetic field from total energy every time the eigenvector transform is made

		gives
		U = (1/tau, ux/tau, uy/tau, uz/tau, By, Bz, ((4*%gamma−4)*pi*uz^2+(4*%gamma−4)*pi*uy^2+(4*%gamma−4)*pi*ux^2+(8*p*pi+(%gamma−1)*By^2+(%gamma−1)*Bx^2)*tau)/((8*%gamma−8)*pi*tau))'

		dU/dW = (
			[−1/tau^2,	0,	0,	0,	0,	0,	0],
			[−ux/tau^2,	1/tau,	0,	0,	0,	0,	0],
			[−uy/tau^2,	0,	1/tau,	0,	0,	0,	0],
			[−uz/tau^2,	0,	0,	1/tau,	0,	0,	0],
			[0,	0,	0,	0,	1,	0,	0],
			[0,	0,	0,	0,	0,	1,	0],
			[−(uz^2+uy^2+ux^2)/(2*tau^2),	ux/tau,	uy/tau,	uz/tau,	By/(4*pi),	0,	1/%gamma−1]
		)

		dW/dU = (
			[−tau^2,	0,	0,	0,	0,	0,	0],
			[−tau*ux,	tau,	0,	0,	0,	0,	0],
			[−tau*uy,	0,	tau,	0,	0,	0,	0],
			[−tau*uz,	0,	0,	tau,	0,	0,	0],
			[0,	0,	0,	0,	1,	0,	0],
			[0,	0,	0,	0,	0,	1,	0],
			[((%gamma−1)*uz^2+(%gamma−1)*uy^2+(%gamma−1)*ux^2)/2,	(1−%gamma)*ux,	(1−%gamma)*uy,	(1−%gamma)*uz,	−((%gamma−1)*By)/(4*pi),	0,	%gamma−1]
		)

		note that we have 7 variables, not 8, so we will have to pad the eigenvalues
		note that the eigenvalue min/max is set to use 0 and NUM_STATES-1, so I'll have to pad it in the middle somewhere ...

		A = dW/dU * dF/dW	 is the matrix we have eigenvectors and eigenvalues for: R_A and L_A are right and left eigenvectors
		looking for the eigenvectors of dF/dU, the deconstruction being L_U * Lambda_U * R_U
		with transformations: R_U = dU/dW * R_A and L_U = L_A * dW/dU


		eigenvectors of A:
			ux +- v_f
			ux +- v_ax
			ux +- v_s
			ux

		v_ax^2 = Bx^2 / (4 pi rho)
		c_s^2 = gamma p / rho
		v_a^2 = (Bx^2 + By^2 + Bz^2) / (4 pi rho)
		v_f^2 = 1/2 (v_a^2 + c_s^2 + sqrt((v_a^2 + c_s^2)^2 - 4 v_ax^2 c_s^2))
		v_s^2 = 1/2 (v_a^2 + c_s^2 - sqrt((v_a^2 + c_s^2)^2 - 4 v_ax^2 c_s^2))

		R0 =
		[ tau	]
		[ 0		]
		[ 0		]
		[ 0		]
		[ 0		]
		[ 0		]
		[ 0		]

		Rs-+ =
		[	alpha_s tau									-alpha_s tau								]
		[	alpha_s v_s									alpha_s v_s									]
		[	alpha_f beta_y c_s sgn(Bx)					alpha_f beta_y c_s sgn(Bx)					]
		[	alpha_f beta_z c_s sgn(Bx)					alpha_f beta_z c_s sgn(Bx)					]
		[	alpha_f beta_y c_s^2 / v_f sqrt(4 pi rho)	-alpha_f beta_y c_s^2 / v_f sqrt(4 pi rho)	]
		[	alpha_f beta_z c_s^2 / v_f sqrt(4 pi rho)	-alpha_f beta_z c_s^2 / v_f sqrt(4 pi rho)	]
		[	-alpha_s gamma p							alpha_s gamma p								]

		Ras-+ =
		[	0						0						]
		[	0						0						]
		[	-Bz						Bz						]
		[	By						-By						]
		[	-sgn(Bx) sqrt(4 pi) Bz	-sgn(Bx) sqrt(4 pi) Bz	]
		[	sgn(Bx) sqrt(4 pi) By	sgn(Bx) sqrt(4 pi) By	]
		[	0						0						]

		Rf-+ =
		[	alpha_f tau								-alpha_f tau							]
		[	alpha_f v_f								alpha_f v_f							    ]
		[	-alpha_s beta_y v_ax sgn(Bx)			-alpha_s beta_y v_ax sgn(Bx)		    ]
		[	-alpha_s beta_z v_ax sgn(Bx)			-alpha_s beta_z v_ax sgn(Bx)		    ]
		[	-alpha_s beta_y v_f sqrt(4 pi rho)		alpha_s beta_y v_f sqrt(4 pi rho)	    ]
		[	-alpha_s beta_z v_f sqrt(4 pi rho)		alpha_s beta_z v_f sqrt(4 pi rho)	    ]
		[	-alpha_f gamma p						alpha_f gamma p					        ]

		*/
		real magneticFieldSq = dot(magneticField, magneticField);
		real alfvenXSpeedSq = magneticField.x * magneticField.x / (vaccuumPermeability * density);
		real alfvenXSpeed = sqrt(alfvenXSpeedSq);
		real speedOfSoundSq = gamma * pressure / density;	//pressure == hydro pressure
		real speedOfSound = sqrt(speedOfSoundSq);
		real alfvenSpeedSq = magneticFieldSq / (vaccuumPermeability * density);
		real alfvenSpeed = sqrt(alfvenSpeedSq);
		real starSpeedSq = (alfvenSpeedSq + speedOfSoundSq) * .5f;
		real discr = sqrt(starSpeedSq * starSpeedSq - alfvenXSpeedSq * speedOfSoundSq);
		real fastSpeedSq = starSpeedSq + discr;
		real slowSpeedSq = starSpeedSq - discr;
		real fastSpeed = sqrt(fastSpeedSq);
		real slowSpeed = sqrt(slowSpeedSq);

		//don't use sign() function because we only want it to be 1 or -1
		real sgnBx = magneticField.x >= 0.f ? 1.f : -1.f;

		eigenvalues[0] = velocity.x - fastSpeed;
		eigenvalues[1] = velocity.x - alfvenXSpeed;
		eigenvalues[2] = velocity.x - slowSpeed;
		eigenvalues[3] = velocity.x;
		eigenvalues[4] = velocity.x;
		eigenvalues[5] = velocity.x + slowSpeed;
		eigenvalues[6] = velocity.x + alfvenXSpeed;
		eigenvalues[7] = velocity.x + fastSpeed;

		real tau = 1.f / density;

		real alpha_fSq = (fastSpeedSq - alfvenXSpeedSq) / (fastSpeedSq - slowSpeedSq);
		real alpha_sSq = (fastSpeedSq - speedOfSoundSq) / (fastSpeedSq - slowSpeedSq);
		real alpha_f = 1.f;
		real alpha_s = 1.f;
		if (alpha_fSq >= 0.f) {
			alpha_f = sqrt(alpha_fSq);
		} else {
			alpha_fSq = 1.f;
		}
		if (alpha_sSq >= 0.f) {
			alpha_s = sqrt(alpha_sSq);
		} else {
			alpha_sSq = 1.f;
		}

		real magneticFieldTSq = magneticField.y * magneticField.y + magneticField.z * magneticField.z;
		real beta_y = 1.f;	//should this be unit 1,0 in case of no field?
		real beta_z = 0.f;	//or should it be 0,0 ?
		if (magneticFieldTSq > 0.f) {
			real magneticFieldT = sqrt(magneticFieldTSq);
			beta_y = magneticField.y / magneticFieldT;
			beta_z = magneticField.y / magneticFieldT;
		}

		real RFast = fastSpeed / sqrt(alpha_fSq * (fastSpeedSq + speedOfSoundSq) + alpha_sSq * (fastSpeedSq + alfvenXSpeedSq));
		real RSlow = fastSpeed * fastSpeed / sqrt(alpha_fSq * speedOfSoundSq * (fastSpeedSq + speedOfSoundSq) + alpha_sSq * fastSpeedSq * (slowSpeedSq + speedOfSoundSq));

		//specify in columns (so it will appear transposed)
		//...so R_ij == R_A[j][i]
		real8 R_A[8] = {	//right eigenvectors
		//fast -
			(real8)(
				alpha_f * tau,
				alpha_f * fastSpeed,
				-alpha_s * beta_y * alfvenXSpeed * sgnBx,
				-alpha_s * beta_z * alfvenXSpeed * sgnBx,
				0.f,
				-alpha_s * beta_y * fastSpeed * sqrt(vaccuumPermeability * density),
				-alpha_s * beta_z * fastSpeed * sqrt(vaccuumPermeability * density),
				-alpha_f * gamma * pressure
			) * RFast,
		//alfven -
			(real8)(
				0.f,
				0.f,
				-beta_z,
				beta_y,
				0.f,
				-sgnBx * sqrt(vaccuumPermeability * density) * beta_z,
				sgnBx * sqrt(vaccuumPermeability * density) * beta_y,
				0.f
			) * M_SQRT1_2_F * fastSpeed,
		//slow -
			(real8)(
				alpha_s * tau,
				alpha_s * slowSpeed,
				alpha_f * beta_y * speedOfSound * sgnBx,
				alpha_f * beta_z * speedOfSound * sgnBx,
				0.f,
				alpha_f * beta_y * speedOfSoundSq / fastSpeed * sqrt(vaccuumPermeability * density),
				alpha_f * beta_z * speedOfSoundSq / fastSpeed * sqrt(vaccuumPermeability * density),
				-alpha_s * gamma * pressure),
		//entropy
			(real8)(tau, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f),
			(real8)(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f),
		//zero
		//slow +
			(real8)(
				-alpha_s * tau,
				alpha_s * slowSpeed,
				alpha_f * beta_y * speedOfSound * sgnBx,
				alpha_f * beta_z * speedOfSound * sgnBx,
				0.f,
				-alpha_f * beta_y * speedOfSoundSq / fastSpeed * sqrt(vaccuumPermeability * density),
				-alpha_f * beta_z * speedOfSoundSq / fastSpeed * sqrt(vaccuumPermeability * density),
				alpha_s * gamma * pressure),
		//alfven +
			(real8)(
				0.f,
				0.f,
				beta_z,
				-beta_y,
				0.f,
				-sgnBx * sqrt(vaccuumPermeability * density) * beta_z,
				sgnBx * sqrt(vaccuumPermeability * density) * beta_y,
				0.f
			) * M_SQRT1_2_F * fastSpeed,
		//fast +
			(real8)(
				-alpha_f * tau,
				alpha_f * fastSpeed,
				-alpha_s * beta_y * alfvenXSpeed * sgnBx,
				-alpha_s * beta_z * alfvenXSpeed * sgnBx,
				0.f,
				alpha_s * beta_y * fastSpeed * sqrt(vaccuumPermeability * density),
				alpha_s * beta_z * fastSpeed * sqrt(vaccuumPermeability * density),
				alpha_f * gamma * pressure
			) * RFast
		};

		//specify in rows (so it will look as-is)
		//so L_ij = L_A[i][j]
		real8 L_A[8] = {
		//fast -
			(real8)(
				0.f,
				 alpha_f * fastSpeed,
				 -alpha_s * beta_y * alfvenXSpeed * sgnBx,
				 -alpha_s * beta_z * alfvenXSpeed * sgnBx,
				 0.f,
				 -alpha_s * beta_y * fastSpeed / sqrt(vaccuumPermeability * density),
				 -alpha_s * beta_z * fastSpeed / sqrt(vaccuumPermeability * density),
				 -alpha_f * tau
			) * RFast / fastSpeedSq,
		//alfven -
			(real8)(
				0.f,
				0.f,
				-beta_z,
				beta_y,
				0.f,
				-sgnBx * beta_z / sqrt(vaccuumPermeability * density), 
				sgnBx * beta_y / sqrt(vaccuumPermeability * density),
				0.f
			) * M_SQRT1_2_F / fastSpeed,
		//slow -
			(real8)(
				0.f,
				alpha_s * slowSpeed,
				alpha_f * beta_y * speedOfSound * sgnBx,
				alpha_f * beta_z * speedOfSound * sgnBx,
				0.f,
				alpha_f * beta_y * speedOfSoundSq / (sqrt(vaccuumPermeability * density) * fastSpeed),
				alpha_f * beta_z * speedOfSoundSq / (sqrt(vaccuumPermeability * density) * fastSpeed),
				-alpha_s * gamma * pressure
			) * RSlow / fastSpeedSq,
		//entropy
			(real8)(density, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f / (gamma * pressure)),
		//zero
			(real8)(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f),
		//slow +
			(real8)(
				0.f,
				alpha_s * slowSpeed,
				alpha_f * beta_y * speedOfSound * sgnBx,
				alpha_f * beta_z * speedOfSound * sgnBx,
				0.f,
				-alpha_f * beta_y * speedOfSoundSq / (sqrt(vaccuumPermeability * density) * fastSpeed),
				-alpha_f * beta_z * speedOfSoundSq / (sqrt(vaccuumPermeability * density) * fastSpeed),
				alpha_s * gamma * pressure
			) * RSlow / fastSpeedSq,
		//alfven +
			(real8)(
				0.f,
				0.f, 
				beta_z,
				-beta_y, 
				0.f, 
				-sgnBx * beta_z / sqrt(vaccuumPermeability * density), 
				sgnBx * beta_y / sqrt(vaccuumPermeability * density), 
				0.f
			) * M_SQRT1_2_F / fastSpeed,
		//fast +
			(real8)(
				0.f,
				 alpha_f * fastSpeed,
				 -alpha_s * beta_y * alfvenXSpeed * sgnBx,
				 -alpha_s * beta_z * alfvenXSpeed * sgnBx,
				 0.f,
				 alpha_s * beta_y * fastSpeed / sqrt(vaccuumPermeability * density),
				 alpha_s * beta_z * fastSpeed / sqrt(vaccuumPermeability * density),
				 alpha_f * tau
			) * RFast / fastSpeedSq
		};

		real tauSq = tau * tau;
		real velocitySq = dot(velocity, velocity);
		real4 momentum = velocity * density;
		real momentumSq = dot(momentum, momentum);

		//specified by row, so dU_dW[i][j] == (dU/dW)_ij
		real8 dU_dW[8] = {
			(real8)(-densitySq,					0.f,		0.f,		0.f,		0.f,									0.f,									0.f,									0.f),
			(real8)(-velocity.x*densitySq,		density,	0.f,		0.f,		0.f,									0.f,									0.f,									0.f),
			(real8)(-velocity.y*densitySq,		0.f,		density,	0.f,		0.f,									0.f,									0.f,									0.f),
			(real8)(-velocity.z*densitySq,		0.f,		0.f,		density,	0.f,									0.f,									0.f,									0.f),
			(real8)(0.f,						0.f,		0.f,		0.f,		1.f,									0.f,									0.f,									0.f),
			(real8)(0.f,						0.f,		0.f,		0.f,		0.f,									1.f,									0.f,									0.f),
			(real8)(0.f,						0.f,		0.f,		0.f,		0.f,									0.f,									1.f,									0.f),
			(real8)(-.5f*momentumSq,			momentum.x,	momentum.y,	momentum.z,	magneticField.x/vaccuumPermeability,	magneticField.y/vaccuumPermeability,	magneticField.z/vaccuumPermeability,	1.f/gammaMinusOne)
		};

		//specified by row, so dW_dU[i][j] == (dW/dU)_ij
		real8 dW_dU[8] = {
			(real8)(-tauSq,				0.f,		0.f,		0.f,		0.f,									0.f,									0.f,									0.f),
			(real8)(-tau*velocity.x,	tau,		0.f,		0.f,		0.f,									0.f,									0.f,									0.f),
			(real8)(-tau*velocity.y,	0.f,		tau,		0.f,		0.f,									0.f,									0.f,									0.f),
			(real8)(-tau*velocity.z,	0.f,		0.f,		tau,		0.f,									0.f,									0.f,									0.f),
			(real8)(0.f,				0.f,		0.f,		0.f,		1.f,									0.f,									0.f,									0.f),
			(real8)(0.f,				0.f,		0.f,		0.f,		0.f,									1.f,									0.f,									0.f),
			(real8)(0.f,				0.f,		0.f,		0.f,		0.f,									0.f,									1.f,									0.f),
			(real8)(.5f*velocitySq,		velocity.x,	velocity.y,	velocity.z,	-magneticField.x/vaccuumPermeability,	-magneticField.y/vaccuumPermeability,	-magneticField.z/vaccuumPermeability,	1.f) * gammaMinusOne 
		};

		//now transform these to the left and right eigenvectors of the flux ...
		//with transformations: R_U = dU/dW * R_A and L_U = L_A * dW/dU
		//don't forget indexing is A_ij == A[i][j] except R_A is transposed
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				real sum = 0.f;
				for (int k = 0; k < NUM_STATES; ++k) {
					sum += dU_dW[i][k] * R_A[j][k];
				}
				eigenvectors[i + NUM_STATES * j] = sum;
			}
			for (int j = 0; j < NUM_STATES; ++j) {
				real sum = 0.f;
				for (int k = 0; k < NUM_STATES; ++k) {
					sum += L_A[i][k] * dW_dU[k][j];
				}
				eigenvectorsInverse[i + NUM_STATES * j] = sum;
			}
		}

#ifdef DEBUG_OUTPUT
		if (index == DEBUG_INDEX) {
			printf("slowSpeed %f\n", slowSpeed);
			printf("alfvenXSpeed %f\n", alfvenXSpeed);
			printf("alfvenSpeed %f\n", alfvenSpeed);
			printf("fastSpeed %f\n", fastSpeed);
			printf("alpha_f %f\n", alpha_f);
			printf("alpha_s %f\n", alpha_s);
			printf("beta_y %f\n", beta_y);
			printf("beta_z %f\n", beta_z);
			printf("RFast %f\n", RFast);
			printf("RSlow %f\n", RSlow);
		
			printf("fastSpeedSq %f\n", fastSpeedSq);
			printf("alfvenXSpeedSq %f\n", alfvenXSpeedSq);
			printf("speedOfSoundSq %f\n", speedOfSoundSq);
			printf("starSpeedSq %f\n", starSpeedSq);
			printf("slowSpeedSq %f\n", slowSpeedSq);
			printf("discr %f\n", discr);
		
			for (int i = 0; i < 8; ++i) {
				printf("eigenvalues[%d] = %f\n", i, eigenvalues[i]);
			}

/*
			printf("eigenvectors\n");
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					printf("\t%f", eigenvectors[j + NUM_STATES * i]);
				}
				printf("\n");
			}
			
			printf("eigenvectors inverse\n");
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					printf("\t%f", eigenvectorsInverse[j + NUM_STATES * i]);
				}
				printf("\n");
			}
*/
			real8 tmp8[8];
			real tmp = 0.f;
			tmp = 0.f;
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					real sum = 0.f;
					for (int k = 0; k < 8; ++k) {
						sum += eigenvectorsInverse[k + 8 * i] * eigenvectors[j + 8 * k];
					}
					tmp += fabs(sum - (i == j ? 1.f : 0.f));
				}
			}
			printf("Q * Q^-1 total error %f\n", tmp);

/*
			printf("dU_dW\n");
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					printf("\t%f", dU_dW[i][j]);
				}
				printf("\n");
			}
			
			printf("dW_dU\n");
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					printf("\t%f", dW_dU[i][j]);
				}
				printf("\n");
			}
*/				
			//printf("dU/dW * dW/dU\n");
			tmp = 0.f;
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					real sum = 0.f;
					for (int k = 0; k < 8; ++k) {
						sum += dU_dW[i][k] * dW_dU[k][j];
					}
					//printf("\t%f", sum);
					tmp += fabs(sum - (i == j ? 1.f : 0.f));
				}
				//printf("\n");
			}
			printf("dU/dW * dW/dU total error %f\n", tmp);
		
			printf("R_A\n");
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					printf("\t%f", R_A[j][i]);
				}
				printf("\n");
			}
			
			printf("L_A\n");
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					printf("\t%f", L_A[i][j]);
				}
				printf("\n");
			}

			printf("L_A * R_A\n");
			tmp = 0.f;
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					real sum = 0.f;
					for (int k = 0; k < 8; ++k) {
						sum += L_A[i][k] * R_A[j][k];
					}
					printf("\t%f", sum);
					tmp += fabs(sum - (i == j ? 1.f : 0.f));
				}
				printf("\n");
			}
			printf("L_A * R_A total error %f\n", tmp);
		
		}
#endif	//DEBUG_OUTPUT
	}

#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;

			//each row's xy <- yx
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y] = tmp;

			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y] = tmp;

			//each column's xy <- yx
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;

			tmp = eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = eigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;

			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;

			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z] = tmp;

			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i] = tmp;

			tmp = eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = eigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif
}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer)
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

	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, potentialBuffer, fluxBuffer, fluxFlagBuffer, 0);
#if DIM > 1
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, potentialBuffer, fluxBuffer, fluxFlagBuffer, 1);
#endif
#if DIM > 2
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, potentialBuffer, fluxBuffer, fluxFlagBuffer, 2);
#endif
}

//just like calcFlux except if the flux flag is already set then don't do it
__kernel void calcMHDFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenvectorsBuffer,
	const __global real* deltaQTildeBuffer,
	const __global real* dtBuffer,
	const __global char* fluxFlagBuffer)
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

	float dt = dtBuffer[0];

	int index = INDEXV(i);
	if (!fluxFlagBuffer[0 + DIM * index]) {
		calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, deltaQTildeBuffer, dt / DX, 0);
	}
#if DIM > 1
	if (!fluxFlagBuffer[1 + DIM * index]) {
		calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, deltaQTildeBuffer, dt / DY, 1);
	}
#endif
#if DIM > 2
	if (!fluxFlagBuffer[2 + DIM * index]) {
		calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, deltaQTildeBuffer, dt / DZ, 2);
	}
#endif
}


