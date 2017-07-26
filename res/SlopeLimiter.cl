#include "HydroGPU/Shared/Common.h"

real slopeLimiter(real r);

real slopeLimiter(real r) {
#ifdef SLOPE_LIMITER_DonorCell
	return 0.;
#endif
#ifdef SLOPE_LIMITER_LaxWendroff
	return 1.;
#endif
#ifdef SLOPE_LIMITER_BeamWarming
	return r;
#endif
#ifdef SLOPE_LIMITER_Fromm
	return .5 * (1. + r);
#endif
#ifdef SLOPE_LIMITER_CHARM
	return (real)max((real)0., (real)r) * (3. * r + 1.) / ((r + 1.) * (r + 1.));
#endif
#ifdef SLOPE_LIMITER_HCUS
	return (real)max((real)0., (real)(1.5 * (r + (real)fabs(r)) / (r + 2.)));
#endif
#ifdef SLOPE_LIMITER_HQUICK
	return (real)max((real)0., (real)(2. * (r + (real)fabs(r)) / (r + 3.)));
#endif
#ifdef SLOPE_LIMITER_Koren
	return (real)max((real)0., (real)min(2. * r, (real)min((1. + 2. * r) / 3., 2.)));
#endif
#ifdef SLOPE_LIMITER_MinMod
	return (real)max((real)0., (real)min(r, (real)1.));
#endif
#ifdef SLOPE_LIMITER_Oshker
	return (real)max((real)0., (real)min(r, (real)1.5));	//replace 1.5 with 1 <= beta <= 2
#endif
#ifdef SLOPE_LIMITER_Ospre
	return .5 * (r * r + r) / (r * r + r + 1.);
#endif
#ifdef SLOPE_LIMITER_Smart
	return (real)max((real)0., (real)min(2. * r, (real)min(.25 + .75 * r, 4.)));
#endif
#ifdef SLOPE_LIMITER_Sweby
	return (real)max((real)0., (real)max((real)min(1.5 * r, 1.), (real)min(r, 1.5)));	//replace 1.5 with 1 <= beta <= 2
#endif
#ifdef SLOPE_LIMITER_UMIST
	return (real)max((real)0., (real)min((real)min(2. * r, .75 + .25 * r), (real)min(.25 + .75 * r, 2.)));
#endif
#ifdef SLOPE_LIMITER_VanAlbada1
	return (r * r + r) / (r * r + 1.);
#endif
#ifdef SLOPE_LIMITER_VanAlbada2
	return 2. * r / (r * r + 1.);
#endif
#ifdef SLOPE_LIMITER_VanLeer
	//Why isn't this working like it is in the JavaScript code?
	return (real)max((real)0., r) * 2. / (1. + r);
	//return (r + (real)fabs(r)) / (1. + (real)fabs(r));
#endif
#ifdef SLOPE_LIMITER_MonotizedCentral
	return (real)max((real)0., (real)min(2., (real)min(.5 * (1. + r), 2. * r)));
#endif
#ifdef SLOPE_LIMITER_Superbee
	return (real)max((real)0., (real)max((real)min((real)1., (real)2. * r), (real)min((real)2., (real)r)));
#endif
#ifdef SLOPE_LIMITER_BarthJespersen
	return .5 * (r + 1.) * (real)min((real)1., (real)min(4. * r / (r + 1.), 4. / (r + 1.)));
#endif
}
