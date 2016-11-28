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
	return max(0., r) * (3. * r + 1.) / ((r + 1.) * (r + 1.));
#endif
#ifdef SLOPE_LIMITER_HCUS
	return max(0., 1.5 * (r + fabs(r)) / (r + 2.));
#endif
#ifdef SLOPE_LIMITER_HQUICK
	return max(0., 2. * (r + fabs(r)) / (r + 3.));
#endif
#ifdef SLOPE_LIMITER_Koren
	return max(0., min(2. * r, min((1. + 2. * r) / 3., 2.)));
#endif
#ifdef SLOPE_LIMITER_MinMod
	return max(0., min(r, 1.));
#endif
#ifdef SLOPE_LIMITER_Oshker
	return max(0., min(r, 1.5));	//replace 1.5 with 1 <= beta <= 2
#endif
#ifdef SLOPE_LIMITER_Ospre
	return .5 * (r * r + r) / (r * r + r + 1.);
#endif
#ifdef SLOPE_LIMITER_Smart
	return max(0., min(2. * r, min(.25 + .75 * r, 4.)));
#endif
#ifdef SLOPE_LIMITER_Sweby
	return max(0., max(min(1.5 * r, 1.), min(r, 1.5)));	//replace 1.5 with 1 <= beta <= 2
#endif
#ifdef SLOPE_LIMITER_UMIST
	return max(0., min(min(2. * r, .75 + .25 * r), min(.25 + .75 * r, 2.)));
#endif
#ifdef SLOPE_LIMITER_VanAlbada1
	return (r * r + r) / (r * r + 1.);
#endif
#ifdef SLOPE_LIMITER_VanAlbada2
	return 2. * r / (r * r + 1.);
#endif
#ifdef SLOPE_LIMITER_VanLeer
	//Why isn't this working like it is in the JavaScript code?
	return max(0., r) * 2. / (1. + r);
	//return (r + fabs(r)) / (1. + fabs(r));
#endif
#ifdef SLOPE_LIMITER_MonotizedCentral
	return max(0., min(2., min(.5 * (1. + r), 2. * r)));
#endif
#ifdef SLOPE_LIMITER_Superbee
	return max(0., max(min(1., 2. * r), min(2., r)));
#endif
#ifdef SLOPE_LIMITER_BarthJespersen
	return .5 * (r + 1.) * min(1., min(4. * r / (r + 1.), 4. / (r + 1.)));
#endif
}
