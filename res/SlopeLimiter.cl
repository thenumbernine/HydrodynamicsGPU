#include "HydroGPU/Shared/Common.h"

real8 slopeLimiter(real8 r);

real8 slopeLimiter(real8 r) {
#ifdef SLOPE_LIMITER_DonorCell
	return (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
#endif
#ifdef SLOPE_LIMITER_LaxWendroff
	return (real8)(1.f);
#endif
#ifdef SLOPE_LIMITER_BeamWarming
	return r;
#endif
#ifdef SLOPE_LIMITER_Fromm
	return .5f * (1.f + r);
#endif
#ifdef SLOPE_LIMITER_CHARM
	return max(0.f, r) * (3.f * r + 1.f) / ((r + 1.f) * (r + 1.f));
#endif
#ifdef SLOPE_LIMITER_HCUS
	return max(0.f, 1.5f * (r + fabs(r)) / (r + 2.f));
#endif
#ifdef SLOPE_LIMITER_HQUICK
	return max(0.f, 2.f * (r + fabs(r)) / (r + 3.f));
#endif
#ifdef SLOPE_LIMITER_Koren
	return max(0.f, min(2.f * r, min((1.f + 2.f * r) / 3.f, 2.f)));
#endif
#ifdef SLOPE_LIMITER_MinMod
	return max(0.f, min(r, 1.f));
#endif
#ifdef SLOPE_LIMITER_Oshker
	return max(0.f, min(r, 1.5f));	//replace 1.5 with 1 <= beta <= 2
#endif
#ifdef SLOPE_LIMITER_Ospre
	return .5f * (r * r + r) / (r * r + r + 1.f);
#endif
#ifdef SLOPE_LIMITER_Smart
	return max(0.f, min(2.f * r, min(.25f + .75f * r, 4.f)));
#endif
#ifdef SLOPE_LIMITER_Sweby
	return max(0.f, max(min(1.5f * r, 1.f), min(r, 1.5f)));	//replace 1.5 with 1 <= beta <= 2
#endif
#ifdef SLOPE_LIMITER_UMIST
	return max(0.f, min(min(2.f * r, .75f + .25f * r), min(.25f + .75f * r, 2.f)));
#endif
#ifdef SLOPE_LIMITER_VanAlbada1
	return (r * r + r) / (r * r + 1.f);
#endif
#ifdef SLOPE_LIMITER_VanAlbada2
	return 2.f * r / (r * r + 1.f);
#endif
#ifdef SLOPE_LIMITER_VanLeer
	return (r + fabs(r)) / (1.f + fabs(r));
#endif
#ifdef SLOPE_LIMITER_MonotizedCentral
	return max(0.f, min(2.f, min(.5f * (1.f + r), 2.f * r)));
#endif
#ifdef SLOPE_LIMITER_Superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
#endif
#ifdef SLOPE_LIMITER_BarthJespersen
	return .5f * (r + 1.f) * min(1.f, min(4.f * r / (r + 1.f), 4.f / (r + 1.f)));
#endif
}

