#pragma once

#include "../include/kinetic_data.h"
#include <cuComplex.h>

__constant__ KDATA dev_dd_;


/** Get a velocity-point on the velocity grid. */
__device__ __host__ __forceinline__
qreal get_v1(YCD vmax, YCD dv, YCU iv){ return (-vmax + dv * iv); }

/** Get a space-point on the spatial grid. */
__device__ __host__ __forceinline__
qreal get_x1(YCD h, YCU ix){ return (h*ix); }

