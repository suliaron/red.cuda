#pragma once

#include "red_type.h"

#define NDIM		4		// Number of dimensions, 4 to coalesce memory copies
#define NTILE		256

#define	NVAR		2		// Number of vector variables per body (coordinate, velocity)
#define NPAR		2		// Number of parameters per body (mass, radius)

#define K			(var_t)0.01720209895
#define K2			(var_t)0.0002959122082855911025

#define	PI			(var_t)3.1415926535897932384626
#define	TWOPI		(var_t)6.2831853071795864769253
#define	TORAD		(var_t)0.0174532925199432957692
#define TODEG		(var_t)57.295779513082320876798
