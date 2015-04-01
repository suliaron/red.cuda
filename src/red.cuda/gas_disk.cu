// includes system
#include <string>

// includes CUDA

// includes project
#include "red_type.h"
#include "red_macro.h"
#include "gas_disk.h"

using namespace std;

gas_disk::gas_disk(string& dir, string& filename, bool verbose) :
	dir(dir),
	filename(filename),
	verbose(verbose)
{
}

gas_disk::~gas_disk()
{
}

//__host__ __device__
//vec_t gas_disk::circular_velocity(var_t mu, const vec_t* rVec)
//{
//	vec_t result = {0.0, 0.0, 0.0, 0.0};
//
//	var_t r  = sqrt(SQR(rVec->x) + SQR(rVec->y));
//	var_t vc = sqrt(mu/r);
//
//	var_t p = 0.0;
//	if (rVec->x == 0.0 && rVec->y == 0.0)
//	{
//		return result;
//	}
//	else if (rVec->y == 0.0)
//	{
//		result.y = rVec->x > 0.0 ? vc : -vc;
//	}
//	else if (rVec->x == 0.0)
//	{
//		result.x = rVec->y > 0.0 ? -vc : vc;
//	}
//	else if (rVec->x >= rVec->y)
//	{
//		p = rVec->y / rVec->x;
//		result.y = rVec->x >= 0 ? vc/sqrt(1.0 + SQR(p)) : -vc/sqrt(1.0 + SQR(p));
//		result.x = -result.y*p;
//	}
//	else
//	{
//		p = rVec->x / rVec->y;
//		result.x = rVec->y >= 0 ? -vc/sqrt(1.0 + SQR(p)) : vc/sqrt(1.0 + SQR(p));
//		result.y = -result.x*p;
//	}
//
//	return result;
//}
