#pragma once

// includes system
#include <string>
#include <vector>

// includes project
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

namespace integrator
{

class euler
{
public:
	euler(ttt_t t0, ttt_t dt, pp_disk *ppd);
	~euler();
	ttt_t	step();

	ttt_t	dt_try;
	ttt_t	dt_did;
	ttt_t	dt_next;
	string name;
private:
	void	calculate_grid(int nData, int threads_per_block);

	dim3	grid;
	dim3	block;

	vector<vec_t*> d_dy;		/*!< Differentials on the device */
	ttt_t	t;					/*!< Actual time of the integrator */
	pp_disk*	ppd;			/*!< simulation data */					
};

} /* namespace integrator */
