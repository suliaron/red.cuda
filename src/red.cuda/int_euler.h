#pragma once
// includes system
#include <string>
#include <vector>

// includes project
#include "integrator.h"
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

class euler : public integrator
{
public:
	euler(pp_disk *ppd, ttt_t dt);
	~euler();

	ttt_t	step();

private:
	void	call_kernel_calc_y_np1();

	vector<vec_t*> d_df;		/*!< Differentials on the device */
};
