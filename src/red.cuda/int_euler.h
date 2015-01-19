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
	euler(pp_disk *ppd, ttt_t dt, bool cpu);
	~euler();

	ttt_t	step();

private:
	void cpu_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result);

	void calc_y_np1(int n_var);
	//void call_kernel_calc_y_np1(int n_var);
	//void cpu_calc_y_np1(int n_var);

	//vector<vec_t*> d_df;	//!< Differentials on the device
	//vector<vec_t*> h_df;	//!< Differentials on the host

	vector<vec_t*> dydx;	//!< Differentials (either in the HOST or the DEVICE memory)
};
