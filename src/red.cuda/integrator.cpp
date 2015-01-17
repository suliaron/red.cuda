// includes system
#include <string>
#include <vector>

// includes CUDA

// includes Thrust

// includes project
#include "integrator.h"
#include "red_constants.h"
#include "red_type.h"
#include "redutilcu.h"

using namespace std;

integrator::integrator(pp_disk *ppd, ttt_t dt, bool cpu) : 
	ppd(ppd),
	dt_try(dt * constants::Gauss), // Transfor time unit
	cpu(cpu),
	error_check_for_tp(false),
	dt_did(0.0),
	dt_next(0.0),
	n_failed_step(0),
	n_passed_step(0),
	n_tried_step(0),
	d_ytemp(2),
	h_ytemp(2),
	ytemp(2)
{
	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	size_t size = n_total * sizeof(vec_t);

	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_VECTOR((void**)&(ytemp[i]), size, cpu);
	}
}

integrator::~integrator()
{
	FREE_VECTOR(ytemp[0], cpu);
	FREE_VECTOR(ytemp[1], cpu);
}

void integrator::update_counters(int iter)
{
	n_tried_step  += iter;
	n_failed_step += (iter - 1);
	n_passed_step++;
}

int integrator::get_n_failed_step()
{
	return n_failed_step;
}

int integrator::get_n_passed_step()
{
	return n_passed_step;
}

int integrator::get_n_tried_step()
{
	return n_tried_step;
}

void integrator::calc_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}
