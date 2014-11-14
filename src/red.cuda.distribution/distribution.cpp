// includes, system 
#include <algorithm>
#include <cassert>
#include <cmath>

// includes, project
#include "distribution.h"

using namespace std;

red_random::red_random(unsigned int seed) :
	idx(0)
{
	I[  0] = seed & 0xffffffff;
	for(int i = 1; i < 624; ++i) 
	{
		I[i] = (1812433253*(I[i-1]^I[i-1]>>30)+i)&0xffffffff;
	}
}

red_random::~red_random()
{ } 

unsigned int red_random::rand()
{
	unsigned int j = idx < 623 ? idx + 1 : 0;
	unsigned int y = I[idx]&0x80000000 | I[j]&0x7fffffff;
	y = I[idx] = I[idx < 227 ? idx + 397 : idx-227]^y>>1^(y&1)*0x9908b0df;
	idx = j;
	return y^(y^=(y^=(y^=y>>11)<<7&0x9d2c5680)<<15&0xefc60000)>>18;
}

var_t red_random::uniform()
{
    return ((rand() + 1.0) / 4294967296.0);
}

int red_random::uniform(int a, int b)
{
    return (a + (unsigned int)(rand()/4294967296.*(b - a + 1)));
}

var_t red_random::uniform(var_t a, var_t b)
{
    return (a + (b - a)*(rand() + 1.0) / 4294967296.0);
}

var_t red_random::normal(var_t m, var_t s)
{
    return (m + s*sqrt(-2.0*log((rand() + 1.0) / 4294967296.0)) * cos(1.4629180792671596E-9*(rand() + 1.0)));
}

var_t red_random::exponential(var_t lambda)
{
    var_t u = uniform(); //(rand() + 1.0)/4294967296.0;
    return (-1.0 / lambda * log(u));
}

var_t red_random::rayleigh(var_t sigma)
{
    var_t u = uniform();//(rand() + 1.0)/4294967296.0;
    return (sigma * sqrt(-2.0 * log(u)));
}

var_t red_random::power_law(var_t x_min, var_t x_max, var_t p)
{
	assert(x_min < x_max);

	var_t y_min = pow(x_min, p);
	var_t y_max = pow(x_max, p);
	if (y_min > y_max)
	{
		swap(y_min, y_max);
	}

	var_t d_y = y_max - y_min;
	var_t d_x = x_max - x_min;

	var_t x, y;
	var_t area_max = d_x * d_y;

	do
	{
		x = uniform(0.0, area_max) / d_y + x_min;
		y = uniform(y_min, y_max);
	} while (y > pow(x, p));

	return x;
}


distribution_base::distribution_base(unsigned int seed, var_t x_min, var_t x_max) :
	rr(seed),
	x_min(x_min),
	x_max(x_max)
{ }

distribution_base::~distribution_base()
{ }

rayleigh_distribution::rayleigh_distribution(unsigned int seed, var_t sigma) :
	distribution_base(seed, 0.0, DBL_MAX),
	sigma(sigma)
{ }

rayleigh_distribution::~rayleigh_distribution()
{ }

var_t rayleigh_distribution::get_next()
{
	var_t u = rr.uniform();
	return (sigma * sqrt(-2.0 * log(u)));
}

