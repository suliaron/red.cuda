// includes, system 
#include <algorithm>
#include <cassert>
#include <cmath>

// includes, project
#include "distribution.h"

using namespace std;

red_random::red_random(uint32_t seed) :
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

uint32_t red_random::rand()
{
	uint32_t j = idx < 623 ? idx + 1 : 0;
	uint32_t y = I[idx]&0x80000000 | I[j]&0x7fffffff;
	y = I[idx] = I[idx < 227 ? idx + 397 : idx-227]^y>>1^(y&1)*0x9908b0df;
	idx = j;
	return y^(y^=(y^=(y^=y>>11)<<7&0x9d2c5680)<<15&0xefc60000)>>18;
}

var_t red_random::uniform()
{
    return ((rand() + 1.0) / 4294967296.0);
}

var_t red_random::uniform(var_t x_min, var_t x_max)
{
    return (x_min + (x_max - x_min)*(rand() + 1.0) / 4294967296.0);
}

int red_random::uniform(int x_min, int x_max)
{
    return (x_min + (uint32_t)(rand()/4294967296.*(x_max - x_min + 1)));
}

//var_t red_random::normal(var_t m, var_t s)
//{
//    return (m + s*sqrt(-2.0*log((rand() + 1.0) / 4294967296.0)) * cos(1.4629180792671596E-9*(rand() + 1.0)));
//}
//
//var_t red_random::exponential(var_t lambda)
//{
//    var_t u = uniform();
//    return (-1.0 / lambda * log(u));
//}
//
//var_t red_random::rayleigh(var_t sigma)
//{
//    var_t u = uniform();
//    return (sigma * sqrt(-2.0 * log(u)));
//}
//
//var_t red_random::power_law(var_t x_min, var_t x_max, var_t p)
//{
//	assert(x_min < x_max);
//
//	var_t y_min = pow(x_min, p);
//	var_t y_max = pow(x_max, p);
//	if (y_min > y_max)
//	{
//		swap(y_min, y_max);
//	}
//
//	var_t d_y = y_max - y_min;
//	var_t d_x = x_max - x_min;
//
//	var_t x, y;
//	var_t area_max = d_x * d_y;
//
//	do
//	{
//		x = uniform(0.0, area_max) / d_y + x_min;
//		y = uniform(y_min, y_max);
//	} while (y > pow(x, p));
//
//	return x;
//}




distribution_base::distribution_base(uint32_t seed, var_t x_min, var_t x_max) :
	rr(seed),
	x_min(x_min),
	x_max(x_max)
{
	assert(x_min <= x_max);
}

distribution_base::~distribution_base()
{ }


uniform_distribution::uniform_distribution(uint32_t seed) :
	distribution_base(seed, 0.0, 1.0)
{ }

uniform_distribution::uniform_distribution(uint32_t seed, var_t x_min, var_t x_max) :
	distribution_base(seed, x_min, x_max)
{ }

uniform_distribution::~uniform_distribution()
{ }

var_t uniform_distribution::get_next()
{
	var_t u = rr.uniform(x_min, x_max);
	return u;
}


exponential_distribution::exponential_distribution(uint32_t seed, var_t lambda) :
	distribution_base(seed, 0.0, 1.0),
	lambda(lambda)
{ 
	assert(lambda > 0.0);
}

exponential_distribution::~exponential_distribution()
{ }

var_t exponential_distribution::get_next()
{
	var_t u = rr.uniform();
    return (-1.0 / lambda * log(u));
}


rayleigh_distribution::rayleigh_distribution(uint32_t seed, var_t sigma) :
	distribution_base(seed, 0.0, 1.0),
	sigma(sigma)
{
	assert(sigma > 0.0);
}

rayleigh_distribution::~rayleigh_distribution()
{ }

var_t rayleigh_distribution::get_next()
{
	var_t u = rr.uniform();
	return (sigma * sqrt(-2.0 * log(u)));
}


normal_distribution::normal_distribution(uint32_t seed, var_t mean, var_t variance) :
	distribution_base(seed, 0.0, 1.0),
	mean(mean),
	variance(variance)
{
	assert(variance > 0.0);
}

normal_distribution::~normal_distribution()
{ }

var_t normal_distribution::get_next()
{
	var_t u = rr.uniform();
	return (mean + variance*sqrt(-2.0*log(u)) * cos(1.4629180792671596E-9*(rr.rand() + 1.0)));
}


power_law_distribution::power_law_distribution(uint32_t seed, var_t x_min, var_t x_max, var_t power):
	distribution_base(seed, x_min, x_max),
	power(power)
{
	assert(x_min != 0.0);
	assert(x_max != 0.0);
}

power_law_distribution::~power_law_distribution()
{ }

var_t power_law_distribution::get_next()
{
	var_t y_min = power == 0.0 ? 0.0 : pow(x_min, power);
	var_t y_max = power == 0.0 ? 1.0 : pow(x_max, power);
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
		x = rr.uniform(0.0, area_max) / d_y + x_min;
		y = rr.uniform(y_min, y_max);
	} while (y > pow(x, power));

	return x;
}


lognormal_distribution::lognormal_distribution(uint32_t seed, var_t x_min, var_t x_max, var_t mu, var_t sigma) :
	distribution_base(seed, x_min, x_max),
	mu(mu),
	sigma(sigma)
{
	assert(x_min > 0.0);
	assert(sigma > 0.0);
}

lognormal_distribution::~lognormal_distribution()
{ }

var_t lognormal_distribution::get_next()
{
	var_t y_min = 0.0;
	var_t max_loc = exp(mu - SQR(sigma));
	var_t y_max = pdf(max_loc);
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
		x = rr.uniform(0.0, area_max) / d_y + x_min;
		y = rr.uniform(y_min, y_max);
	} while (y > pdf(x));

	return x;
}

var_t lognormal_distribution::pdf(var_t x)
{
	return (1.0/(sqrt(2.0*PI)*sigma * x) * exp( -SQR(log(x) - mu) / (2.0 * SQR(sigma))) );
}
