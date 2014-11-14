// includes, system 

// includes, project

typedef double var_t;

class red_random
{
public:
	red_random(unsigned int seed);
	~red_random();

	unsigned int rand();

	var_t uniform();
	int   uniform(int a, int b);
	var_t uniform(var_t a, var_t b);

	var_t normal(var_t m, var_t s);
	var_t exponential(var_t lambda);
	var_t rayleigh(var_t sigma);

	var_t power_law(var_t lower, var_t upper, var_t p);

private:
	unsigned int idx;
	unsigned int I[624];
};


class distribution_base
{
public:
	distribution_base(unsigned int seed, var_t x_min, var_t x_max);
	~distribution_base();

	virtual var_t get_next() = 0;

protected:
	var_t x_min;
	var_t x_max;
	red_random rr;
};

class rayleigh_distribution : public distribution_base
{
public:
	rayleigh_distribution(unsigned int seed, var_t sigma);
	~rayleigh_distribution();

	var_t get_next();
private:
	var_t sigma;
};

