// includes, system 

// includes, project

typedef double var_t;

#define PI		3.1415926535897932384626433832795

#define SQR(x)	((x)*(x))

class red_random
{
public:
	red_random(unsigned int seed);
	~red_random();

	unsigned int rand();

	var_t uniform();
	var_t uniform(var_t x_min, var_t x_max);
	int   uniform(int x_min, int x_max);

	//var_t normal(var_t m, var_t s);
	//var_t exponential(var_t lambda);
	//var_t rayleigh(var_t sigma);

	//var_t power_law(var_t lower, var_t upper, var_t p);

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

class uniform_distribution : public distribution_base
{
public:
	uniform_distribution(unsigned int seed);
	uniform_distribution(unsigned int seed, var_t x_min, var_t x_max);
	~uniform_distribution();

	var_t get_next();
private:
	var_t sigma;
};

class exponential_distribution : public distribution_base
{
public:
	exponential_distribution(unsigned int seed, var_t lambda);
	~exponential_distribution();

	var_t get_next();
private:
	var_t lambda;
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

class normal_distribution : public distribution_base
{
public:
	normal_distribution(unsigned int seed, var_t mean, var_t variance);
	~normal_distribution();

	var_t get_next();
private:
	var_t mean;
	var_t variance;
};

class power_law_distribution : public distribution_base
{
public:
	power_law_distribution(unsigned int seed, var_t x_min, var_t x_max, var_t power);
	~power_law_distribution();

	var_t get_next();
private:
	var_t power;
};

class lognormal_distribution : public distribution_base
{
public:
	lognormal_distribution(unsigned int seed, var_t x_min, var_t x_max, var_t mu, var_t sigma);
	~lognormal_distribution();

	var_t get_next();
private:
	var_t pdf(var_t);

	var_t mu;
	var_t sigma;
};
