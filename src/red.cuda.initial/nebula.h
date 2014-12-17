#pragma once

// includes, project
#include "red_type.h"

class gas_component
{
public:
	gas_component(var_t r_1, var_t r_2, var_t r_SL, var_t f_neb, var_t Sigma_1, var_t f_gas, var_t p);
	~gas_component();

	//! Calculates the mass of the gas using a MMSN (Hayhasi, 1981)
	var_t calc_mass();

	var_t get_r_1()  { return r_1;  }
	var_t get_r_2()  { return r_2;  }
	var_t get_r_SL() { return r_SL; }

	var_t get_p()    { return p;    }

private:
	var_t r_1;		// inner radius of the gas disk [AU]
	var_t r_2;		// outer radius of the gas disk [AU]
	var_t r_SL;		// radial distance of the snowline [AU]

	var_t f_neb;	// nebular mass scaling factor (if f_neb = 1 one gets the MMSN)
	var_t Sigma_1;	// surface density of solids at 1 AU [g/cm^2]
	var_t f_gas;	// gas to dust ratio
	var_t p;		// profile index of the radial gas density function
};

class solid_component
{
public:
	solid_component(var_t r_1, var_t r_2, var_t r_SL, var_t f_neb, var_t Sigma_1, var_t f_ice, var_t p);
	~solid_component();

	//! Calculates the mass of the solid using a MMSN (Hayhasi, 1981)
	var_t calc_mass();

	var_t get_r_1()  { return r_1;  }
	var_t get_r_2()  { return r_2;  }
	var_t get_r_SL() { return r_SL; }

	var_t get_p()    { return p;    }

private:
	var_t r_1;		// inner radius of the gas disk [AU]
	var_t r_2;		// outer radius of the gas disk [AU]
	var_t r_SL;		// radial distance of the snowline [AU]

	var_t f_neb;	// nebular mass scaling factor (if f_neb = 1 one gets the MMSN)
	var_t Sigma_1;	// surface density of solids at 1 AU [g/cm^2]
	var_t f_ice;	// ice condensation coefficient
	var_t p;		// profile index of the radial solid density function
};

class nebula
{
public:
	nebula(gas_component	gas_c, solid_component	solid_c);
	~nebula();

	gas_component	gas_c;
	solid_component	solid_c;

private:
};
