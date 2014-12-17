
// includes, project
#include "nebula.h"
#include "red_type.h"
#include "red_macro.h"


gas_component::gas_component(var_t r_1, var_t r_2, var_t r_SL, var_t f_neb, var_t Sigma_1, var_t f_gas, var_t p) :
	r_1(r_1),
	r_2(r_2), 
	r_SL(r_SL),
	f_neb(f_neb),
	Sigma_1(Sigma_1),
	f_gas(f_gas), 
	p(p)
{ }

gas_component::~gas_component()
{ }

var_t gas_component::calc_mass()
{
	// Calculate mass of the gas using a MMSN (Hayhasi, 1981)
	var_t m_gas = 0.0;

	var_t C_gas = 0.0;
	var_t I_gas = 0.0;
	var_t p2 = p + 2.0;
	if (p2 != 0.0)
	{
		// Calculate factor of the integrand
		C_gas = 2.0 * PI * f_neb * f_gas * Sigma_1 / p2;
		// Calculate integrand
		I_gas = pow(r_2, p2) - pow(r_1, p2);
	}
	else
	{
		// Calculate factor of the integrand
		C_gas = 2.0 * PI * f_neb * f_gas * Sigma_1;
		// Calculate integrand
		I_gas = log(r_2) - log(r_1);
	}
	m_gas = C_gas * I_gas;

	return m_gas;
}

solid_component::solid_component(var_t r_1, var_t r_2, var_t r_SL, var_t f_neb, var_t Sigma_1, var_t f_ice, var_t p) :
	r_1(r_1),
	r_2(r_2), 
	r_SL(r_SL),
	f_neb(f_neb),
	Sigma_1(Sigma_1),
	f_ice(f_ice), 
	p(p)
{ }

solid_component::~solid_component()
{ }

var_t solid_component::calc_mass()
{
	// Calculate mass of the solids using a MMSN (Hayhasi, 1981)
	var_t m_solid = 0.0;
		
	var_t p2 = p + 2.0;
	var_t C_solid = p2 != 0.0 ? 2.0 * PI * f_neb * Sigma_1 / p2 : 
							    2.0 * PI * f_neb * Sigma_1;
	var_t I_solid = 0.0;
	if (r_SL <= r_1)
	{
		I_solid = p2 != 0.0 ? f_ice*(pow(r_2, p2) - pow(r_1, p2)) :
							  f_ice*(log(r_2) - log(r_1));
	}
	else if (r_SL <= r_2)
	{
		I_solid = p2 != 0.0 ? pow(r_SL, p2) - pow(r_1, p2) + f_ice*(pow(r_2, p2) - pow(r_SL, p2)) :
							 (log(r_SL) - log(r_1)) + f_ice*((log(r_2) - log(r_SL)));
	}
	else
	{
		I_solid = p2 != 0.0 ? pow(r_2, p2) - pow(r_1, p2) :
							 (log(r_2) - log(r_1));
	}
	m_solid = C_solid * I_solid;

	return m_solid;
}

nebula::nebula(gas_component gas_c, solid_component solid_c) :
	gas_c(gas_c),
	solid_c(solid_c)
{ }

nebula::~nebula()
{ }
