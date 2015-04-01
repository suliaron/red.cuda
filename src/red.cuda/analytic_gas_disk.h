#pragma once
// includes system
#include <string>
#include <vector>

// includes project
#include "gas_disk.h"
#include "red_type.h"

using namespace std;

class analytic_gas_disk
{
public:

	analytic_gas_disk(string& dir, string& filename, bool verbose);
	~analytic_gas_disk();

	//! Calculate the mean free path length and temperature profile
	/*
		\param m_star The mass of the star (time dependent)
	*/
	void calc(var_t	m_star);

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const analytic_gas_disk* g_disk);

	string name;
	string desc;
	
	var2_t rho;   //!< The density of the gas disk in the midplane (time dependent)	
	var2_t sch;   //!< The scale height of the gas disk
	var2_t eta;   //!< Describes how the velocity of the gas differs from the circular velocity	
	var2_t tau;   //!< Describes the Type 2 migartion of the giant planets

	var2_t mfp;   //!< The mean free path of the gas molecules (calculated based on rho, time dependent)	
	var2_t temp;  //!< The temperaterure of the gas (calculated based on sch)
	
	var_t c_vth;  //!< Constant for computing the mean thermal velocity (calculated, constant)

	gas_decrease_t gas_decrease;  //!< The decrease type for the gas density

	ttt_t t0;   //!< Time when the decrease of gas starts (for linear and exponential)
	ttt_t t1;   //!< Time when the linear decrease of the gas ends
	ttt_t e_folding_time; //!< The exponent for the exponential decrease

	var_t alpha;  //!< The viscosity parameter for the Shakura & Sunyaev model (constant)
    var_t mean_molecular_weight;  //!< The mean molecular weight in units of the proton mass (constant)
	var_t particle_diameter;  //!< The mean molecular diameter (constant)

private:
	void set_default_values();
	void parse();
	void set_param(string& key, string& value);

	bool verbose;
	string dir;
	string filename;
	string data;  //!< holds a copy of the file containing the parameters of the simulation
};
