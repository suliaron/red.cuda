// includes system
#include <algorithm>
#include <cmath>
#include <iostream>

// includes project
#include "gas_disk.h"
#include "red_constants.h"
#include "redutilcu.h"
#include "tokenizer.h"

using namespace redutilcu;

gas_disk::gas_disk(string& dir, string& filename, bool verbose) : /* initialization list: */
	gas_decrease(GAS_DENSITY_CONSTANT),
	t0(0.0),
	t1(0.0),
	e_folding_time(0.0),
	c_vth(0.0),
	alpha(0.0),			/* TODO: howto initialize struct in initialization list */
	mean_molecular_weight(0.0),
	particle_diameter(0.0)
{
	eta.x = 0.0, eta.y = 0.0;
	rho.x = 0.0, rho.y = 0.0;
	sch.x = 0.0, sch.y = 0.0;
	tau.x = 0.0, tau.y = 0.0;

	mfp.x = 0.0,  mfp.y = 0.0;
	temp.x = 0.0, temp.y = 0.0;

 	string path = file::combine_path(dir, filename);
	file::load_ascii_file(path, data);
	parse();
	data.clear();
}

gas_disk::~gas_disk()
{
}

void gas_disk::parse()
{
	// instantiate Tokenizer classes
	Tokenizer data_tokenizer;
	Tokenizer line_tokenizer;
	string line;

	data_tokenizer.set(data, "\r\n");

	while ((line = data_tokenizer.next()) != "") {
		line_tokenizer.set(line, "=");
		string token;
		int tokenCounter = 1;

		string key; 
		string value;
		while ((token = line_tokenizer.next()) != "" && tokenCounter <= 2) {

			if (tokenCounter == 1)
				key = token;
			else if (tokenCounter == 2)
				value = token;

			tokenCounter++;
		}
		if (tokenCounter > 2) {
			set_param(key, value);
		}
		else {
			throw string("Invalid key/value pair: " + line + ".");
		}
	}
}

void gas_disk::set_param(string& key, string& value)
{
	static char n_call = 0;

	n_call++;
	tools::trim(key);
	tools::trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "name") {
		name = value;
    } 
    else if (key == "description") {
		desc = value;
    }

	else if (key == "mean_molecular_weight" || key == "mmw") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		mean_molecular_weight = atof(value.c_str());
	}
	else if (key == "particle_diameter" || key == "diameter") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		particle_diameter = atof(value.c_str());
	}

	else if (key == "alpha") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		alpha = atof(value.c_str());
	}

	else if (key == "time_dependence") {
		if (     value == "constant" || value == "const") {
			gas_decrease = GAS_DENSITY_CONSTANT;
		}
		else if (value == "linear" || value == "lin") {
			gas_decrease = GAS_DENSITY_DECREASE_LINEAR;
		}
		else if (value == "exponential" || value == "exp") {
			gas_decrease = GAS_DENSITY_DECREASE_EXPONENTIAL;
		}
		else {
			throw string("Invalid value at: " + key);
		}
	}

	else if (key == "t0") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		t0 = atof(value.c_str()) * constants::YearToDay;
	}
	else if (key == "t1") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		t1 = atof(value.c_str()) * constants::YearToDay;
	}
	else if (key == "e_folding_time") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		e_folding_time = atof(value.c_str()) * constants::YearToDay;
	}

	else if (key == "eta_c") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		eta.x = atof(value.c_str());
	}
    else if (key == "eta_p") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		eta.y = atof(value.c_str());
	}

    else if (key == "rho_c") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		rho.x = atof(value.c_str());
	}
    else if (key == "rho_p") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		rho.y = atof(value.c_str());
	}

    else if (key == "sch_c") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		sch.x = atof(value.c_str());
	}
    else if (key == "sch_p") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		sch.y = atof(value.c_str());
	}

    else if (key == "tau_c") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		tau.x = atof(value.c_str());
	}
    else if (key == "tau_p") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		tau.y = atof(value.c_str());
	}

	else {
		throw string("Invalid parameter :" + key + ".");
	}

	if (verbose) {
		if (n_call == 1) {
			cout << "The following gas disk parameters are setted:" << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << endl;
	}
}

void	gas_disk::calc(var_t m_star)
{
	c_vth = sqrt((8.0 * constants::Boltzman_CMU)/(constants::Pi * mean_molecular_weight * constants::ProtonMass_CMU));

	mfp.x = mean_molecular_weight * constants::ProtonMass_CMU / (sqrt(2.0) * constants::Pi * SQR(particle_diameter) * rho.x);
	mfp.y = -rho.y;

	temp.x = SQR(sch.x) * constants::Gauss2 * m_star * mean_molecular_weight * constants::ProtonMass_CMU / constants::Boltzman_CMU;
	temp.y = 2.0 * sch.y - 3.0;
}

__host__ __device__
var_t	gas_disk::reduction_factor(ttt_t t)
{
	switch (gas_decrease) 
	{
	case GAS_DENSITY_CONSTANT:
		return 1.0;
	case GAS_DENSITY_DECREASE_LINEAR:
		if (t <= t0) {
			return 1.0;
		}
		else if (t0 < t && t <= t1 && t0 != t1) {
			return 1.0 - (t - t0)/(t1 - t0);
		}
		else {
			return 0.0;
		}
	case GAS_DENSITY_DECREASE_EXPONENTIAL:
		return exp(-(t - t0)/e_folding_time);
	default:
		return 1.0;
	}
}

ostream& operator<<(ostream& stream, const gas_disk* g_disk)
{
	const char* gas_decrease_name[] = 
		{
			"GAS_DENSITY_CONSTANT",
			"GAS_DENSITY_DECREASE_LINEAR",
			"GAS_DENSITY_DECREASE_EXPONENTIAL"
		};

	stream << "name: " << g_disk->name << endl;
	stream << "desc: " << g_disk->desc << endl << endl;
	stream << "eta: " << g_disk->eta.x << ", " << g_disk->eta.y << endl;
	stream << "rho: " << g_disk->rho.x << ", " << g_disk->rho.y << endl;
	stream << "sch: " << g_disk->sch.x << ", " << g_disk->sch.y << endl;
	stream << "tau: " << g_disk->tau.x << ", " << g_disk->tau.y << endl << endl;
	stream << " mfp: " << g_disk->mfp.x << ", " << g_disk->mfp.y << endl;
	stream << "temp: " << g_disk->temp.x << ", " << g_disk->temp.y << endl << endl;

	stream << "  gas_decrease: " << gas_decrease_name[g_disk->gas_decrease] << endl;
	stream << "            t0: " << g_disk->t0 << " [d]" << endl;
	stream << "            t1: " << g_disk->t1 << " [d]" << endl;
	stream << "e_folding_time: " << g_disk->e_folding_time << " [d]" << endl << endl;

	stream << " c_vth: " << g_disk->c_vth << endl;
	stream << " alpha: " << g_disk->alpha << endl;
	stream << "mean_molecular_weight: " << g_disk->mean_molecular_weight << endl;
	stream << "    particle_diameter: " << g_disk->particle_diameter << endl;
		
	return stream;
}
