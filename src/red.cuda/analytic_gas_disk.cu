// includes system
#include <algorithm>
#include <cmath>
#include <iostream>

// includes project
#include "gas_disk.h"
#include "analytic_gas_disk.h"
#include "red_constants.h"
#include "redutilcu.h"
#include "tokenizer.h"
#include "red_macro.h"

using namespace redutilcu;

analytic_gas_disk::analytic_gas_disk(string& dir, string& filename, bool verbose)
{
	set_default_values();

 	string path = file::combine_path(dir, filename);
	file::load_ascii_file(path, data);
	parse();
	data.clear();
}

analytic_gas_disk::~analytic_gas_disk()
{
}

void analytic_gas_disk::set_default_values()
{
	params.gas_decrease          = GAS_DENSITY_CONSTANT;
	params.t0                    = 0.0;
	params.t1                    = 0.0;
	params.e_folding_time        = 0.0;

	params.c_vth                 = 0.0;
	params.alpha                 = 0.0;
	params.mean_molecular_weight = 0.0;
	params.particle_diameter     = 0.0;

	params.eta.x = 0.0, params.eta.y = 0.0;
	params.rho.x = 0.0, params.rho.y = 0.0;
	params.sch.x = 0.0, params.sch.y = 0.0;
	params.tau.x = 0.0, params.tau.y = 0.0;

	params.mfp.x  = 0.0,  params.mfp.y = 0.0;
	params.temp.x = 0.0,  params.temp.y = 0.0;
}						  
						  
void analytic_gas_disk::parse()
{
	// instantiate Tokenizer classes
	Tokenizer data_tokenizer;
	Tokenizer line_tokenizer;
	string line;

	data_tokenizer.set(data, "\r\n");

	while ((line = data_tokenizer.next()) != "")
	{
		line_tokenizer.set(line, "=");
		string token;
		int tokenCounter = 1;

		string key; 
		string value;
		while ((token = line_tokenizer.next()) != "" && tokenCounter <= 2)
		{
			if (     1 == tokenCounter)
			{
				key = token;
			}
			else if (2 == tokenCounter)
			{
				value = token;
			}
			tokenCounter++;
		}
		if (2 < tokenCounter)
		{
			set_param(key, value);
		}
		else
		{
			throw string("Invalid key/value pair: " + line + ".");
		}
	}
}

void analytic_gas_disk::set_param(string& key, string& value)
{
	static char n_call = 0;

	n_call++;
	tools::trim(key);
	tools::trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "name")
	{
		name = value;
    } 
    else if (key == "description")
	{
		desc = value;
    }

	else if (key == "mean_molecular_weight" || key == "mmw")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.mean_molecular_weight = atof(value.c_str());
	}
	else if (key == "particle_diameter" || key == "diameter")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.particle_diameter = atof(value.c_str()) * constants::MeterToAu;
	}
	else if (key == "alpha")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.alpha = atof(value.c_str());
	}
	else if (key == "time_dependence")
	{
		if (     value == "constant" || value == "const")
		{
			params.gas_decrease = GAS_DENSITY_CONSTANT;
		}
		else if (value == "linear" || value == "lin")
		{
			params.gas_decrease = GAS_DENSITY_DECREASE_LINEAR;
		}
		else if (value == "exponential" || value == "exp")
		{
			params.gas_decrease = GAS_DENSITY_DECREASE_EXPONENTIAL;
		}
		else
		{
			throw string("Invalid value at: " + key);
		}
	}
	else if (key == "t0")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.t0 = atof(value.c_str()) * constants::YearToDay * constants::Gauss;
	}
	else if (key == "t1")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.t1 = atof(value.c_str()) * constants::YearToDay * constants::Gauss;
	}
	else if (key == "e_folding_time")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.e_folding_time = atof(value.c_str()) * constants::YearToDay * constants::Gauss;
	}
	else if (key == "eta_c")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.eta.x = atof(value.c_str());
	}
    else if (key == "eta_p")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.eta.y = atof(value.c_str());
	}
    else if (key == "rho_c")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.rho.x = atof(value.c_str()) * constants::GramPerCm3ToSolarPerAu3;
	}
    else if (key == "rho_p")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.rho.y = atof(value.c_str());
	}
    else if (key == "sch_c")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.sch.x = atof(value.c_str());
	}
    else if (key == "sch_p")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.sch.y = atof(value.c_str());
	}
    else if (key == "tau_c")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.tau.x = atof(value.c_str());
	}
    else if (key == "tau_p")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.tau.y = atof(value.c_str());
	}
	else
	{
		throw string("Invalid parameter :" + key + ".");
	}

	if (verbose)
	{
		if (n_call == 1)
		{
			cout << "The following gas disk parameters are setted:" << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << endl;
	}
}

void analytic_gas_disk::calc(var_t m_star)
{
	params.c_vth = sqrt((8.0 * constants::Boltzman_CMU)/(constants::Pi * params.mean_molecular_weight * constants::ProtonMass_CMU));

	params.mfp.x = params.mean_molecular_weight * constants::ProtonMass_CMU / (sqrt(2.0) * constants::Pi * SQR(params.particle_diameter) * params.rho.x);
	params.mfp.y = -params.rho.y;

	params.temp.x = SQR(params.sch.x) * constants::Gauss2 * m_star * params.mean_molecular_weight * constants::ProtonMass_CMU / constants::Boltzman_CMU;
	params.temp.y = 2.0 * params.sch.y - 3.0;
}

ostream& operator<<(ostream& stream, const analytic_gas_disk* g_disk)
{
	const char* gas_decrease_name[] = 
		{
			"GAS_DENSITY_CONSTANT",
			"GAS_DENSITY_DECREASE_LINEAR",
			"GAS_DENSITY_DECREASE_EXPONENTIAL"
		};

	stream << "name: " << g_disk->name << endl;
	stream << "desc: " << g_disk->desc << endl << endl;

	stream << "eta: " << g_disk->params.eta.x << ", " << g_disk->params.eta.y << endl;
	stream << "rho: " << g_disk->params.rho.x << ", " << g_disk->params.rho.y << endl;
	stream << "sch: " << g_disk->params.sch.x << ", " << g_disk->params.sch.y << endl;
	stream << "tau: " << g_disk->params.tau.x << ", " << g_disk->params.tau.y << endl << endl;
	stream << " mfp: " << g_disk->params.mfp.x << ", " << g_disk->params.mfp.y << endl;
	stream << "temp: " << g_disk->params.temp.x << ", " << g_disk->params.temp.y << endl << endl;

	stream << "  gas_decrease: " << gas_decrease_name[g_disk->params.gas_decrease] << endl;
	stream << "            t0: " << g_disk->params.t0 << " [d]" << endl;
	stream << "            t1: " << g_disk->params.t1 << " [d]" << endl;
	stream << "e_folding_time: " << g_disk->params.e_folding_time << " [d]" << endl << endl;

	stream << " c_vth: " << g_disk->params.c_vth << endl;
	stream << " alpha: " << g_disk->params.alpha << endl;
	stream << "mean_molecular_weight: " << g_disk->params.mean_molecular_weight << endl;
	stream << "    particle_diameter: " << g_disk->params.particle_diameter << endl;
		
	return stream;
}
