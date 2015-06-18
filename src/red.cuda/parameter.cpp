// includes system
#include <algorithm>
#include <cstring>
#include <iostream>

// includes project
#include "red_constants.h"
#include "parameter.h"
#include "redutilcu.h"
#include "tokenizer.h"

using namespace redutilcu;

parameter::parameter(string& dir, string& filename, bool verbose) :
	filename(filename),
	verbose(verbose)
{
	create_default();

	string path;
	if (0 < file::get_directory(filename).length())
	{
		path = filename;
	}
	else
	{
		path = file::combine_path(dir, filename);
	}
	file::load_ascii_file(path, data);
	parse();
	transform_time();
	stop_time = start_time + simulation_length;
}

parameter::~parameter() 
{
}

void parameter::create_default()
{
	adaptive           = true;
	error_check_for_tp = false;

	int_type           = INTEGRATOR_RUNGEKUTTAFEHLBERG78;
	tolerance          = 1.0e-10;

	start_time         = 0.0;		// [day]
	simulation_length  = 0.0;		// [day]
	output_interval    = 0.0;		// [day]

	memset(thrshld, 0, sizeof(thrshld));
}

void parameter::parse()
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
			if (1 == tokenCounter)
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

void parameter::set_param(string& key, string& value)
{
	static char n_call = 0;

	n_call++;
	tools::trim(key);
	tools::trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "name")
	{
		simulation_name = value;
    } 
    else if (key == "description")
	{
		simulation_desc = value;
    }
    else if (key == "integrator")
	{
		transform(value.begin(), value.end(), value.begin(), ::tolower);
		if (     value == "e"    || value == "euler")
		{
			int_type = INTEGRATOR_EULER;
		}
		else if (value == "rk2"  || value == "rungekutta2")
		{
			int_type = INTEGRATOR_RUNGEKUTTA2;
		}
		else if (value == "rk4"  || value == "rungekutta4")
		{
			int_type = INTEGRATOR_RUNGEKUTTA4;
		}
		else if (value == "rkf8" || value == "rungekuttafehlberg8")
		{
			int_type = INTEGRATOR_RUNGEKUTTAFEHLBERG78;
		}			
		else if (value == "rkn"  || value == "rungekuttanystrom")
		{
			int_type = INTEGRATOR_RUNGEKUTTANYSTROM;
		}
		else
		{
			throw string("Invalid integrator type: " + value);
		}
	}
    else if (key == "tolerance")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		adaptive = true;
		tolerance = atof(value.c_str());
	}
	else if (key == "error_check_for_tp")
	{
		if      (value == "true")
		{
			error_check_for_tp = true;
		}
		else if (value == "false")
		{
			error_check_for_tp = false;
		}
		else
		{
			throw string("Invalid value at: " + key);
		}
	}
    else if (key == "start_time")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		start_time = atof(value.c_str()) * constants::YearToDay;
	}
    else if (key == "length")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		simulation_length = atof(value.c_str()) * constants::YearToDay;
	}
    else if (key == "output_interval")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		output_interval = atof(value.c_str()) * constants::YearToDay;
	}
    else if (key == "ejection")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		thrshld[THRESHOLD_EJECTION_DISTANCE] = atof(value.c_str());
		thrshld[THRESHOLD_EJECTION_DISTANCE_SQUARED] = SQR(thrshld[THRESHOLD_EJECTION_DISTANCE]);
	}
    else if (key == "hit_centrum")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE] = atof(value.c_str());
		thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] = SQR(thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	}
    else if (key == "radii_enhance_factor" || key == "collision_factor")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		thrshld[THRESHOLD_RADII_ENHANCE_FACTOR] = atof(value.c_str());
	}
	else
	{
		throw string("Invalid parameter: " + key + ".");
	}

	if (verbose)
	{
		if (n_call == 1)
		{
			cout << "The following parameters are setted (from file :'" << filename << "'): " << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << endl;
	}
}

void parameter::transform_time()
{
	start_time        *= constants::Gauss;    // [day * k]
	simulation_length *= constants::Gauss;    // [day * k]
	output_interval	  *= constants::Gauss;    // [day * k]
}

ostream& operator<<(ostream& stream, const parameter* p)
{
	const char* integrator_name[] = 
		{
			"INTEGRATOR_EULER"
			"INTEGRATOR_RUNGEKUTTA2",
			"INTEGRATOR_RUNGEKUTTA4",
			"INTEGRATOR_RUNGEKUTTAFEHLBERG78",
			"INTEGRATOR_RUNGEKUTTANYSTROM"
		};

	const char* threshold_name[] = 
		{
			"THRESHOLD_HIT_CENTRUM_DISTANCE",
			"THRESHOLD_EJECTION_DISTANCE",
			"THRESHOLD_RADII_ENHANCE_FACTOR",
			"THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED",
			"THRESHOLD_EJECTION_DISTANCE_SQUARED"
		};

	stream << "simulation name: " << p->simulation_name << endl;
	stream << "simulation description: " << p->simulation_desc << endl;
	stream << "simulation frame center: barycentric" << endl;
	stream << "simulation integrator: " << integrator_name[p->int_type] << endl;
	stream << "simulation tolerance: " << p->tolerance << endl;
	stream << "simulation adaptive: " << (p->adaptive ? "true" : "false") << endl;
	stream << "simulation start time: " << p->start_time << endl;
	stream << "simulation length: " << p->simulation_length << endl;
	stream << "simulation output interval: " << p->output_interval << endl;
	for (int i = 0; i < THRESHOLD_N; i++)
	{
		stream << "simulation threshold[" << threshold_name[i] << "]: " << p->thrshld[i] << endl;
	}

	return stream;
}
