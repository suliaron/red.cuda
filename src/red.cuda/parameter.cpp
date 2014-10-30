// includes system
#include <algorithm>
#include <iostream>

// includes project
#include "red_constants.h"
#include "parameter.h"
#include "redutilcu.h"
#include "tokenizer.h"

using namespace redutilcu;

parameter::parameter(string& dir, string& filename, bool verbose) :
	filename(filename),
	verbose(verbose),
	adaptive(false),
	error_check_for_tp(false),
	fr_cntr(FRAME_CENTER_BARY),
	int_type(INTEGRATOR_EULER),
	tolerance(1.0e-10),
	start_time(0.0),
	simulation_length(0.0),
	output_interval(0.0),
	threshold()			/* syntax for zeroing the array */
{
	string path = file::combine_path(dir, filename);
	file::load_ascii_file(path, data);
	parse();
	transform_time();
	data.clear();

	stop_time = start_time + simulation_length;
}

parameter::~parameter() 
{
}

void parameter::parse()
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

void parameter::set_param(string& key, string& value)
{
	static char n_call = 0;

	n_call++;
	tools::trim(key);
	tools::trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "name") {
		simulation_name = value;
    } 
    else if (key == "description") {
		simulation_desc = value;
    }
    else if (key == "frame_center") {
		transform(value.begin(), value.end(), value.begin(), ::tolower);
		if (     value == "bary") {
			fr_cntr = FRAME_CENTER_BARY;
		}
		else if (value == "astro") {
			fr_cntr = FRAME_CENTER_ASTRO;
		}
		else {
			throw string("Invalid frame center type: " + value);
		}
    }
    else if (key == "integrator") {
		transform(value.begin(), value.end(), value.begin(), ::tolower);
		if (value == "e" || value == "euler") {
			int_type = INTEGRATOR_EULER;
		}
		else if (value == "rk2" || value == "rungekutta2")	{
			int_type = INTEGRATOR_RUNGEKUTTA2;
		}
		else if (value == "rk4" || value == "rungekutta4")	{
			int_type = INTEGRATOR_RUNGEKUTTA4;
		}
		else if (value == "rkf8" || value == "rungekuttafehlberg8")	{
			int_type = INTEGRATOR_RUNGEKUTTAFEHLBERG78;
		}			
		else if (value == "rkn" || value == "rungekuttanystrom") {
			int_type = INTEGRATOR_RUNGEKUTTANYSTROM;
		}
		else {
			throw string("Invalid integrator type: " + value);
		}
	}
    else if (key == "tolerance") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		adaptive = true;
		tolerance = atof(value.c_str());
	}
	else if (key == "error_check_for_tp") {
		if      (value == "true")
		{
			error_check_for_tp = true;
		}
		else if (value == "false")
		{
			error_check_for_tp = false;
		}
		else {
			throw string("Invalid value at: " + key);
		}
	}

    else if (key == "start_time") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		start_time = atof(value.c_str()) * constants::YearToDay;
	}
    else if (key == "length") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		simulation_length = atof(value.c_str()) * constants::YearToDay;
	}
    else if (key == "output_interval") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		output_interval = atof(value.c_str()) * constants::YearToDay;
	}
    else if (key == "ejection") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		threshold[THRESHOLD_EJECTION_DISTANCE] = atof(value.c_str());
	}
    else if (key == "hit_centrum") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]  = atof(value.c_str());
	}
    else if (key == "collision_factor") {
		if (!tools::is_number(value)) {
			throw string("Invalid number at: " + key);
		}
		threshold[THRESHOLD_COLLISION_FACTOR]  = atof(value.c_str());
	}
	else {
		throw string("Invalid parameter :" + key + ".");
	}

	if (verbose) {
		if (n_call == 1) {
			cout << "The following common parameters are setted:" << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << endl;
	}
}

void parameter::transform_time()
{
	start_time			*= constants::Gauss;
	simulation_length	*= constants::Gauss;
	output_interval		*= constants::Gauss;
}

ostream& operator<<(ostream& stream, const parameter* p)
{
	const char* integrator_name[] = 
		{
			"INTEGRATOR_EULER"
			"INTEGRATOR_RUNGEKUTTA2",
			"INTEGRATOR_RUNGEKUTTA4",
			"INTEGRATOR_RUNGEKUTTAFEHLBERG78",
			"INTEGRATOR_RUNGEKUTTANYSTROM",
		};
	const char* frame_center_name[] = 
		{
			"FRAME_CENTER_BARY",
			"FRAME_CENTER_ASTRO"
		};

	const char* threshold_name[] = 
		{
			"THRESHOLD_HIT_CENTRUM_DISTANCE",
			"THRESHOLD_EJECTION_DISTANCE",
			"THRESHOLD_COLLISION_FACTOR",
		};

	stream << "simulation name: " << p->simulation_name << endl;
	stream << "simulation description: " << p->simulation_desc << endl;
	stream << "simulation frame center: " << frame_center_name[p->fr_cntr] << endl;
	stream << "simulation integrator: " << integrator_name[p->int_type] << endl;
	stream << "simulation tolerance: " << p->tolerance << endl;
	stream << "simulation adaptive: " << (p->adaptive ? "true" : "false") << endl;
	stream << "simulation start time: " << p->start_time << endl;
	stream << "simulation length: " << p->simulation_length << endl;
	stream << "simulation output interval: " << p->output_interval << endl;
	for (int i = 0; i < THRESHOLD_N; i++) {
		stream << "simulation threshold[" << threshold_name[i] << "]: " << p->threshold[i] << endl;
	}

	return stream;
}
