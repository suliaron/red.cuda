// includes system
#include <algorithm>
#include <cmath>
#include <iostream>

// includes project
#include "fargo_gas_disk.h"
#include "red_constants.h"
#include "redutilcu.h"
#include "tokenizer.h"
#include "red_macro.h"

using namespace redutilcu;

fargo_gas_disk::fargo_gas_disk(string& dir, string& filename, computing_device_t comp_dev, bool verbose) :
	dir(dir),
	filename(filename),
	comp_dev(comp_dev),
	verbose(verbose)
{
	initialize();

	// Load the parameters controling the FARGO run
 	string path = file::combine_path(dir, filename);
	file::load_ascii_file(path, data);
	parse();

	allocate_storage();

	// Load the first frame from the FARGO run (the results for t = t_0)
	path = file::combine_path(dir, "used_rad.dat");
	load_used_rad(path, params.n_rad);
	load(0.0);

	transform_data();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_to_device();
	}

	create_aliases();
}

fargo_gas_disk::~fargo_gas_disk()
{
	deallocate_device_storage();
	deallocate_host_storage();
}

void fargo_gas_disk::initialize()
{
	params.aspect_ratio      = 0.0;       // Thickness over Radius in the disc
	params.sigma_0           = 0.0;       // Surface Density at r=1
	params.alpha_viscosity   = 0.0;       // Uniform kinematic viscosity
	params.sigma_slope       = 0.0;       // Slope of surface density profile.
	params.flaring_index     = 0.0;
	params.exclude_hill      = true;

// Planet parameters
	params.thickness_smoothing = 0.0;     // Smoothing parameters in disk thickness

// Numerical method parameters
	params.omega_frame       = 0;

// Mesh parameters
	params.n_rad             = 0;         // Radial number of zones
	params.n_sec             = 0;         // Azimuthal number of zones (sectors)
	params.r_min             = 0.0;       // Inner boundary radius
	params.r_max             = 0.0;       // Outer boundary radius
												 
// Output control parameters					 
	params.n_tot             = 0;         // Total number of time steps
	params.n_interm          = 0;         // Time steps between outputs
	params.dT                = 0.0;       // Time step length. 2PI = 1 orbit

// Viscosity damping due to a dead zone
	params.visc_mod_r1       = 0.0;       // Inner radius of dead zone
	params.visc_mod_delta_r1 = 0.0;       // Width of viscosity transition at inner radius
	params.visc_mod_r2       = 0.0;       // Outer radius of dead zone
	params.visc_mod_delta_r2 = 0.0;       // Width of viscosity transition at outer radius
	params.visc_mod          = 0.0;       // Viscosity damp
}						  

void fargo_gas_disk::allocate_storage()
{
	const int n_cell = params.n_rad * params.n_sec;

	h_density.resize(1);
	h_vrad.resize(1);
	h_vtheta.resize(1);
	h_used_rad.resize(1);

	d_density.resize(1);
	d_vrad.resize(1);
	d_vtheta.resize(1);
	d_used_rad.resize(1);

	density.resize(1);
	vrad.resize(1);
	vtheta.resize(1);
	used_rad.resize(1);

	allocate_host_storage(n_cell);
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(n_cell);
	}
}

void fargo_gas_disk::allocate_host_storage(int n_cell)
{
	ALLOCATE_HOST_VECTOR((void**)&(h_density[0]), n_cell * sizeof(var_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_vrad[0]),    n_cell * sizeof(var_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_vtheta[0]),  n_cell * sizeof(var_t));

	ALLOCATE_HOST_VECTOR((void**)&(h_used_rad[0]), params.n_rad * sizeof(var_t));
}

void fargo_gas_disk::allocate_device_storage(int n_cell)
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_density[0]), n_cell * sizeof(var_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_vrad[0]),    n_cell * sizeof(var_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_vtheta[0]),  n_cell * sizeof(var_t));

	ALLOCATE_DEVICE_VECTOR((void**)&(d_used_rad[0]), params.n_rad * sizeof(var_t));
}

void fargo_gas_disk::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_density[0]));
	FREE_HOST_VECTOR((void **)&(h_vrad[0]));
	FREE_HOST_VECTOR((void **)&(h_vtheta[0]));
	FREE_HOST_VECTOR((void **)&(h_used_rad[0]));
}

void fargo_gas_disk::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(d_density[0]));
	FREE_DEVICE_VECTOR((void **)&(d_vrad[0]));
	FREE_DEVICE_VECTOR((void **)&(d_vtheta[0]));
	FREE_DEVICE_VECTOR((void **)&(d_used_rad[0]));
}

void fargo_gas_disk::create_aliases()
{
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		density[0] = h_density[0];
		vrad[0]    = h_vrad[0];
		vtheta[0]  = h_vtheta[0];
		used_rad[0]= h_used_rad[0];
		break;
	case COMPUTING_DEVICE_GPU:
		density[0] = d_density[0];
		vrad[0]    = d_vrad[0];
		vtheta[0]  = d_vtheta[0];
		used_rad[0]= d_used_rad[0];
		break;
	default:
		throw string("Parameter 'comp_dev' is out of range.");
	}
}

void fargo_gas_disk::copy_to_device()
{
	const int n_cell = params.n_rad * params.n_sec;

	copy_vector_to_device((void *)d_density[0],	 (void *)h_density[0],  n_cell * sizeof(var_t));
	copy_vector_to_device((void *)d_vrad[0],	 (void *)h_density[0],  n_cell * sizeof(var_t));
	copy_vector_to_device((void *)d_vtheta[0],	 (void *)h_density[0],  n_cell * sizeof(var_t));

	copy_vector_to_device((void *)d_used_rad[0], (void *)h_used_rad[0], params.n_rad * sizeof(var_t));
}

// TODO: implement
int fargo_gas_disk::create_index_for_filename(ttt_t t)
{
	int result = 0;
	/*
	 *  use the
	 *  int n_interm;              // Time steps between outputs
	 *  var_t dT;                  // Time step length. 2PI = 1 orbit
	 *  parameters to compute the new index based on the value of t .
	 */

	return result;
}

void fargo_gas_disk::load(ttt_t t)
{
	const size_t n = params.n_rad * params.n_sec;

	string filename = "gasdens" + number_to_string(create_index_for_filename(t)) + ".dat";
	string path = file::combine_path(dir, filename);
	load_gas_density(path, n);

	filename = "gasvrad" + number_to_string(create_index_for_filename(t)) + ".dat";
	path = file::combine_path(dir, filename);
	load_gas_vrad(path, n);

	filename = "gasvtheta" + number_to_string(create_index_for_filename(t)) + ".dat";
	path = file::combine_path(dir, filename);
	load_gas_vtheta(path, n);
}

void fargo_gas_disk::load_gas_density(string& path, size_t n)
{
	var_t* p = h_density[0];
	file::load_binary_file(path, n, p);
}

void fargo_gas_disk::load_gas_vrad(string& path, size_t n)
{
	var_t* p = h_vrad[0];
	file::load_binary_file(path, n, p);
}

void fargo_gas_disk::load_gas_vtheta(string& path, size_t n)
{
	var_t* p = h_vtheta[0];
	file::load_binary_file(path, n, p);
}

void fargo_gas_disk::load_used_rad(string& path, size_t)
{
	string result;
	file::load_ascii_file(path, result);

	int m = 0;
	for (unsigned int i = 0; i < result.length(); i++)
	{
		string num;
		num.resize(30);
		int k = 0;
		while (i < result.length() && k < 30 && result[i] != '\n')
		{
			num[k] = result[i];
			k++;
			i++;
		}
		num.resize(k);
		if (!tools::is_number(num))
		{
			throw string("Invalid number (" + num + ") in file '" + path + "'!\n");
		}
		h_used_rad[0][m] = atof(num.c_str());   // [AU]
		m++;
	}
}

void fargo_gas_disk::transform_data()
{
	transform_time();
	transform_velocity();
	transform_density();
}

void fargo_gas_disk::transform_time()
{
	throw string("fargo_gas_disk::transform_time() is not yet implemented.");
	// TODO
	//params.dT *= 1.0;
}

void fargo_gas_disk::transform_velocity()
{
	throw string("fargo_gas_disk::transform_velocity() is not yet implemented.");
}

void fargo_gas_disk::transform_density()
{
	throw string("fargo_gas_disk::transform_density() is not yet implemented.");
}

vec_t fargo_gas_disk::get_velocity(vec_t rVec)
{
	vec_t v_gas = {0.0, 0.0, 0.0, 0.0};

	return v_gas;
}

var_t fargo_gas_disk::get_density(vec_t r)
{
	var_t result = 0.0;

	return result;
}

void fargo_gas_disk::parse()
{
	// instantiate Tokenizer classes
	Tokenizer data_tokenizer;
	Tokenizer line_tokenizer;
	string line;

	data_tokenizer.set(data, "\r\n");

	while ((line = data_tokenizer.next()) != "")
	{
		line_tokenizer.set(line, " \t");
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

void fargo_gas_disk::set_param(string& key, string& value)
{
	static char n_call = 0;

	n_call++;
	tools::trim(key);
	tools::trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "aspectratio")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.aspect_ratio = atof(value.c_str());
    } 
    else if (key == "sigma0")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.sigma_0 = atof(value.c_str());
    }
    else if (key == "alphaviscosity")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.alpha_viscosity = atof(value.c_str());
    }
    else if (key == "sigmaslope")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.sigma_slope = atof(value.c_str());
    }
    else if (key == "flaringindex")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.flaring_index = atof(value.c_str());
    }
    else if (key == "excludehill")
	{
		exclude_hill = value;
		params.exclude_hill = (value == "YES" ? true : false);
    }
    else if (key == "planetconfig")
	{
		planet_config = value;
    }
    else if (key == "thicknesssmoothing")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.thickness_smoothing = atof(value.c_str());
    }
    else if (key == "transport")
	{
		transport = value;
    }
    else if (key == "innerboundary")
	{
		inner_boundary = value;
    }
    else if (key == "outerboundary")
	{
		outer_boundary = value;
    }
    else if (key == "disk")
	{
		disk = value;
    }
    else if (key == "omegaframe")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.omega_frame = atof(value.c_str());
    }
    else if (key == "frame")
	{
		frame = value;
    }
    else if (key == "indirectterm")
	{
		indirect_term = value;
    }
    else if (key == "nrad")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.n_rad = atoi(value.c_str());
    }
    else if (key == "nsec")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.n_sec = atoi(value.c_str());
    }
    else if (key == "rmin")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.r_min = atof(value.c_str());
    }
    else if (key == "rmax")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.r_max = atof(value.c_str());
    }
    else if (key == "radialspacing")
	{
		radial_spacing = value;
    }
    else if (key == "ntot")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.n_tot = atoi(value.c_str());
    }
    else if (key == "ninterm")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.n_interm = atoi(value.c_str());
    }
    else if (key == "dt")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.dT = atof(value.c_str());
    }
    else if (key == "outputdir")
	{
		output_dir = value;
    }
    else if (key == "viscmodr1")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.visc_mod_r1 = atof(value.c_str());
    }
    else if (key == "viscmoddeltar1")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.visc_mod_delta_r1 = atof(value.c_str());
    }
    else if (key == "viscmodr2")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.visc_mod_r2 = atof(value.c_str());
    }
    else if (key == "viscmoddeltar2")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.visc_mod_delta_r2 = atof(value.c_str());
    }
    else if (key == "viscmod")
	{
		if (!tools::is_number(value))
		{
			throw string("Invalid number at: " + key);
		}
		params.visc_mod = atof(value.c_str());
    }
	else if (key == "damprmin")
	{
		//cout << "The following gas disk parameters was skipped:" << endl;
		//cout << "\t'" << key << "' '" << value << "'" << endl;
	}
	else if (key == "damprmax")
	{
		//cout << "The following gas disk parameters was skipped:" << endl;
		//cout << "\t'" << key << "' '" << value << "'" << endl;
	}
	else
	{
		throw string("Invalid parameter :" + key + ".");
	}

	if (verbose)
	{
		if (1 == n_call)
		{
			cout << "The following gas disk parameters are setted:" << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << endl;
	}
}
