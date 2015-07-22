#pragma once
// includes system
#include <string>
#include <vector>

// includes project
#include "red_type.h"

using namespace std;

class fargo_gas_disk
{
public:
	fargo_gas_disk(string& dir, string& filename, computing_device_t comp_dev, bool verbose);
	~fargo_gas_disk();

	void allocate_storage();
	void allocate_host_storage(int n_cell);
	void allocate_device_storage(int n_cell);

	void deallocate_host_storage();
	void deallocate_device_storage();

	void create_aliases();
	void copy_to_device();

	void load(ttt_t);
	void load_gas_density(string& path, size_t n);
	void load_gas_vrad(string& path, size_t n);
	void load_gas_vtheta(string& path, size_t n);
	void load_used_rad(string& path, size_t);

	vec_t get_velocity(vec_t r);
	var_t get_density(vec_t r);

	fargo_gas_disk_params_t params;

	vector<var_t*> h_density;    //!< Gas density in the HOST memory
	vector<var_t*> d_density;    //!< Gas density in the DEVICE memory
	vector<var_t*> density;      //!< Alias to the gas density (either in the HOST or the DEVICE memory)

	vector<var_t*> h_vrad;       //!< Gas radial velocity in the HOST memory
	vector<var_t*> d_vrad;       //!< Gas radial velocity in the DEVICE memory
	vector<var_t*> vrad;         //!< Alias to the gas radial velocity (either in the HOST or the DEVICE memory)

	vector<var_t*> h_vtheta;     //!< Gas azimuthal velocity in the HOST memory
	vector<var_t*> d_vtheta;     //!< Gas azimuthal velocity in the DEVICE memory
	vector<var_t*> vtheta;       //!< Alias to the gas azimuthal velocity (either in the HOST or the DEVICE memory)

	vector<var_t*> h_used_rad;
	vector<var_t*> d_used_rad;
	vector<var_t*> used_rad;

private:
	void initialize();
	void parse();
	void set_param(string& key, string& value);
	int create_index_for_filename(ttt_t t);

	void transform_data();
	void transform_time();
	void transform_velocity();
	void transform_density();

	bool verbose;
	string dir;
	string filename;
	string data;                 //!< holds a copy of the file containing the parameters of the simulation

	string exclude_hill;
	string planet_config;
	string transport;
	string inner_boundary;		 // choose : OPEN or RIGID or NONREFLECTING
	string outer_boundary;
	string disk;
	string frame;
	string indirect_term;
	string radial_spacing;       // Zone interfaces evenly spaced
	string output_dir;

	computing_device_t comp_dev;
};
