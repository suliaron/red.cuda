#pragma once

#include <string>
#include <vector>

#include "red_type.h"

class fargo_gas_disk
{
public:
	fargo_gas_disk(std::string& dir, std::string& filename, computing_device_t comp_dev, bool verbose);
	~fargo_gas_disk();

	void allocate_storage();
	void allocate_host_storage(int n_cell);
	void allocate_device_storage(int n_cell);

	void deallocate_host_storage();
	void deallocate_device_storage();

	void create_aliases();
	void copy_to_device();

	void load(ttt_t);
	void load_gas_density(std::string& path, size_t n);
	void load_gas_vrad(std::string& path, size_t n);
	void load_gas_vtheta(std::string& path, size_t n);
	void load_used_rad(std::string& path, size_t);

	var4_t get_velocity(var4_t r);
	var_t get_density(var4_t r);

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
	void set_param(std::string& key, std::string& value);
	int create_index_for_filename(ttt_t t);

	void transform_data();
	void transform_time();
	void transform_velocity();
	void transform_density();

	bool verbose;
	std::string dir;
	std::string filename;
	std::string data;                 //!< holds a copy of the file containing the parameters of the simulation

	std::string exclude_hill;
	std::string planet_config;
	std::string transport;
	std::string inner_boundary;		 // choose : OPEN or RIGID or NONREFLECTING
	std::string outer_boundary;
	std::string disk;
	std::string frame;
	std::string indirect_term;
	std::string radial_spacing;       // Zone interfaces evenly spaced
	std::string output_dir;

	computing_device_t comp_dev;
};
