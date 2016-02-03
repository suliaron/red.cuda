#pragma once

#include "ode.h"

#include "red_type.h"

class tbp1D : public ode
{
public:
	tbp1D(uint16_t n_ppo, computing_device_t comp_dev);
	~tbp1D();

	void load(std::string& path);
	void load_ascii(ifstream& input);
	void load_ascii_record(ifstream& input, ttt_t* t, tbp1D_t::metadata_t *md, tbp1D_t::param_t* p, var_t* x, var_t* vx);
	void load_binary(ifstream& input);

	//! Print the data of all bodies
	/*!
		\param sout   print the data to this stream
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_result(ofstream** sout, data_rep_t repres);
	//! Print the data of all bodies in text format
	/*   
		\param sout print the data to this stream
	*/
	void print_result_ascii(ofstream& sout);
	//! Print the data of all bodies in binary format
	/*!
		\param sout print the data to this stream
	*/
	void print_result_binary(ofstream& sout);

	//! Print the energy integral
	/*!
		\param sout print the data to this stream
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_integral_data(ofstream& sout, data_rep_t repres);
	//! Print the energy integral in text format
	/*   
		\param sout print the data to this stream
	*/
	void print_integral_data_ascii(ofstream& sout);
	//! Print the energy integral in binary format
	/*!
		\param sout print the data to this stream
	*/
	void print_integral_data_binary(ofstream& sout);

	void calc_energy();
	void calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);

//private:
	void initialize();
	void allocate_storage();
	void allocate_host_storage();
	void allocate_device_storage();
	
	void deallocate_storage();
	void deallocate_host_storage();
	void deallocate_device_storage();

	void cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);
	void gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);

	var_t h;               //! Energy of the system

	tbp1D_t::metadata_t* h_md;
	tbp1D_t::metadata_t* d_md;
	tbp1D_t::metadata_t* md;

	var_t* h_epoch;
	var_t* d_epoch;
	var_t* epoch;
};
