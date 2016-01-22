#pragma once
#include <string>

#include "red_type.h"

namespace redutilcu
{
	namespace file 
	{
		string combine_path(const std::string& dir, const std::string& filename);
		string get_filename(const std::string& path);
		string get_filename_without_ext(const std::string& path);
		string get_directory(const std::string& path);
		string get_extension(const std::string& path);

		void load_ascii_file(const std::string& path, std::string& result);
		void load_binary_file(const std::string& path, size_t n_data, var_t* data);

		void Emese_data_format_to_red_cuda_format(const std::string& input_path, const std::string& output_path);

		void log_start(ostream& sout, int argc, const char** argv, const char** env, std::string& params, bool print_to_screen);
		void log_message(ostream& sout, std::string msg, bool print_to_screen);

		void print_body_record(         ofstream &sout, std::string name, var_t epoch, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v);
		void print_body_record_Emese(   ofstream &sout, std::string name, var_t epoch, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v);
		void print_body_record_HIPERION(ofstream &sout, std::string name, var_t epoch, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v);
		void print_oe_record(ofstream &sout, orbelem_t* oe);
		void print_oe_record(ofstream &sout, orbelem_t* oe, pp_disk_t::param_t *p);
		void print_oe_record(ofstream &sout, orbelem_t* oe, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd);
		void print_oe_record(ofstream &sout, ttt_t epoch, orbelem_t* oe, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd);
	} /* file */
} /* redutilcu */
