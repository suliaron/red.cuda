#pragma once
#include <string>

#include "red_type.h"

using namespace std;

namespace redutilcu
{
	namespace file 
	{
		string combine_path(const string& dir, const string& filename);
		string get_filename(const string& path);
		string get_filename_without_ext(const string& path);
		string get_directory(const string& path);
		string get_extension(const string& path);

		void load_ascii_file(const string& path, string& result);
		void load_binary_file(const string& path, size_t n_data, var_t* data);

		void Emese_data_format_to_red_cuda_format(const string& input_path, const string& output_path);

		void log_start(ostream& sout, int argc, const char** argv, const char** env, collision_detection_model_t cdm, bool print_to_screen);
		void log_message(ostream& sout, string msg, bool print_to_screen);

		void print_body_record(         ofstream &sout, string name, var_t epoch, param_t *p, body_metadata_t *body_md, vec_t *r, vec_t *v);
		void print_body_record_Emese(   ofstream &sout, string name, var_t epoch, param_t *p, body_metadata_t *body_md, vec_t *r, vec_t *v);
		void print_body_record_HIPERION(ofstream &sout, string name, var_t epoch, param_t *p, body_metadata_t *body_md, vec_t *r, vec_t *v);
		void print_oe_record(ofstream &sout, orbelem_t* oe);
		void print_oe_record(ofstream &sout, orbelem_t* oe, param_t *p);
		void print_oe_record(ofstream &sout, orbelem_t* oe, param_t *p, body_metadata_t *bmd);
		void print_oe_record(ofstream &sout, ttt_t epoch, orbelem_t* oe, param_t *p, body_metadata_t *bmd);
	} /* file */
} /* redutilcu */

