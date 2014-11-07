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
		void log_start_cmd(ostream& sout, int argc, const char** argv, const char** env);
		void log_rebuild_vectors(ostream& sout, ttt_t t);
		void log_message(ostream& sout, string msg);
	} /* file */
} /* redutilcu */

