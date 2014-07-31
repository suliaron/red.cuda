#pragma once
#include <string>

using namespace std;

namespace redutilcu
{
	namespace file 
	{
		string combine_path(string dir, string filename);
		string get_filename(const string& path);
		string get_filename_without_ext(const string& path);
		string get_directory(const string& path);
		string get_extension(const string& path);
		void load_ascii_file(string& path, string& result);
	} /* file */
} /* redutilcu */

