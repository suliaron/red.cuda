#pragma once
#include <string>

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
	} /* file */
} /* redutilcu */

