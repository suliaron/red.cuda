// includes system 
#include <fstream>

// includes project
#include "file_util.h"
#include "tools.h"

namespace redutilcu_file
{
string combine_path(string dir, string filename)
{
	if (dir.size() > 0) {
		if (*(dir.end() - 1) != '/' && *(dir.end() - 1) != '\\') {
			return dir + '/' + filename;
		}
		else {
			return dir + filename;
		}
	}
	else {
		return filename;
	}
}

string get_filename(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(pos + 1);
	}

	return result;
}

string get_filename_without_ext(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(pos + 1);
		pos = result.find_last_of('.');
		result = result.substr(0, pos);
	}

	return result;
}

string get_directory(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(0, pos);
	}

	return result;
}

string get_extension(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of('.');
		result = path.substr(pos + 1);
	}

	return result;
}

void load_ascii_file(string& path, string& result)
{
	std::ifstream file(path);
	if (file) {
		string str;
		while (std::getline(file, str))
		{
			// ignore zero length lines
			if (str.length() == 0)
				continue;
			// ignore comment lines
			if (str[0] == '#')
				continue;
			// delete comment after the value
			redutilcu_tools::trim_right(str, '#');
			result += str;
			result.push_back('\n');
		} 	
	}
	else {
		throw new std::string("The file '" + path + "' could not opened!\r\n");
	}
	file.close();
}

} /* redutilcu_file */
