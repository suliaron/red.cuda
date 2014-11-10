// includes system
#include <iomanip>
#include <fstream>

// includes project
#include "file_util.h"
#include "tools.h"
#include "red_macro.h"

namespace redutilcu
{
namespace file
{
string combine_path(const string& dir, const string& filename)
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

void load_ascii_file(const string& path, string& result)
{
	std::ifstream file(path.c_str());
	if (file) {
		string str;
		while (getline(file, str))
		{
			// ignore zero length lines
			if (str.length() == 0)
				continue;
			// ignore comment lines
			if (str[0] == '#')
				continue;
			// delete comment after the value
			tools::trim_right(str, '#');
			result += str;
			result.push_back('\n');
		} 	
	}
	else {
		throw string("The file '" + path + "' could not opened!\r\n");
	}
	file.close();
}

void log_start_cmd(ostream& sout, int argc, const char** argv, const char** env)
{
	sout << tools::get_time_stamp() << " starting " << argv[0] << endl;
	sout << "Command line arguments: " << endl;
	for (int i = 1; i < argc; i++)
	{
		sout << argv[i] << SEP;
	}
	sout << endl << endl;

	while (*env)
	{
		string s = *env;
#ifdef __GNUC__
		// TODO
#else
		if(      s.find("COMPUTERNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("OS=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("USERNAME=") < s.length())
		{
			sout << s << endl;
		}
		env++;
#endif
	}
	sout << endl;
}

void log_rebuild_vectors(ostream& sout, ttt_t t)
{
	sout << tools::get_time_stamp() << " Rebuild the vectors and remove inactive bodies " << " t: " << scientific << std::setprecision(16) << t << endl;
}

void log_message(ostream& sout, string msg)
{
	sout << tools::get_time_stamp() << msg << endl;
}

} /* file */
} /* redutilcu */
