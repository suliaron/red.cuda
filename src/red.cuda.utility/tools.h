#pragma once
#include <string>

using namespace std;

namespace redutilcu
{
	namespace tools
	{
		bool is_number(const string& str);
		void trim_right(string& str);
		void trim_right(string& str, char c);
		void trim_left(string& str);
		void trim(string& str);
		string get_time_stamp();
		string convert_time_t(time_t t);
	} /* tools */
} /* redutilcu_tools */
