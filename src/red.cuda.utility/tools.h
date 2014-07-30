#pragma once
#include <string>

using namespace std;

namespace redutilcu_tools
{
	bool is_number(const string& str);
	void trim_right(string& str);
	void trim_right(string& str, char c);
	void trim_left(string& str);
	void trim(string& str);
	string get_time_stamp();
} /* redutilcu_tools */
