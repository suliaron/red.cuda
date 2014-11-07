// includes system
#include <iostream>
#include <sstream>
#include <cctype>
#include <ctime>
#include <string>

// includes project
#include "tools.h"

using namespace std;

namespace redutilcu
{
namespace tools
{
bool is_number(const string& str)
{
   for (size_t i = 0; i < str.length(); i++) {
	   if (std::isdigit(str[i]) || str[i] == 'e' || str[i] == 'E' || str[i] == '.' || str[i] == '-' || str[i] == '+')
           continue;
	   else
		   return false;
   }
   return true;
}

/// Removes all trailing white-space characters from the current std::string object.
void trim_right(string& str)
{
	// trim trailing spaces
	size_t endpos = str.find_last_not_of(" \t");
	if (string::npos != endpos ) {
		str = str.substr( 0, endpos+1 );
	}
}

/// Removes all trailing characters after the first # character
void trim_right(string& str, char c)
{
	// trim trailing spaces

	size_t endpos = str.find(c);
	if (string::npos != endpos ) {
		str = str.substr( 0, endpos);
	}
}

/// Removes all leading white-space characters from the current std::string object.
void trim_left(string& str)
{
	// trim leading spaces
	size_t startpos = str.find_first_not_of(" \t");
	if (string::npos != startpos ) {
		str = str.substr( startpos );
	}
}

/// Removes all leading and trailing white-space characters from the current std::string object.
void trim(string& str)
{
	trim_right(str);
	trim_left(str);
}

string get_time_stamp()
{
	static char time_stamp[20];
	time_t now = time(0);
	strftime(time_stamp, 20, "%Y-%m-%d %H:%M:%S", localtime(&now));

	return string(time_stamp);
}

string convert_time_t(time_t t)
{
	string result;

	ostringstream convert;	// stream used for the conversion
	convert << t;			// insert the textual representation of 't' in the characters in the stream
	result = convert.str();

	return result;
}

} /* tools */
} /* redutilcu */
