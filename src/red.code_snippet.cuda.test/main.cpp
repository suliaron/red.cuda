// includes system
#include <iostream>
#include <string>

// includes project
#include "red_type.h"

using namespace std;

int main(int argc, const char** argv)
{
	
}

# if 0
/*
 *  Basic example of an exception
 */
int main(int argc, const char** argv)
{
	try
	{
		throw string("This is an exception!\n");
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}
}
#endif