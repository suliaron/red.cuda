// includes system
#include <iostream>
#include <string>

using namespace std;

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