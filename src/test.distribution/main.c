//includes system
//#include <algorithm>
//#include <cstdlib>
#include <cmath>
//#include <fstream>
//#include <iostream>
//#include <iomanip>
#include <string>
//#include <vector>

//includes project
#include "distribution.h"

using namespace std;

//void print_data(ostream& sout, vector<var_t>& data)
//{
//	static int var_t_w = 17;
//
//	sout.precision(8);
//	sout.setf(ios::right);
//	sout.setf(ios::scientific);
//
//	for (int i = 0; i < data.size(); i++)
//	{
//		sout << setw(var_t_w) << data[i] << endl;
//	}
//
//	sout.flush();
//}

int main()
{
	const int N = 10000;
	string dir = "C:\\Work\\Projects\\red.cuda\\TestRandomGenerator\\";

	//string path = dir + "D_rayleigh_1.txt";
	//try
	//{
	//	ostream* data_f = new ofstream(path.c_str(), ios::out);
	//	vector<var_t>* data = new vector<var_t>[N];

	//	rayleigh_distribution rd(1, 1.0);

	//	for (int i = 0; i < N; i++)
	//	{
	//		data->push_back(rd.get_next());
	//	}
	//	print_data(*data_f, *data);

	//}
	//catch (const exception ex)
	//{
	//	cerr << "Error: " << ex.what() << endl;
	//}

	return 0;
}