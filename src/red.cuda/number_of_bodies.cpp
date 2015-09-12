// includes system
#include <cstring>
#include <iomanip>
#include <iostream>

// includes project
#include "number_of_bodies.h"
#include "redutilcu.h"

using namespace std;

number_of_bodies::number_of_bodies(unsigned int n_s, unsigned int n_gp, unsigned int n_rp, unsigned int n_pp, unsigned int n_spl, unsigned int n_pl, unsigned int n_tp)
{
	initial[BODY_TYPE_STAR]              = n_s;
	initial[BODY_TYPE_GIANTPLANET]       = n_gp;
	initial[BODY_TYPE_ROCKYPLANET]       = n_rp;
	initial[BODY_TYPE_PROTOPLANET]       = n_pp;
	initial[BODY_TYPE_SUPERPLANETESIMAL] = n_spl;
	initial[BODY_TYPE_PLANETESIMAL]      = n_pl;
	initial[BODY_TYPE_TESTPARTICLE]      = n_tp;

	memcpy(playing, initial, sizeof(playing));

	memset(inactive, 0, sizeof(inactive));
	memset(removed,  0, sizeof(removed));

	n_removed = 0;

    sink.x   = sink.y   = 0;
    source.x = source.y = 0;
}

void number_of_bodies::update()
{
	n_removed = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		playing[i] -= inactive[i];
		removed[i] += inactive[i];
		n_removed  += inactive[i];
		inactive[i] = 0;
	}
}

unsigned int number_of_bodies::get_n_SI() 
{
	return (playing[BODY_TYPE_STAR] + playing[BODY_TYPE_GIANTPLANET] + playing[BODY_TYPE_ROCKYPLANET] + playing[BODY_TYPE_PROTOPLANET]);
}

unsigned int number_of_bodies::get_n_NSI()
{
	return (playing[BODY_TYPE_SUPERPLANETESIMAL] + playing[BODY_TYPE_PLANETESIMAL]);
}

unsigned int number_of_bodies::get_n_NI()
{
	return playing[BODY_TYPE_TESTPARTICLE];
}

unsigned int number_of_bodies::get_n_total_initial()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += initial[i];
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_playing()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += playing[i];
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_active()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += playing[i] - inactive[i];
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_inactive()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += inactive[i];
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_removed()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += removed[i];
	}
	return n; 
}

string number_of_bodies::get_n_playing()
{
	string result;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		result += redutilcu::number_to_string(playing[i]);
		if (BODY_TYPE_TESTPARTICLE > i)
		{
			result += '_';
		}
	}
	return result;
}

unsigned int number_of_bodies::get_n_GD()
{
	return (playing[BODY_TYPE_SUPERPLANETESIMAL] + playing[BODY_TYPE_PLANETESIMAL]);
}

unsigned int number_of_bodies::get_n_MT1()
{
	return (playing[BODY_TYPE_ROCKYPLANET] + playing[BODY_TYPE_PROTOPLANET]);
}

unsigned int number_of_bodies::get_n_MT2()
{
	return playing[BODY_TYPE_GIANTPLANET];
}

unsigned int number_of_bodies::get_n_massive()
{
	return (get_n_SI() + get_n_NSI());
}

unsigned int number_of_bodies::get_n_active_by(body_type_t type)
{
	return (playing[type] - inactive[type]);
}

interaction_bound number_of_bodies::get_bound_SI(unsigned int n_tpb)
{
	sink.x   = 0, sink.y   = sink.x + get_n_SI();
	source.x = 0, source.y = source.x + get_n_massive();

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NSI(unsigned int n_tpb)
{
	sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
	source.x = 0,		   source.y = source.x + get_n_SI();

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NI(unsigned int n_tpb)
{
	sink.x   = get_n_massive(), sink.y   = sink.x + get_n_NI();
	source.x = 0,   	        source.y = source.x + get_n_massive();

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_GD(unsigned int n_tpb)
{
	sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
	source.x = 0,		   source.y = 0;

	return interaction_bound(sink, source);
}

//interaction_bound number_of_bodies::get_bound_MT1()
//{
//	sink.x   = n_s + n_gp, sink.y   = sink.x + n_pp + n_rp;
//	source.x = 0,	       source.y = source.x + 0;
//
//	return interaction_bound(sink, source);
//}
//
//interaction_bound number_of_bodies::get_bound_MT2()
//{
//	sink.x   = n_s,   sink.y = sink.x + n_gp;
//	source.x = 0, 	source.y = source.x + 0;
//
//	return interaction_bound(sink, source);
//}

ostream& operator<<(ostream& stream, const number_of_bodies* n_bodies)
{
	stream << "Number of bodies:" << endl;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		stream << setw(20) << body_type_name[i] << ": " << setw(5) << n_bodies->playing[i] - n_bodies->inactive[i] << endl;
	}
		
	return stream;
}
