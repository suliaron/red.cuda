// includes system
#include <iomanip>
#include <iostream>

// includes project
#include "number_of_bodies.h"

using namespace std;

number_of_bodies::number_of_bodies(int n_s, int n_gp, int n_rp, int n_pp, int n_spl, int n_pl, int n_tp, int n_tpb, bool ups) :
	n_tpb(n_tpb),
	ups(ups)
{
	playing[BODY_TYPE_STAR]              = initial[BODY_TYPE_STAR]              = n_s;
	playing[BODY_TYPE_GIANTPLANET]       = initial[BODY_TYPE_GIANTPLANET]       = n_gp;
	playing[BODY_TYPE_ROCKYPLANET]       = initial[BODY_TYPE_ROCKYPLANET]       = n_rp;
	playing[BODY_TYPE_PROTOPLANET]       = initial[BODY_TYPE_PROTOPLANET]       = n_pp;
	playing[BODY_TYPE_SUPERPLANETESIMAL] = initial[BODY_TYPE_SUPERPLANETESIMAL] = n_spl;
	playing[BODY_TYPE_PLANETESIMAL]      = initial[BODY_TYPE_PLANETESIMAL]      = n_pl;
	playing[BODY_TYPE_TESTPARTICLE]      = initial[BODY_TYPE_TESTPARTICLE]      = n_tp;
	playing[BODY_TYPE_PADDINGPARTICLE]   = initial[BODY_TYPE_PADDINGPARTICLE]   = 0;

	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		inactive[i] = 0;
		removed[i]  = 0;
	}

    sink.x   = sink.y   = 0;
    source.x = source.y = 0;
}

void number_of_bodies::update()
{
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		if (BODY_TYPE_PADDINGPARTICLE == i)
		{
			continue;
		}
		playing[i] -= inactive[i];
		removed[i] += inactive[i];
		inactive[i] = 0;
	}
}

int	number_of_bodies::get_n_SI() 
{
	return (playing[BODY_TYPE_STAR] + playing[BODY_TYPE_GIANTPLANET] + playing[BODY_TYPE_ROCKYPLANET] + playing[BODY_TYPE_PROTOPLANET]);
}

int number_of_bodies::get_n_NSI()
{
	return (playing[BODY_TYPE_SUPERPLANETESIMAL] + playing[BODY_TYPE_PLANETESIMAL]);
}

int	number_of_bodies::get_n_NI()
{
	return playing[BODY_TYPE_TESTPARTICLE];
}

int	number_of_bodies::get_n_total_initial()
{
	int n = 0;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? initial[i] : 0);
	}
	return n; 
}

int	number_of_bodies::get_n_total_playing()
{
	int n = 0;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? playing[i] : 0);
	}
	return n; 
}

int	number_of_bodies::get_n_total_active()
{
	int n = 0;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? playing[i] - inactive[i] : 0);
	}
	return n; 
}

int	number_of_bodies::get_n_total_inactive()
{
	int n = 0;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? inactive[i] : 0);
	}
	return n; 
}

int	number_of_bodies::get_n_total_removed()
{
	int n = 0;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? removed[i] : 0);
	}
	return n; 
}

int	number_of_bodies::get_n_GD()
{
	return (playing[BODY_TYPE_SUPERPLANETESIMAL] + playing[BODY_TYPE_PLANETESIMAL]);
}

int	number_of_bodies::get_n_MT1()
{
	return (playing[BODY_TYPE_ROCKYPLANET] + playing[BODY_TYPE_PROTOPLANET]);
}

int	number_of_bodies::get_n_MT2()
{
	return playing[BODY_TYPE_GIANTPLANET];
}

int	number_of_bodies::get_n_massive()
{
	return (get_n_SI() + get_n_NSI());
}

int number_of_bodies::get_n_prime_SI()
{
	// The number of self-interacting (SI) bodies alligned to n_tbp
	return ((get_n_SI() + n_tpb - 1) / n_tpb) * n_tpb;
}

int number_of_bodies::get_n_prime_NSI()
{
	// The number of non-self-interacting (NSI) bodies alligned to n_tbp
	return ((get_n_NSI() + n_tpb - 1) / n_tpb) * n_tpb;
}

int number_of_bodies::get_n_prime_NI()
{
	// The number of non-interacting (NI) bodies alligned to n_tbp
	return ((get_n_NI() + n_tpb - 1) / n_tpb) * n_tpb;
}

int number_of_bodies::get_n_prime_total()
{
	return (get_n_prime_SI() + get_n_prime_NSI() + get_n_prime_NI());
}

//int	number_of_bodies::get_n_prime_GD()
//{
//	return ((active_body[BODY_TYPE_SUPERPLANETESIMAL] + active_body[BODY_TYPE_PLANETESIMAL] + n_tpb - 1) / n_tpb) * n_tpb;
//}
//
//int	number_of_bodies::get_n_prime_MT1()
//{
//	return (n_rp + n_pp);
//}
//
//int	number_of_bodies::get_n_prime_MT2()
//{
//	return n_gp;
//}

int	number_of_bodies::get_n_prime_massive()
{
	return (get_n_prime_SI() + get_n_prime_NSI());
}


interaction_bound number_of_bodies::get_bound_SI()
{
	if (ups)
	{
		sink.x	 = 0, sink.y   = sink.x + get_n_prime_SI();
		source.x = 0, source.y = source.x + get_n_prime_massive();
	}
	else
	{
		sink.x   = 0, sink.y   = sink.x + get_n_SI();
		source.x = 0, source.y = source.x + get_n_massive();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NSI()
{
	if (ups)
	{
		sink.x   = get_n_prime_SI(), sink.y   = sink.x + get_n_prime_NSI();
		source.x = 0,				 source.y = source.x + get_n_prime_SI();;
	}
	else
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
		source.x = 0,		   source.y = source.x + get_n_SI();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NI()
{
	if (ups)
	{
		sink.x   = get_n_prime_massive(), sink.y   = sink.x + get_n_prime_NI();
		source.x = 0,				      source.y = source.x + get_n_prime_massive();
	}
	else
	{
		sink.x   = get_n_massive(), sink.y   = sink.x + get_n_NI();
		source.x = 0,   	        source.y = source.x + get_n_massive();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_GD()
{
	if (ups)
	{
		sink.x   = get_n_prime_SI(), sink.y   = sink.x;// + get_n_prime_GD();
		source.x = 0,		         source.y = 0;
	}
	else
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
		source.x = 0,		   source.y = 0;
	}

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
	static const char* body_type_name[] = 
	{
		"STAR",
		"GIANTPLANET",
		"ROCKYPLANET",
		"PROTOPLANET",
		"SUPERPLANETESIMAL",
		"PLANETESIMAL",
		"TESTPARTICLE",
		"PADDINGPARTICLE"
	};

	stream << "Number of bodies:" << endl;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		stream << setw(20) << body_type_name[i] << ": " << setw(5) << n_bodies->playing[i] - n_bodies->inactive[i] << endl;
	}
		
	return stream;
}
