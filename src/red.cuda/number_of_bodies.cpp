// includes system
#include <iomanip>
#include <iostream>

// includes project
#include "number_of_bodies.h"

using namespace std;

number_of_bodies::number_of_bodies(unsigned int n_s, unsigned int n_gp, unsigned int n_rp, unsigned int n_pp, unsigned int n_spl, unsigned int n_pl, unsigned int n_tp)
{
	playing[BODY_TYPE_STAR]              = initial[BODY_TYPE_STAR]              = n_s;
	playing[BODY_TYPE_GIANTPLANET]       = initial[BODY_TYPE_GIANTPLANET]       = n_gp;
	playing[BODY_TYPE_ROCKYPLANET]       = initial[BODY_TYPE_ROCKYPLANET]       = n_rp;
	playing[BODY_TYPE_PROTOPLANET]       = initial[BODY_TYPE_PROTOPLANET]       = n_pp;
	playing[BODY_TYPE_SUPERPLANETESIMAL] = initial[BODY_TYPE_SUPERPLANETESIMAL] = n_spl;
	playing[BODY_TYPE_PLANETESIMAL]      = initial[BODY_TYPE_PLANETESIMAL]      = n_pl;
	playing[BODY_TYPE_TESTPARTICLE]      = initial[BODY_TYPE_TESTPARTICLE]      = n_tp;
	playing[BODY_TYPE_PADDINGPARTICLE]   = initial[BODY_TYPE_PADDINGPARTICLE]   = 0;

	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		inactive[i] = 0;
		removed[i]  = 0;
	}

    sink.x   = sink.y   = 0;
    source.x = source.y = 0;
}

void number_of_bodies::update()
{
	n_removed = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		if (BODY_TYPE_PADDINGPARTICLE == i)
		{
			continue;
		}
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
		n += (i != BODY_TYPE_PADDINGPARTICLE ? initial[i] : 0);
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_playing()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? playing[i] : 0);
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_active()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? playing[i] - inactive[i] : 0);
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_inactive()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? inactive[i] : 0);
	}
	return n; 
}

unsigned int number_of_bodies::get_n_total_removed()
{
	unsigned int n = 0;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		n += (i != BODY_TYPE_PADDINGPARTICLE ? removed[i] : 0);
	}
	return n; 
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

unsigned int number_of_bodies::get_n_prime_SI(unsigned int n_tpb)
{
	// The number of self-interacting (SI) bodies alligned to n_tbp
	return ((get_n_SI() + n_tpb - 1) / n_tpb) * n_tpb;
}

unsigned int number_of_bodies::get_n_prime_NSI(unsigned int n_tpb)
{
	// The number of non-self-interacting (NSI) bodies alligned to n_tbp
	return ((get_n_NSI() + n_tpb - 1) / n_tpb) * n_tpb;
}

unsigned int number_of_bodies::get_n_prime_NI(unsigned int n_tpb)
{
	// The number of non-interacting (NI) bodies alligned to n_tbp
	return ((get_n_NI() + n_tpb - 1) / n_tpb) * n_tpb;
}

unsigned int number_of_bodies::get_n_prime_total(unsigned int n_tpb)
{
	return (get_n_prime_SI(n_tpb) + get_n_prime_NSI(n_tpb) + get_n_prime_NI(n_tpb));
}

unsigned int number_of_bodies::get_n_active_by(body_type_t type)
{
	return (playing[type] - inactive[type]);
}


//unsigned int	number_of_bodies::get_n_prime_GD()
//{
//	return ((active_body[BODY_TYPE_SUPERPLANETESIMAL] + active_body[BODY_TYPE_PLANETESIMAL] + n_tpb - 1) / n_tpb) * n_tpb;
//}
//
//unsigned int	number_of_bodies::get_n_prime_MT1()
//{
//	return (n_rp + n_pp);
//}
//
//unsigned int	number_of_bodies::get_n_prime_MT2()
//{
//	return n_gp;
//}

unsigned int number_of_bodies::get_n_prime_massive(unsigned int n_tpb)
{
	return (get_n_prime_SI(n_tpb) + get_n_prime_NSI(n_tpb));
}

interaction_bound number_of_bodies::get_bound_SI(bool ups, unsigned int n_tpb)
{
	if (ups)
	{
		sink.x	 = 0, sink.y   = sink.x + get_n_prime_SI(n_tpb);
		source.x = 0, source.y = source.x + get_n_prime_massive(n_tpb);
	}
	else
	{
		sink.x   = 0, sink.y   = sink.x + get_n_SI();
		source.x = 0, source.y = source.x + get_n_massive();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NSI(bool ups, unsigned int n_tpb)
{
	if (ups)
	{
		sink.x   = get_n_prime_SI(n_tpb), sink.y   = sink.x + get_n_prime_NSI(n_tpb);
		source.x = 0,				      source.y = source.x + get_n_prime_SI(n_tpb);
	}
	else
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
		source.x = 0,		   source.y = source.x + get_n_SI();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NI(bool ups, unsigned int n_tpb)
{
	if (ups)
	{
		sink.x   = get_n_prime_massive(n_tpb), sink.y   = sink.x + get_n_prime_NI(n_tpb);
		source.x = 0,				           source.y = source.x + get_n_prime_massive(n_tpb);
	}
	else
	{
		sink.x   = get_n_massive(), sink.y   = sink.x + get_n_NI();
		source.x = 0,   	        source.y = source.x + get_n_massive();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_GD(bool ups, unsigned int n_tpb)
{
	if (ups)
	{
		sink.x   = get_n_prime_SI(n_tpb), sink.y   = sink.x;// + get_n_prime_GD();
		source.x = 0,		              source.y = 0;
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
	stream << "Number of bodies:" << endl;
	for (unsigned int i = 0; i < BODY_TYPE_N; i++)
	{
		stream << setw(20) << body_type_name[i] << ": " << setw(5) << n_bodies->playing[i] - n_bodies->inactive[i] << endl;
	}
		
	return stream;
}
