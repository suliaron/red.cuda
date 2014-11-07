// includes system
#include <iomanip>
#include <iostream>

// includes project
#include "number_of_bodies.h"

using namespace std;

number_of_bodies::number_of_bodies(int n_s, int n_gp, int n_rp, int n_pp, int n_spl, int n_pl, int n_tp, int n_tpb, bool ups) : 
		n_s(n_s), 
		n_gp(n_gp), 
		n_rp(n_rp), 
		n_pp(n_pp), 
		n_spl(n_spl), 
		n_pl(n_pl), 
		n_tp(n_tp),
		n_tpb(n_tpb),
		ups(ups)
{
    sink.x   = sink.y   = 0;
    source.x = source.y = 0;
}

void number_of_bodies::update_numbers(body_metadata_t *body_md)
{
	int n_active_body		= 0;
	int n_inactive_body		= 0;

	int	star				= 0;
	int	giant_planet		= 0;
	int	rocky_planet		= 0;
	int	proto_planet		= 0;
	int	super_planetesimal	= 0;
	int	planetesimal		= 0;
	int	test_particle		= 0;

	int n_total = ups ? get_n_prime_total() : get_n_total();
	for (int i = 0; i < n_total; i++)
	{
		if (0 < body_md[i].id && BODY_TYPE_PADDINGPARTICLE > body_md[i].body_type)
		{
			n_active_body++;
		}
		// Count the inactive bodies by type
		else
		{
			n_inactive_body++;
			switch (body_md[i].body_type)
			{
			case BODY_TYPE_STAR:
				star++;
				break;
			case BODY_TYPE_GIANTPLANET:
				giant_planet++;
				break;
			case BODY_TYPE_ROCKYPLANET:
				rocky_planet++;
				break;
			case BODY_TYPE_PROTOPLANET:
				proto_planet++;
				break;
			case BODY_TYPE_SUPERPLANETESIMAL:
				super_planetesimal++;
				break;
			case BODY_TYPE_PLANETESIMAL:
				planetesimal++;
				break;
			case BODY_TYPE_TESTPARTICLE:
				test_particle++;
				break;
			case BODY_TYPE_PADDINGPARTICLE:
				n_inactive_body--;
				break;
			default:
				throw string("Undefined body type!");
			}
		}
	}
	cout << "There are " << star << " inactive star" << endl;
	cout << "There are " << giant_planet << " inactive giant planet" << endl;
	cout << "There are " << rocky_planet << " inactive rocky planet" << endl;
	cout << "There are " << proto_planet << " inactive protoplanet" << endl;
	cout << "There are " << super_planetesimal << " inactive super planetesimal" << endl;
	cout << "There are " << planetesimal << " inactive planetesimal" << endl;
	cout << "There are " << test_particle << " inactive test particle" << endl;

	n_s		-= star;
	n_gp	-= giant_planet;
	n_rp	-= rocky_planet;
	n_pp	-= proto_planet;
	n_spl	-= super_planetesimal;
	n_pl	-= planetesimal;
	n_tp	-= test_particle;
}

int	number_of_bodies::get_n_SI() 
{
	return (n_s + n_gp + n_rp + n_pp);
}

int number_of_bodies::get_n_NSI()
{
	return (n_spl + n_pl);
}

int	number_of_bodies::get_n_NI()
{
	return n_tp;
}

int	number_of_bodies::get_n_total()
{
	return (n_s + n_gp + n_rp + n_pp + n_spl + n_pl + n_tp); 
}

int	number_of_bodies::get_n_GD()
{
	return (n_spl + n_pl);
}

int	number_of_bodies::get_n_MT1()
{
	return (n_rp + n_pp);
}

int	number_of_bodies::get_n_MT2()
{
	return n_gp;
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
//	return ((n_spl + n_pl + n_tpb - 1) / n_tpb) * n_tpb;
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
		sink.x   = get_n_prime_SI(), sink.y   = sink.x + n_spl + n_pl;
		source.x = 0,		         source.y = 0;
	}
	else
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + n_spl + n_pl;
		source.x = 0,		   source.y = 0;
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_MT1()
{
	sink.x   = n_s + n_gp, sink.y   = sink.x + n_pp + n_rp;
	source.x = 0,	       source.y = source.x + 0;

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_MT2()
{
	sink.x   = n_s,   sink.y = sink.x + n_gp;
	source.x = 0, 	source.y = source.x + 0;

	return interaction_bound(sink, source);
}

ostream& operator<<(ostream& stream, const number_of_bodies* n_bodies)
{
	const char* body_type_name[] = 
	{
		"STAR",
		"GIANTPLANET",
		"ROCKYPLANET",
		"PROTOPLANET",
		"SUPERPLANETESIMAL",
		"PLANETESIMAL",
		"TESTPARTICLE",
	};

	stream << "Number of bodies:" << endl;
	stream << setw(20) << body_type_name[0] << ": " << n_bodies->n_s << endl;
	stream << setw(20) << body_type_name[1] << ": " << n_bodies->n_gp << endl;
	stream << setw(20) << body_type_name[2] << ": " << n_bodies->n_rp << endl;
	stream << setw(20) << body_type_name[3] << ": " << n_bodies->n_pp << endl;
	stream << setw(20) << body_type_name[4] << ": " << n_bodies->n_spl << endl;
	stream << setw(20) << body_type_name[5] << ": " << n_bodies->n_pl << endl;
	stream << setw(20) << body_type_name[6] << ": " << n_bodies->n_tp << endl;
		
	return stream;
}
