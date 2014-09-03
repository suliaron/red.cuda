// includes system
#include <iomanip>
#include <iostream>

// includes project
#include "number_of_bodies.h"

using namespace std;

number_of_bodies::number_of_bodies(int star, int giant_planet, int rocky_planet, int proto_planet, int super_planetesimal, int planetesimal, int test_particle) : 
		star(star), 
		giant_planet(giant_planet), 
		rocky_planet(rocky_planet), 
		proto_planet(proto_planet), 
		super_planetesimal(super_planetesimal), 
		planetesimal(planetesimal), 
		test_particle(test_particle) 
{
	total = star + giant_planet + rocky_planet + proto_planet + super_planetesimal + planetesimal + test_particle;
	sink.x = sink.y = 0;
	source.x = source.y = 0;
}

int	number_of_bodies::get_n_total()
{
	total = star + giant_planet + rocky_planet + proto_planet + super_planetesimal + planetesimal + test_particle; 
	return total;
}

int	number_of_bodies::get_n_massive()
{
	return star + giant_planet + rocky_planet + proto_planet + super_planetesimal + planetesimal;
}

int	number_of_bodies::get_n_self_interacting() 
{
	return star + giant_planet + rocky_planet + proto_planet;
}

int	number_of_bodies::get_n_gas_drag()
{
	return super_planetesimal + planetesimal;
}

int	number_of_bodies::get_n_migrate_typeI()
{
	return rocky_planet + proto_planet;
}

int	number_of_bodies::get_n_migrate_typeII()
{
	return giant_planet;
}

interaction_bound number_of_bodies::get_self_interacting()
{
	sink.x		= 0;
	sink.y		= get_n_self_interacting();
	source.x	= 0;
	source.y	= get_n_massive();
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_nonself_interacting()
{
	sink.x			= get_n_self_interacting();
	sink.y			= get_n_massive();
	source.x		= 0;
	source.y		= get_n_self_interacting();
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_non_interacting()
{
	sink.x			= get_n_massive();
	sink.y			= total;
	source.x		= 0;
	source.y		= get_n_massive();
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_bodies_gasdrag() {
	sink.x			= get_n_self_interacting();
	sink.y			= get_n_massive();
	source.x		= 0;
	source.y		= 0;
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_bodies_migrate_typeI() {
	sink.x			= star + giant_planet;
	sink.y			= get_n_massive();
	source.x		= 0;
	source.y		= 0;
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_bodies_migrate_typeII() {
	sink.x			= star;
	sink.y			= star + giant_planet;
	source.x		= 0;
	source.y		= 0;
	interaction_bound iBound(sink, source);

	return iBound;
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
	stream << setw(20) << body_type_name[0] << ": " << n_bodies->star << endl;
	stream << setw(20) << body_type_name[1] << ": " << n_bodies->giant_planet << endl;
	stream << setw(20) << body_type_name[2] << ": " << n_bodies->rocky_planet << endl;
	stream << setw(20) << body_type_name[3] << ": " << n_bodies->proto_planet << endl;
	stream << setw(20) << body_type_name[4] << ": " << n_bodies->super_planetesimal << endl;
	stream << setw(20) << body_type_name[5] << ": " << n_bodies->planetesimal << endl;
	stream << setw(20) << body_type_name[6] << ": " << n_bodies->test_particle << endl;
		
	return stream;
}
