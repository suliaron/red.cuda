#pragma once

#include "red_type.h"

class number_of_bodies {
public:
	number_of_bodies(int star, int giant_planet, int rocky_planet, int proto_planet, int super_planetesimal, int planetesimal, int test_particle);

	int		get_n_total();
	//! Calculates the number of bodies with mass, i.e. sum of the number of stars, giant planets, 
	/*  rocky planets, protoplanets, super-planetesimals and planetesimals.
	*/
	int		get_n_massive();
	//! Calculates the number of bodies which are self-interacting, i.e. sum of the number of stars, giant planets, 
	/*  rocky planets and protoplanets.
	*/
	int		get_n_self_interacting();
	//! Calculates the number of non-self-interating bodies (super_planetesimal + planetesimal)
	int		get_n_nonself_interacting();
	//! Calculates the number of bodies which feels the drag force, i.e. sum of the number of super-planetesimals and planetesimals.
	int		get_n_gas_drag();
	//! Calculates the number of bodies which are experiencing type I migartion, i.e. sum of the number of rocky- and proto-planets.
	int		get_n_migrate_typeI();
	//! Calculates the number of bodies which are experiencing type II migartion, i.e. the number of giant planets.
	int		get_n_migrate_typeII();

	interaction_bound get_self_interacting();
	interaction_bound get_nonself_interacting();
	interaction_bound get_non_interacting();
	interaction_bound get_bodies_gasdrag();
	interaction_bound get_bodies_migrate_typeI();
	interaction_bound get_bodies_migrate_typeII();

	int		star;
	int		giant_planet;
	int		rocky_planet;
	int		proto_planet;
	int		super_planetesimal;
	int		planetesimal;
	int		test_particle;
	int		total;

private:
	int2_t		sink;
	int2_t		source;
};
