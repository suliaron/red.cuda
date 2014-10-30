#pragma once

#include "red_type.h"

class number_of_bodies {
public:
	number_of_bodies(int n_s, int n_gp, int n_rp, int n_pp, int n_spl, int n_pl, int n_tp);

	void update_numbers(body_metadata_t *body_md);

	int	get_n_total();
	//! Calculates the number of bodies with mass, i.e. sum of the number of stars, giant planets, 
	/*  rocky planets, protoplanets, super-planetesimals and planetesimals.
	*/
	int	get_n_massive();
	//! Calculates the number of bodies which are self-interacting (i.e. it returns n_s + n_gp + n_rp + n_pp)
	int	get_n_SI();
	//! Calculates the number of non-self-interating bodies (i.e. it returns n_spl + n_pl)
	int	get_n_NSI();
	//! Calculates the number of non-interating bodies (i.e. returns n_tp)
	int	get_n_NI();
	//! Calculates the number of bodies which feels the drag force, i.e. sum of the number of super-planetesimals and planetesimals.
	int	get_n_gas_drag();
	//! Calculates the number of bodies which are experiencing type I migartion, i.e. sum of the number of rocky- and proto-planets.
	int	get_n_migrate_typeI();
	//! Calculates the number of bodies which are experiencing type II migartion, i.e. the number of giant planets.
	int	get_n_migrate_typeII();

	int get_n_prime_SI(int n_tpb);
	int get_n_prime_NSI(int n_tpb);
	int get_n_prime_NI(int n_tpb);
	int get_n_prime_total(int n_tpb);

	interaction_bound get_self_interacting();
	interaction_bound get_nonself_interacting();
	interaction_bound get_non_interacting();
	interaction_bound get_bodies_gasdrag();
	interaction_bound get_bodies_migrate_typeI();
	interaction_bound get_bodies_migrate_typeII();

	int	n_s;
	int	n_gp;
	int	n_rp;
	int	n_pp;
	int	n_spl;
	int	n_pl;
	int	n_tp;

private:

	int2_t sink;
	int2_t source;
};
