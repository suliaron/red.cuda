#pragma once

#include "red_type.h"

class number_of_bodies
{
public:
	number_of_bodies(int n_s, int n_gp, int n_rp, int n_pp, int n_spl, int n_pl, int n_tp, int n_tpb, bool ups);

	void update();

	int	get_n_total_initial();
	int	get_n_total_playing();
	int	get_n_total_active();
	int	get_n_total_inactive();
	int	get_n_total_removed();

	//int	get_n_total();
	//int	get_n_total_inactive();
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
	int	get_n_GD();
	//! Calculates the number of bodies which are experiencing type I migartion, i.e. sum of the number of rocky- and proto-planets.
	int	get_n_MT1();
	//! Calculates the number of bodies which are experiencing type II migartion, i.e. the number of giant planets.
	int	get_n_MT2();

	int get_n_prime_total();
	int	get_n_prime_massive();

	int get_n_prime_SI();
	int get_n_prime_NSI();
	int get_n_prime_NI();

	void set_n_tpb(int n) { n_tpb = n;    }
	int  get_n_tpb()      { return n_tpb; }

	//int	get_n_prime_GD();
	//int	get_n_prime_MT1();
	//int	get_n_prime_MT2();

	interaction_bound get_bound_SI();
	interaction_bound get_bound_NSI();
	interaction_bound get_bound_NI();
	interaction_bound get_bound_GD();
	//interaction_bound get_bound_MT1();
	//interaction_bound get_bound_MT2();

	int initial[BODY_TYPE_N];   //!< Number of initial bodies
	int playing[BODY_TYPE_N];   //!< Number of bodies which are iterated over in the gravitational computation (may have negative id)
	int inactive[BODY_TYPE_N];  //!< Number of bodies which has negative id (these are part of the playing bodies, and are flaged to be removed in the next call to remove inactive bodies function)
	int removed[BODY_TYPE_N];   //!< Number of removed bodies

private:
	int n_tpb;          //!< The number of threads per block
	bool ups;

	int2_t sink;
	int2_t source;
};
