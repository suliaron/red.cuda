#pragma once

// includes system
#include <string>

// includes project
#include "red_type.h"

class number_of_bodies
{
public:
	number_of_bodies(unsigned int n_s, unsigned int n_gp, unsigned int n_rp, unsigned int n_pp, unsigned int n_spl, unsigned int n_pl, unsigned int n_tp);

	unsigned int get_n_total_initial();
	unsigned int get_n_total_playing();
	unsigned int get_n_total_active();
	unsigned int get_n_total_inactive();
	unsigned int get_n_total_removed();

	std::string get_n_playing();

	//! Calculates the number of playing bodies with mass
	unsigned int get_n_massive();
	//! Calculates the number of playing bodies which are self-interacting (i.e. it returns n_s + n_gp + n_rp + n_pp)
	unsigned int get_n_SI();
	//! Calculates the number of playing non-self-interating bodies (i.e. it returns n_spl + n_pl)
	unsigned int get_n_NSI();
	//! Calculates the number of playing non-interating bodies (i.e. returns n_tp)
	unsigned int get_n_NI();
	//! Calculates the number of playing bodies which feels the drag force, i.e. sum of the number of super-planetesimals and planetesimals.
	unsigned int get_n_GD();
	//! Calculates the number of playing bodies which are experiencing type I migartion, i.e. sum of the number of rocky- and proto-planets.
	unsigned int get_n_MT1();
	//! Calculates the number of playing bodies which are experiencing type II migartion, i.e. the number of giant planets.
	unsigned int get_n_MT2();
	//! Returns the number of body with the specified type
	unsigned int get_n_active_by(body_type_t type);

	interaction_bound get_bound_SI( unsigned int n_tpb);
	interaction_bound get_bound_NSI(unsigned int n_tpb);
	interaction_bound get_bound_NI( unsigned int n_tpb);
	interaction_bound get_bound_GD( unsigned int n_tpb);
	//interaction_bound get_bound_MT1();
	//interaction_bound get_bound_MT2();

	void update();

	unsigned int initial[BODY_TYPE_N];   //!< Number of initial bodies
	unsigned int playing[BODY_TYPE_N];   //!< Number of bodies which are iterated over in the gravitational computation (may have negative id)
	unsigned int inactive[BODY_TYPE_N];  //!< Number of bodies which has negative id (these are part of the playing bodies, and are flaged to be removed in the next call to remove inactive bodies function)
	unsigned int removed[BODY_TYPE_N];   //!< Number of removed bodies

	unsigned int n_removed;              //!< Number of bodies which were removed during the last update() function call

private:
	uint2_t sink;
	uint2_t source;
};
