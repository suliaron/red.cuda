#pragma once

#include <string>

#include "red_type.h"

class number_of_bodies
{
public:
	number_of_bodies(uint32_t n_s, uint32_t n_gp, uint32_t n_rp, uint32_t n_pp, uint32_t n_spl, uint32_t n_pl, uint32_t n_tp);

	uint32_t get_n_total_initial();
	uint32_t get_n_total_playing();
	uint32_t get_n_total_active();
	uint32_t get_n_total_inactive();
	uint32_t get_n_total_removed();

	std::string get_n_playing();

	//! Calculates the number of playing bodies with mass
	uint32_t get_n_massive();
	//! Calculates the number of playing bodies which are self-interacting (i.e. it returns n_s + n_gp + n_rp + n_pp)
	uint32_t get_n_SI();
	//! Calculates the number of playing non-self-interating bodies (i.e. it returns n_spl + n_pl)
	uint32_t get_n_NSI();
	//! Calculates the number of playing non-interating bodies (i.e. returns n_tp)
	uint32_t get_n_NI();
	//! Calculates the number of playing bodies which feels the drag force, i.e. sum of the number of super-planetesimals and planetesimals.
	uint32_t get_n_GD();
	//! Calculates the number of playing bodies which are experiencing type I migartion, i.e. sum of the number of rocky- and proto-planets.
	uint32_t get_n_MT1();
	//! Calculates the number of playing bodies which are experiencing type II migartion, i.e. the number of giant planets.
	uint32_t get_n_MT2();
	//! Returns the number of body with the specified type
	uint32_t get_n_active_by(body_type_t type);

	interaction_bound get_bound_SI( uint32_t n_tpb);
	interaction_bound get_bound_NSI(uint32_t n_tpb);
	interaction_bound get_bound_NI( uint32_t n_tpb);
	interaction_bound get_bound_GD( uint32_t n_tpb);
	//interaction_bound get_bound_MT1();
	//interaction_bound get_bound_MT2();

	void update();

	uint32_t initial[BODY_TYPE_N];   //!< Number of initial bodies
	uint32_t playing[BODY_TYPE_N];   //!< Number of bodies which are iterated over in the gravitational computation (may have negative id)
	uint32_t inactive[BODY_TYPE_N];  //!< Number of bodies which has negative id (these are part of the playing bodies, and are flaged to be removed in the next call to remove inactive bodies function)
	uint32_t removed[BODY_TYPE_N];   //!< Number of removed bodies

	uint32_t n_removed;              //!< Number of bodies which were removed during the last update() function call

private:
	uint2_t sink;
	uint2_t source;
};
