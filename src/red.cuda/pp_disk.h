#pragma once

// includes system
#include <string>

// includes project
#include "red_type.h"

using namespace std;

typedef enum migration_type
	{
		MIGRATION_TYPE_NO,
		MIGRATION_TYPE_TYPE_I,
		MIGRATION_TYPE_TYPE_II
	} migration_type_t;

typedef enum body_type
	{
		BODY_TYPE_STAR,
		BODY_TYPE_GIANTPLANET,
		BODY_TYPE_ROCKYPLANET,
		BODY_TYPE_PROTOPLANET,
		BODY_TYPE_SUPERPLANETESIMAL,
		BODY_TYPE_PLANETESIMAL,
		BODY_TYPE_TESTPARTICLE,
		BODY_TYPE_N
	} body_type_t;

typedef enum orbelem_name
	{
		ORBELEM_SMA,
		ORBELEM_ECC,
		ORBELEM_INC,
		ORBELEM_PERI,
		ORBELEM_NODE,
		ORBELEM_MEAN
	} orbelem_name_t;

typedef enum phys_prop_name
	{
		MASS,
		RADIUS,
		DENSITY,
		DRAG_COEFF
	} phys_prop_name_t;

typedef enum event_name
	{
		EVENT_NAME_NONE,
		EVENT_NAME_HIT_CENTRUM,
		EVENT_NAME_EJECTION,
		EVENT_NAME_CLOSE_ENCOUNTER,
		EVENT_NAME_COLLISION,
		EVENT_NAME_N
	} event_name_t;

class pp_disk
{
public:

	var_t*	id;
};