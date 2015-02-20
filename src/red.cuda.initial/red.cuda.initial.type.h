#pragma once

#include "distribution.h"
#include "red_type.h"


typedef enum input_format_name
		{
			INPUT_FORMAT_RED,
			INPUT_FORMAT_NONAME,
			INPUT_FORMAT_HIPERION,
            INPUT_FORMAT_N
		} input_format_name_t;

typedef enum orbelem_name
		{
			ORBITAL_ELEMENT_SMA,
			ORBITAL_ELEMENT_ECC,
			ORBITAL_ELEMENT_INC,
			ORBITAL_ELEMENT_PERI,
			ORBITAL_ELEMENT_NODE,
			ORBITAL_ELEMENT_MEAN,
            ORBITAL_ELEMENT_N
		} orbelem_name_t;

typedef enum phys_prop_name
	    {
		    MASS,
		    RADIUS,
		    DENSITY,
		    DRAG_COEFF
	    } phys_prop_name_t;

typedef struct oe_dist
		{
			var2_t				range[ORBITAL_ELEMENT_N];
			distribution_base*  item[ORBITAL_ELEMENT_N];
		} oe_dist_t;

typedef struct phys_prop_dist
		{
			var2_t				range[4];
			distribution_base*  item[4];
		} phys_prop_dist_t;

typedef struct body_disk
		{
			vector<string>		names;
			int_t				nBody[BODY_TYPE_N];
			oe_dist_t			oe_d[BODY_TYPE_N];
			phys_prop_dist_t	pp_d[BODY_TYPE_N];
			migration_type_t	*mig_type;
			var_t				*stop_at;

			body_disk()
			{
				for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
				{
					nBody[type] = 0;
					for (int i = 0; i < ORBITAL_ELEMENT_N; i++) 
					{
						oe_d[type].range[i].x = oe_d[type].range[i].y = 0.0;
						oe_d[type].item[i] = 0x0;
					}
					for (int i = 0; i < 4; i++) 
					{
						pp_d[type].range[i].x = pp_d[type].range[i].y = 0.0;
						pp_d[type].item[i] = 0x0;
					}
				}
				mig_type = 0x0;
				stop_at  = 0x0;
			}

		} body_disk_t;
