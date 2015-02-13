#pragma once

// includes project
#include "red_type.h"
#include "red_macro.h"

namespace constants
{
	const int_t	 CheckForSM			      = 100;
	const var_t SmallestNumber		      = 1.0e-50;

	const var_t Pi					      = 3.14159265358979323846;
	const var_t SqrtTwoPi			      = 2.50662827463100024161;

	const var_t Boltzman_SI			 	  = 1.3806488e-23;			        // J/K = (kg m^2) / (s^2 K)
	const var_t ProtonMass_SI		      = 1.672621777e-27;		        // kg
    const var_t BoltzmanProtonMass_SI     = Boltzman_SI / ProtonMass_SI;    // m^2 / (s^2 K) 8.25439928491377e3
    const var_t ProtonMassBoltzman_SI     = 1.0 / BoltzmanProtonMass_SI;    // (s^2 K) / m^2

    const var_t NewtonG					  = 6.67384e-11;		        	// m^3 / (kg s^2) = N m^2 / kg^2
	const var_t Gauss				      = 1.720209895e-2;                 // Au^3/2 / (Solar^1/2 day)
	const var_t Gauss2				      = 2.959122082855911025e-4;        // Au^3 / (Solar day^2)

	const var_t DegreeToRadian		      = 1.745329251994329509e-2;
	const var_t RadianToDegree		      = 1.0 / DegreeToRadian;

	const var_t SolarToMercury		      = 6.023600e6;
	const var_t SolarToVenus		      = 4.0852371e5;
	const var_t SolarToEarth		      = 3.3294605e5;
	const var_t SolarToEarthMoon	      = 3.2890056e5;
	const var_t SolarToMars				  = 3.098708e6;
	const var_t SolarToJupiter		      = 1.0473486e3;
	const var_t SolarToSaturn	 	      = 3.497898e3;
	const var_t SolarToUranus		      = 2.290298e4;
	const var_t SolarToNeptune		      = 1.941224e4;
	const var_t SolarToCeres			  = 2.109342e9;			  
	const var_t EarthToMoon				  = 8.130059e1;

	const var_t SolarToKilogram			  = 1.98911e30;
	const var_t SolarToGram				  = 1.0e3 * SolarToKilogram;

	const var_t MercuryToSolar		      = 1.0 / SolarToMercury;
	const var_t VenusToSolar		      = 1.0 / SolarToVenus;
	const var_t EarthToSolar		      = 1.0 / SolarToEarth;
	const var_t EarthMoonToSolar	      = 1.0 / SolarToEarthMoon; 
	const var_t MarsToSolar				  = 1.0 / SolarToMars;
	const var_t JupiterToSolar		      = 1.0 / SolarToJupiter;
	const var_t SaturnToSolar		      = 1.0 / SolarToSaturn;
	const var_t UranusToSolar		      = 1.0 / SolarToUranus;
	const var_t NeptuneToSolar		      = 1.0 / SolarToNeptune;
	const var_t CeresToSolar              = 1.0 / SolarToCeres;
	const var_t MoonToEarth				  = 1.0 / EarthToMoon;


	const var_t KilogramToSolar			  = 1.0 / SolarToKilogram;
	const var_t GramToSolar				  = 1.0 / SolarToGram;

	const var_t AuToMeter			      = 1.495978707e11;
	const var_t AuToCentimeter		      = AuToMeter * 1.0e2;
	const var_t AuToKilometer		      = AuToMeter / 1.0e3;
	const var_t AuToSolarRadius		      = 215.094;

	const var_t MeterToAu			      = 1.0 / AuToMeter;
	const var_t KilometerToAu		      = 1.0 / AuToKilometer;
	const var_t SolarRadiusToAu			  = 1.0 / AuToSolarRadius;

	const var_t YearToDay			      = 365.25;
	const var_t DayToSecond				  = 86400.0;

	const var_t DayToYear			      = 1.0 / YearToDay;
	const var_t SecondToDay				  = 1.0 / DayToSecond;

	const var_t GramPerCm2ToEarthPerAu2   = GramToSolar*SolarToEarth / (SQR(1.0e-2*MeterToAu));
	const var_t EarthPerAu2ToGramPerCm2   = 1.0 / GramPerCm2ToEarthPerAu2;

	const var_t GramPerCm2ToSolarPerAu2   = GramToSolar / (SQR(1.0e-2*MeterToAu));
	const var_t SolarPerAu2ToGramPerCm2   = 1.0 / GramPerCm2ToSolarPerAu2;
	const var_t EarthPerAu2ToSolarPerAu2  = EarthToSolar;

	const var_t GramPerCm3ToSolarPerAu3   = GramToSolar / (CUBE(1.0e-2*MeterToAu));
	const var_t SolarPerAu3ToGramPerCm3   = 1.0 / GramPerCm3ToSolarPerAu3;

	const var_t Boltzman_CMU		      = Boltzman_SI * (KilogramToSolar * SQR(MeterToAu)) / (SQR(SecondToDay));	         // (Solar AU^2)/(day^2 K) 2.315266990160467e-66
	const var_t ProtonMass_CMU		      = ProtonMass_SI * KilogramToSolar; // Solar       8.408895320017495e-58 
    const var_t BoltzmanProtonMass_CMU    = Boltzman_CMU / ProtonMass_CMU;   // Au^2 / (d^2 K) 2.753354515722108e-9
    const var_t ProtonMassBoltzman_CMU    = 1.0 / BoltzmanProtonMass_CMU;    // (d^2 K) / Au^2  

} /* constants */
