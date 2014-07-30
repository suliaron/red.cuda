#pragma once

// These macro functions must be enclosed in parentheses in order to give
// correct results in the case of a division i.e. 1/SQR(x) -> 1/((x)*(x))
#define	SQR(x)		((x)*(x))
#define	CUBE(x)		((x)*(x)*(x))
#define FORTH(x)	((x)*(x)*(x)*(x))
#define FIFTH(x)	((x)*(x)*(x)*(x)*(x))
