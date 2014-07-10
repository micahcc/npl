#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

int main()
{

	MRImageStore<3, double> testimage({10, 23, 39});

	MRImage* testbase = &testimage;

	testimage.dbl({0,0,0}, 0);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;
	
	testimage.dbl({0,0,0}, 10);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;

}
