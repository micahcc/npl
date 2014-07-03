#include <iostream>

#include "ndimage.h"

using namespace std;

int main()
{

	size_t sz[3] = {10, 23, 39};
	NDImageStore<3, double> testimage(sz);

	NDImage* testbase = &testimage;

	testimage.dbl({0,0,0}, 0);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;
	
	testimage.dbl({0,0,0}, 10);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;

}
