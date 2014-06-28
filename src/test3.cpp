#include <iostream>

#include "ndimage.h"

using namespace std;

int main()
{

	size_t sz[3] = {10, 23, 39};
	NDImageStore<3, double> testimage(sz);

	NDImage* testbase = &testimage;

	testimage.setdbl(0,{0,0,0});
	cerr << testimage.getdbl({0,0,0}) << endl;
	cerr << testbase->getdbl({0,0,0}) << endl;
	
	testimage.setdbl(10,{0,0,0});
	cerr << testimage.getdbl({0,0,0}) << endl;
	cerr << testbase->getdbl({0,0,0}) << endl;

}
