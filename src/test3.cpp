#include <iostream>

#include "ndimage.h"

using namespace std;

int main()
{

	size_t sz[3] = {10, 23, 39};
	NDImageStore<3, double> testimage(sz);

	NDImage* testbase = &testimage;

	testimage.setD(0,0,0,0);
	cerr << testimage.getD(0,0,0) << endl;
	cerr << testbase->getD(0,0,0) << endl;
	
	testimage.setD(10,0,0,0);
	cerr << testimage.getD(0,0,0) << endl;
	cerr << testbase->getD(0,0,0) << endl;

}
