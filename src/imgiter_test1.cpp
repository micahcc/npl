#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	if(argc != 2) {
		cerr << "Need a filename!" << endl;
		return -1;
	}

	/* Create an image with: x+y*100+z*10000*/
	size_t sz[4] = {10, 23, 39, 8};
	MRImage* testimg = createMRImage(4, sz, FLOAT64);

	size_t ii = 0;
	for(auto iter = testimg->begin_dbl(); !iter.isEnd(); ++iter) {
		iter.set(ii++);
	}

	/* Write the Image */
	writeMRImage(testimg, argv[1]);

	/* Read the Image */
	MRImage* iimage = readMRImage(argv[1], true);
	
	/* Check the Image */
	ii = 0;
	for(auto iter = iimage->begin_dbl(); !iter.isEnd(); ++iter) {
		if(iter.get() != ii++) {
			cerr << "Error, mismatch in read image" << endl;
			return -1;
		}
	}

	cerr << "PASS!" << endl;
	return 0;
}

