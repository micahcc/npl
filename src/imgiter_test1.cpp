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
	std::vector<size_t> sz({10, 23, 39, 8});
	MRImage* testimg = createMRImage(sz, FLOAT64);

	int64_t ii = 0;
	for(auto iter = testimg->begin(); !iter.isEnd(); ++iter) {
		iter.int64(ii++);
	}

	/* Write the Image */
	writeMRImage(testimg, argv[1]);

	/* Read the Image */
	MRImage* iimage = readMRImage(argv[1], true);
	
	/* Check the Image */
	ii = 0;
	for(auto iter = iimage->begin(); !iter.isEnd(); ++iter) {
		if(iter.int64() != ii++) {
			cerr << "Error, mismatch in read image" << endl;
			return -1;
		}
	}

	cerr << "PASS!" << endl;
	return 0;
}

