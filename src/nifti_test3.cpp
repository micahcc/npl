#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

// read image, write image, read it again, check that the image information 
// is the same, and dump out the final for more information
int main(int argc, char** argv)
{
	if(argc != 4) {
		cerr << "Need input and output image" << endl;
		cerr << argv[0] << "input output-v1 output-v2" << endl;
		return -1;
	}

	MRImage* iimage = readMRImage(argv[1], true);
	cerr << "Original: " << endl << *iimage << endl;

	writeMRImage(iimage, argv[2]);
	writeMRImage(iimage, argv[3], true);
	
	MRImage* version1, *version2;
	if(!(version1 = readMRImage(argv[2], true)))
		return -1;
	if(!(version2 = readMRImage(argv[3], true)))
		return -1;

	cerr << "Version 1: " << endl << *version1 << endl;
	cerr << "Version 2: " << endl << *version2 << endl;
	return 0;
}

