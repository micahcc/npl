#include "ndarray.h"
#include <iostream>

using namespace std;

int main()
{
	size_t dim1[] = {4,4,3};
	NDArrayStore<3, float> test1(dim1);
	cerr << "Bytes: " << test1.getBytes() << endl;

	for(size_t ii = 0; ii < test1.getBytes()/sizeof(float); ii++)
		test1.m_data[ii] = ii;
	
	NDArray* testp = &test1;

	cerr << "Dimensions:" << testp->getNDim() << endl;
	for(size_t zz=0; zz < testp->dim(2); zz++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t xx=0; xx < testp->dim(0); xx++) {
				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
}
