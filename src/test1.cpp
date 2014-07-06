#include "ndarray.h"
#include <iostream>
#include <ctime>

using namespace std;
using namespace npl;

int main()
{
	size_t dim1[] = {100,1000,1000};
	NDArrayStore<3, float> test1(dim1);
	cerr << "Bytes: " << test1.bytes() << endl;

	for(size_t ii = 0; ii < test1.bytes()/sizeof(float); ii++)
		test1._m_data[ii] = ii;
	
	NDArray* testp = &test1;
	clock_t t;
	
	cerr << "Dimensions:" << testp->ndim() << endl;

	double total = 0;
	t = clock();
	for(size_t zz=0; zz < testp->dim(2); zz++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t xx=0; xx < testp->dim(0); xx++) {
				total += testp->dbl({xx,yy,zz});
//				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
	t = clock()-t;
	std::cout << "zyx: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
    
	t = clock();
	for(size_t xx=0; xx < testp->dim(0); xx++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t zz=0; zz < testp->dim(2); zz++) {
				total += testp->dbl({xx,yy,zz});
//				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
	t = clock()-t;
	std::cout << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
}
