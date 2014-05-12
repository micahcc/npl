#include "ndarray.h"
#include <iostream>

using namespace std;

int main()
{
	size_t dim1[] = {1024};
	NDArrayStore<1, float> test1(dim1,13);
	cerr << "Bytes: " << test1.getBytes() << endl;

	for(size_t ii = 0; ii < dim1[0]; ii++)
		test1.m_data[ii] = ii;
	
	for(size_t ii = 0; ii < dim1[0]; ii++)
		cerr << ii << ": " << test1.getDouble(ii) << endl;
	
	size_t dim2[] = {23, 117};
	NDArrayStore<2, float> test2(dim2,13);

	for(size_t ii = 0; ii < dim2[0]*dim2[1]; ii++)
		test2.m_data[ii] = ii;
	
	for(size_t ii = 0; ii < dim1[0]; ii++)
		for(size_t jj = 0; jj < dim1[1]; jj++)
			cerr << ii << "," << jj << ": " << test1.getDouble(ii,jj) << endl;
}
