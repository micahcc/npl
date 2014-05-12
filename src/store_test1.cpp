#include "ndarray.h"
#include <iostream>
#include <set>

using namespace std;

int main()
{
	set<size_t> check;
	size_t sz = 0;

	size_t dim1[] = {1024};
	NDArrayStore<1, float> test1(dim1,13);
	cerr << "Bytes: " << test1.getBytes() << endl;

	for(size_t ii = 0; ii < dim1[0]; ii++) {
		test1.m_data[ii] = ii;
		check.insert(ii);
	}
	
	for(size_t ii = 0; ii < dim1[0]; ii++) {
		double val = test1.getDouble(ii);
		cerr << ii << "(" << test1.getAddr(ii) << "): " << val << endl;
		size_t er = check.erase(val);
		if(er != 1) {
			cerr << "Should have erased 1 but erased " << er << endl;
			return -1;
		}
	}
	
	sz = 1;
	for(size_t ii = 0; ii < 3; ii++) 
		sz *= test1.m_pdim[ii];
	size_t diff = sz;
	sz = 1;
	for(size_t ii = 0; ii < 2; ii++) 
		sz *= test1.m_dim[ii];
	diff -= sz;

	if(check.size() != diff) {
		cerr << "Should have erased all but " << sz << " but " << diff << "remain" << endl;
		return -1;
	}
	check.clear();
	
	////////////////////////////////////////////////////////////////////////////////////
	cerr << "2D Test" << endl;
	size_t dim2[] = {23, 117};
	NDArrayStore<2, float> test2(dim2,13);

	cerr << "Filling..." << endl;
	for(size_t ii = 0; ii < test2.m_pdim[0]*test2.m_pdim[1]; ii++) {
		test2.m_data[ii] = ii;
		check.insert(ii);
	}
	
	cerr << "Checking..." << endl;
	for(size_t ii = 0; ii < test2.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < test2.m_dim[1]; jj++) {
			double val = test2.getDouble(ii, jj);
			cerr << ii << "," << jj << " (" << test2.getAddr(ii, jj) << "): " 
				<< val << endl;
			
			size_t er = check.erase(val);
			if(er != 1) {
				cerr << "Should have erased 1 but erased " << er << endl;
				return -1;
			}
		}
	}

	sz = 1;
	for(size_t ii = 0; ii < 2; ii++) 
		sz *= test2.m_pdim[ii];
	diff = sz;
	sz = 1;
	for(size_t ii = 0; ii < 2; ii++) 
		sz *= test2.m_dim[ii];
	diff -= sz;

	if(check.size() != diff) {
		cerr << "Should have erased all but " << sz << " but " << diff << "remain" << endl;
		return -1;
	}
	check.clear();
	
	////////////////////////////////////////////////////////////////////////////////////
	cerr << "3D Test" << endl;
	size_t dim3[] = {23, 17, 23};
	NDArrayStore<3, float> test3(dim3,19);

	cerr << "Filling..." << endl;
	for(size_t ii = 0; ii < test3.m_pdim[0]*test3.m_pdim[1]*test3.m_pdim[2]; ii++) {
		test3.m_data[ii] = ii;
		check.insert(ii);
	}
	
	cerr << "Checking..." << endl;
	for(size_t ii = 0; ii < test3.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < test3.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < test3.m_dim[2]; kk++) {
				double val = test3.getDouble(ii, jj, kk);
				cerr << ii << "," << jj << " (" << test3.getAddr(ii, jj, kk) << "): " 
					<< val << endl;

				size_t er = check.erase(val);
				if(er != 1) {
					cerr << "Should have erased 1 but erased " << er << endl;
					return -1;
				}
			}
		}
	}

	sz = 1;
	for(size_t ii = 0; ii < 3; ii++) 
		sz *= test3.m_pdim[ii];
	diff = sz;
	sz = 1;
	for(size_t ii = 0; ii < 3; ii++) 
		sz *= test3.m_dim[ii];
	diff -= sz;

	if(check.size() != diff) {
		cerr << "Should have erased all but " << sz << " but " << diff << "remain" << endl;
		return -1;
	}
	check.clear();
}
