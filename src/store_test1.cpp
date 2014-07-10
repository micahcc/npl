#include "ndarray.h"
#include <iostream>
#include <set>

using namespace std;
using namespace npl;

int main()
{
	set<size_t> check;

	NDArrayStore<1, float> test1({1024});
	cerr << "Bytes: " << test1.bytes() << endl;

	for(size_t ii = 0; ii < test1._m_dim[0]; ii++) {
		test1._m_data[ii] = ii;
		check.insert(ii);
	}
	
	for(size_t ii = 0; ii < test1._m_dim[0]; ii++) {
		double val = test1.dbl({ii});
		cerr << ii << "(" << test1.getAddr({ii}) << "): " << val << endl;
		size_t er = check.erase(val);
		if(er != 1) {
			cerr << "Should have erased 1 but erased " << er << endl;
			return -1;
		}
	}
	
	if(check.size() != 0) {
		cerr << "Should have erased all but " << check.size() << "remain" << endl;
		return -1;
	}
	
	////////////////////////////////////////////////////////////////////////////////////
	cerr << "2D Test" << endl;
	NDArrayStore<2, float> test2({23, 117});

	cerr << "Filling..." << endl;
	for(size_t ii = 0; ii < test2._m_dim[0]*test2._m_dim[1]; ii++) {
		test2._m_data[ii] = ii;
		check.insert(ii);
	}
	
	cerr << "Checking..." << endl;
	for(size_t ii = 0; ii < test2._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < test2._m_dim[1]; jj++) {
			double val = test2.dbl({ii, jj});
			cerr << ii << "," << jj << " (" << test2.getAddr({ii, jj}) << "): " 
				<< val << endl;
			
			size_t er = check.erase(val);
			if(er != 1) {
				cerr << "Should have erased 1 but erased " << er << endl;
				return -1;
			}
		}
	}

	if(check.size() != 0) {
		cerr << "Should have erased all but " << check.size() << "remain" << endl;
		return -1;
	}
	
	////////////////////////////////////////////////////////////////////////////////////
	cerr << "3D Test" << endl;
	NDArrayStore<3, float> test3({23, 17, 23});

	cerr << "Filling..." << endl;
	for(size_t ii = 0; ii < test3._m_dim[0]*test3._m_dim[1]*test3._m_dim[2]; ii++) {
		test3._m_data[ii] = ii;
		check.insert(ii);
	}
	
	cerr << "Checking..." << endl;
	for(size_t ii = 0; ii < test3._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < test3._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < test3._m_dim[2]; kk++) {
				double val = test3.dbl({ii, jj, kk});
				cerr << ii << "," << jj << " (" << test3.getAddr({ii, jj, kk}) << "): " 
					<< val << endl;

				size_t er = check.erase(val);
				if(er != 1) {
					cerr << "Should have erased 1 but erased " << er << endl;
					return -1;
				}
			}
		}
	}

	if(check.size() != 0) {
		cerr << "Should have erased all but " << check.size() << "remain" << endl;
		return -1;
	}
	check.clear();
}

