#include "ndarray.h"
#include <iostream>
#include <set>
#include <map>

using namespace std;

struct Less
{
bool operator()(const tuple<int,int,int>& lhs, const tuple<int,int,int>& rhs)
{
	if(std::get<0>(lhs) < std::get<0>(rhs))
		return true;
	if(std::get<0>(lhs) > std::get<0>(rhs))
		return false;

	if(std::get<1>(lhs) < std::get<1>(rhs))
		return true;
	if(std::get<1>(lhs) > std::get<1>(rhs))
		return false;
	
	if(std::get<2>(lhs) < std::get<2>(rhs))
		return true;
	if(std::get<2>(lhs) > std::get<2>(rhs))
		return false;

	return false;
}
};

inline
int64_t clamp(int64_t ii, int64_t low, int64_t high)
{
	return std::min<int64_t>(high, std::max<int64_t>(ii,low));
}


int main()
{
	// store 
	map<tuple<int,int,int>, float, Less> img1;
	map<tuple<int,int,int>, float, Less> img2;

	////////////////////////////////////////////////////////////////////////////////////
	cerr << "3D Test" << endl;
	size_t dim3[] = {13, 17, 13};
	NDArrayStore<3, float> array1(dim3);
	NDArrayStore<3, float> array2(dim3);

	cerr << "Filling..." << endl;
	for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
				double val = rand()/(double)RAND_MAX;
				array1.setD(val, {ii, jj, kk});
				img1[make_tuple<int,int,int>(ii,jj,kk)] = val;
				
				if(array1.getD({ii, jj, kk}) != 
						img1[make_tuple<int,int,int>(ii,jj,kk)]) {
					cerr << "Error difference between map and array" << endl;
					cerr << ii << "," << jj << "," << kk << endl;
					cerr << array1.getD({ii, jj, kk}) << " vs. " << 
								img1[make_tuple<int,int,int>(ii,jj,kk)]  << endl;
					return -1;
				}
			}
		}
	}
	
	cerr << "Comparing Set Values..." << endl;
	for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
				if(array1.getD({ii, jj, kk}) != 
						img1[make_tuple<int,int,int>(ii,jj,kk)]) {
					cerr << "Error difference between map and array" << endl;
					cerr << ii << "," << jj << "," << kk << endl;
					cerr << array1.getD({ii, jj, kk}) << " vs. " << 
								img1[make_tuple<int,int,int>(ii,jj,kk)]  << endl;
					return -1;
				}
			}
		}
	}
	
	int64_t radius = 2;
	cerr << "Kernel..." << endl;
	for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
				double sum = 0;
				double n = 0;
				for(int64_t tt=-radius; tt<=radius ; tt++) {
					for(int64_t uu=-radius; uu<=radius ; uu++) {
						for(int64_t vv=-radius; vv<=radius ; vv++) {
							int64_t it = clamp(ii+tt, 0, array1._m_dim[0]-1);
							int64_t ju = clamp(jj+uu, 0, array1._m_dim[1]-1);
							int64_t kv = clamp(kk+vv, 0, array1._m_dim[2]-1);
							sum += img1[make_tuple<int,int,int>(it,ju,kv)];
							n++;
						}
					}
				}
				img2[make_tuple<int,int,int>(ii,jj,kk)] = sum/n;
			}
		}
	}
	
	cerr << "Kernel..." << endl;
	for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
				double sum = 0;
				double n = 0;
				for(int64_t tt=-radius; tt<=radius ; tt++) {
					for(int64_t uu=-radius; uu<=radius ; uu++) {
						for(int64_t vv=-radius; vv<=radius ; vv++) {
							size_t it = clamp(ii+tt, 0, array1._m_dim[0]-1);
							size_t ju = clamp(jj+uu, 0, array1._m_dim[1]-1);
							size_t kv = clamp(kk+vv, 0, array1._m_dim[2]-1);
							sum += array1.getD({it, ju, kv});
							n++;
						}
					}
				}
				array2.setD(sum/n, {ii, jj, kk});
			}
		}
	}

	cerr << "Comparing..." << endl;
	for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
				if(array2.getD({ii, jj, kk}) != 
						img2[make_tuple<int,int,int>(ii,jj,kk)]) {
					cerr << "Error difference between map and array" << endl;
					cerr << ii << "," << jj << "," << kk << endl;
					cerr << ii << "," << jj << "," << kk << endl;
					cerr << array2.getD({ii, jj, kk}) << " vs. " << 
								img2[make_tuple<int,int,int>(ii,jj,kk)]  << endl;
					return -1;
				}
			}
		}
	}

}


