#include "ndarray.h"
#include <iostream>
#include <set>
#include <map>

using namespace std;

struct Less
{
bool operator()(const tuple<int,int,int,int>& lhs, const tuple<int,int,int,int>& rhs)
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
	
	if(std::get<3>(lhs) < std::get<3>(rhs))
		return true;
	if(std::get<3>(lhs) > std::get<3>(rhs))
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
	map<tuple<int,int,int,int>, float, Less> img1;
	map<tuple<int,int,int,int>, float, Less> img2;

	////////////////////////////////////////////////////////////////////////////////////
	cerr << "4D Test" << endl;
	size_t dim4[] = {13, 7, 13, 8};
	NDArrayStore<4, float> array1(dim4);
	NDArrayStore<4, float> array2(dim4);

	cerr << "Filling..." << endl;
	for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
					double val = rand()/(double)RAND_MAX;
					array1.setDouble(val, ii, jj, kk, tt);
					img1[make_tuple<int,int,int,int>(ii,jj,kk,tt)] = val;

					if(array1.getDouble(ii, jj, kk,tt) != 
							img1[make_tuple<int,int,int>(ii,jj,kk,tt)]) {
						cerr << "Error difference between map and array" << endl;
						cerr << ii << "," << jj << "," << kk << "," << tt << endl;
						cerr << array1.getDouble(ii, jj, kk,tt) << " vs. " << 
							img1[make_tuple<int,int,int,int>(ii,jj,kk,tt)]  << endl;
						return -1;
					}
				}
			}
		}
	}
	
	cerr << "Comparing Set Values..." << endl;
	for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
					if(array1.getDouble(ii, jj, kk,tt) != 
							img1[make_tuple<int,int,int>(ii,jj,kk,tt)]) {
						cerr << "Error difference between map and array" << endl;
						cerr << ii << "," << jj << "," << kk << "," << tt << endl;
						cerr << array1.getDouble(ii, jj, kk, tt) << " vs. " << 
							img1[make_tuple<int,int,int>(ii,jj,kk, tt)]  << endl;
						return -1;
					}
				}
			}
		}
	}
	
	int64_t radius = 2;
	cerr << "Kernel..." << endl;
	for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
					double sum = 0;
					double n = 0;
					for(int64_t tt=-radius; tt<=radius ; tt++) {
						for(int64_t uu=-radius; uu<=radius ; uu++) {
							for(int64_t vv=-radius; vv<=radius ; vv++) {
								for(int64_t ww=-radius; ww<=radius ; ww++) {
									int64_t it = clamp(ii+tt, 0, array1.m_dim[0]-1);
									int64_t ju = clamp(jj+uu, 0, array1.m_dim[1]-1);
									int64_t kv = clamp(kk+vv, 0, array1.m_dim[2]-1);
									int64_t tw = clamp(tt+ww, 0, array1.m_dim[2]-1);
									sum += img1[make_tuple<int,int,int>(it,ju,kv,tw)];
									n++;
								}
							}
						}
					}
					img2[make_tuple<int,int,int>(ii,jj,kk,tt)] = sum/n;
				}
			}
		}
	}
	
	cerr << "Kernel..." << endl;
	for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
					double sum = 0;
					double n = 0;
					for(int64_t tt=-radius; tt<=radius ; tt++) {
						for(int64_t uu=-radius; uu<=radius ; uu++) {
							for(int64_t vv=-radius; vv<=radius ; vv++) {
								for(int64_t ww=-radius; ww<=radius ; ww++) {
									int64_t it = clamp(ii+tt, 0, array1.m_dim[0]-1);
									int64_t ju = clamp(jj+uu, 0, array1.m_dim[1]-1);
									int64_t kv = clamp(kk+vv, 0, array1.m_dim[2]-1);
									int64_t tw = clamp(tt+ww, 0, array1.m_dim[2]-1);
									sum += array1.getDouble(it, ju, kv, tw);
									n++;
								}
							}
						}
					}
					array2.setDouble(sum/n, ii, jj, kk, tt);
				}
			}
		}
	}
	
	cerr << "Comparing..." << endl;
	for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
					if(array2.getDouble(ii, jj, kk,tt) != 
							img2[make_tuple<int,int,int>(ii,jj,kk,tt)]) {
						cerr << "Error difference between map and array" << endl;
						cerr << ii << "," << jj << "," << kk << "," << tt << endl;
						cerr << ii << "," << jj << "," << kk << "," << tt << endl;
						cerr << array2.getDouble(ii, jj, kk, tt) << " vs. " << 
							img2[make_tuple<int,int,int>(ii,jj,kk,tt)]  << endl;
						return -1;
					}
				}
			}
		}
	}

}


