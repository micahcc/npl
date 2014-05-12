#include "ndarray.h"
#include <iostream>
#include <set>
#include <map>

using namespace std;

inline
int64_t clamp(int64_t ii, int64_t low, int64_t high)
{
	return std::min<int64_t>(high, std::max<int64_t>(ii,low));
}


int main(int argc, char** argv)
{
	if(argc != 2) {
		cerr << "Need one argument (the size of clusters)" << endl;
		return -1;
	}

	size_t csize = atoi(argv[1]);

	////////////////////////////////////////////////////////////////////////////////////
	cerr << "4D Test" << endl;
	size_t dim4[] = {500, 500, 1, 1};
	NDArrayStore<4, float> array1(dim4,csize);
	NDArrayStore<4, float> array2(dim4,csize);

	cerr << "Filling..." << endl;
	auto t = clock();
	for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
					double val = rand()/(double)RAND_MAX;
					array1.setDouble(val, ii, jj, kk, tt);
				}
			}
		}
	}
	t = clock() - t;
	printf("Fill: %li clicks (%f seconds)\n",t,((float)t)/CLOCKS_PER_SEC);
	
	int64_t radius = 4;
	t = clock();
	for(size_t tt = 0; tt < array1.m_dim[3]; tt++) {
		for(size_t kk = 0; kk < array1.m_dim[2]; kk++) {
			for(size_t jj = 0; jj < array1.m_dim[1]; jj++) {
				for(size_t ii = 0; ii < array1.m_dim[0]; ii++) {
					double sum = 0;
					double n = 0;
					for(int64_t xx=-radius; xx<=radius ; xx++) {
						for(int64_t yy=-radius; yy<=radius ; yy++) {
							for(int64_t zz=-radius; zz<=radius ; zz++) {
								for(int64_t rr=-radius; rr<=radius ; rr++) {
									int64_t ix = clamp(ii+xx, 0, array1.m_dim[0]-1);
									int64_t jy = clamp(jj+yy, 0, array1.m_dim[1]-1);
									int64_t kz = clamp(kk+zz, 0, array1.m_dim[2]-1);
									int64_t tr = clamp(tt+rr, 0, array1.m_dim[2]-1);
									sum += array1.getDouble(ix, jy, kz, tr);
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
	t = clock() - t;
	printf("Radius %li Kernel: %li clicks (%f seconds)\n",radius, t,((float)t)/CLOCKS_PER_SEC);
}


