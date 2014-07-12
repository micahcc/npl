/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#include "ndarray.h"
#include "hasher.h"
#include <iostream>

#include <random>
#include <memory>
#include <unordered_map>

using namespace std;
using namespace npl;

std::default_random_engine rangen;

template <typename T>
int test1d(std::vector<size_t>& size, size_t bytes, size_t elements)
{
	std::uniform_real_distribution<T> dist(0, 100000);
	auto arr = std::make_shared<NDArrayStore<1,T>>(size);

	if(bytes != arr->bytes()) 
		return -1;
	
	if(elements != arr->elements()) 
		return -2;

	std::vector<size_t> index(1);
	unordered_map<std::vector<size_t>, double, hash_vector<size_t>> mapcomp;

	size_t count = 0;
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++, count++) {
		T val = dist(rangen);

		mapcomp[index] = val;
		arr->set_dbl(index.size(), index.data(), val);
	}

	if(count != size[0]) {
		return -3;
	}
	
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		try {
			double v = arr->get_dbl(index.size(), index.data());
			if(mapcomp.at(index) != v) {
				cerr << "Error, value mismatch" << endl;
				return -4;
			}
		} catch(...) {
			cerr << "Error, index not found" << endl;
			return -5;
		}
	}

	return 0;
}

template <typename T>
int test2d(std::vector<size_t>& size, size_t bytes, size_t elements)
{
	std::uniform_real_distribution<T> dist(0, 100000);
	auto arr = std::make_shared<NDArrayStore<2,T>>(size);

	if(bytes != arr->bytes()) 
		return -1;
	
	if(elements != arr->elements()) 
		return -2;

	std::vector<size_t> index(2);
	unordered_map<std::vector<size_t>, double, hash_vector<size_t>> mapcomp;

	size_t count = 0;
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		for(index[1] = 0; index[1] < arr->dim(1); index[1]++, count++) {
			T val = dist(rangen);

			mapcomp[index] = val;
			arr->set_dbl(index.size(), index.data(), val);
		}
	}

	if(count != elements) {
		return -3;
	}
	
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		for(index[1] = 0; index[1] < arr->dim(1); index[1]++, count++) {

			try {
				double v = arr->get_dbl(index.size(), index.data());
				if(mapcomp.at(index) != v) {
					cerr << "Error, value mismatch" << endl;
					return -4;
				}
			} catch(...) {
				cerr << "Error, index not found" << endl;
				return -5;
			}
		}
	}

	return 0;
}

template <typename T>
int test3d(std::vector<size_t>& size, size_t bytes, size_t elements)
{
	std::uniform_real_distribution<T> dist(0, 100000);
	auto arr = std::make_shared<NDArrayStore<3,T>>(size);

	if(bytes != arr->bytes()) 
		return -1;
	
	if(elements != arr->elements()) 
		return -2;

	std::vector<size_t> index(3);
	unordered_map<std::vector<size_t>, double, hash_vector<size_t>> mapcomp;

	size_t count = 0;
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		for(index[1] = 0; index[1] < arr->dim(1); index[1]++) {
			for(index[2] = 0; index[2] < arr->dim(2); index[2]++, count++) {
				T val = dist(rangen);

				mapcomp[index] = val;
				arr->set_dbl(index.size(), index.data(), val);
			}
		}
	}

	if(count != elements) {
		return -3;
	}
	
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		for(index[1] = 0; index[1] < arr->dim(1); index[1]++) {
			for(index[2] = 0; index[2] < arr->dim(2); index[2]++) {

				try {
					double v = arr->get_dbl(index.size(), index.data());
					if(mapcomp.at(index) != v) {
						cerr << "Error, value mismatch" << endl;
						return -4;
					}
				} catch(...) {
					cerr << "Error, index not found" << endl;
					return -5;
				}
			}
		}
	}

	return 0;
}

template <typename T>
int test5d(std::vector<size_t>& size, size_t bytes, size_t elements)
{
	std::uniform_real_distribution<T> dist(0, 100000);
	auto arr = std::make_shared<NDArrayStore<5,T>>(size);

	if(bytes != arr->bytes()) 
		return -1;
	
	if(elements != arr->elements()) 
		return -2;

	std::vector<size_t> index(5);
	unordered_map<std::vector<size_t>, double, hash_vector<size_t>> mapcomp;

	size_t count = 0;
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		for(index[1] = 0; index[1] < arr->dim(1); index[1]++) {
			for(index[2] = 0; index[2] < arr->dim(2); index[2]++) {
				for(index[3] = 0; index[3] < arr->dim(3); index[3]++) {
					for(index[4] = 0; index[4] < arr->dim(4); index[4]++, count++) {
						T val = dist(rangen);

						mapcomp[index] = val;
						arr->set_dbl(index.size(), index.data(), val);
					}
				}
			}
		}
	}

	if(count != elements) {
		return -3;
	}
	
	for(index[0] = 0; index[0] < arr->dim(0); index[0]++) {
		for(index[1] = 0; index[1] < arr->dim(1); index[1]++) {
			for(index[2] = 0; index[2] < arr->dim(2); index[2]++) {
				for(index[3] = 0; index[3] < arr->dim(3); index[3]++) {
					for(index[4] = 0; index[4] < arr->dim(4); index[4]++) {

						try {
							double v = arr->get_dbl(index.size(), index.data());
							if(mapcomp.at(index) != v) {
								cerr << "Error, value mismatch" << endl;
								return -4;
							}
						} catch(...) {
							cerr << "Error, index not found" << endl;
							return -5;
						}
					}
				}
			}
		}
	}

	return 0;
}

int main()
{
	std::vector<size_t> sz(1, 83);
	if(test1d<float>(sz, 83*sizeof(float), 83) < 0){
		cerr << "Error during 1d float test" << endl;
		return -1;
	}
	if(test1d<long double>(sz, 83*sizeof(long double), 83) < 0){
		cerr << "Error during 1d long doubletest" << endl;
		return -1;
	}
	
	sz.assign({31,3});
	if(test2d<float>(sz, 31*3*sizeof(float), 31*3) < 0){
		cerr << "Error during 2d float test" << endl;
		return -1;
	}
	if(test2d<long double>(sz, 31*3*sizeof(long double), 31*3) < 0){
		cerr << "Error during 2d long double test" << endl;
		return -1;
	}
	
	sz.assign({17,9,3});
	if(test3d<double>(sz, 17*9*3*sizeof(double), 17*9*3) < 0){
		cerr << "Error during 3d double test" << endl;
		return -1;
	}
	if(test3d<long double>(sz, 17*9*3*sizeof(long double), 17*9*3) < 0){
		cerr << "Error during 3d long double test" << endl;
		return -1;
	}
	
	sz.assign({7,9,8,10,6});
	if(test5d<double>(sz, 7*9*8*10*6*sizeof(double), 7*9*8*10*6) < 0){
		cerr << "Error during 5d double test" << endl;
		return -1;
	}
	if(test5d<float>(sz, 7*9*8*10*6*sizeof(float), 7*9*8*10*6) < 0){
		cerr << "Error during 5d float test" << endl;
		return -1;
	}
	return 0;
}

