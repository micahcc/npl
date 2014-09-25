/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file img_kernel_test.cpp
 *
 *****************************************************************************/
#include <cassert>
#include <iostream>
#include <list>
#include <random>
#include <unordered_map>
#include "hasher.h"
#include "iterators.h"
#include "mrimage.h"
#include "mrimage_utils.h"

using namespace std;
using namespace npl;

std::default_random_engine rangen;

template <typename T>
void fillRandom(PixelT type, std::vector<size_t>& size,
		shared_ptr<MRImage>& testimg,
		unordered_map<std::vector<int64_t>, T, hash_vector<int64_t>>& mp)
{
	std::uniform_real_distribution<double> dist(0, 100000);
	testimg = createMRImage(size, type);
	OrderIter<double> it(testimg);
	assert(it.isBegin());
	assert(!it.isEnd());
	std::vector<int64_t> index(size.size());
	for(it.goBegin(); !it.eof(); ++it) {
		T val = (T)dist(rangen);
		it.index(index.size(), index.data());

		mp[index] = val;
		it.set(val);
	}
	assert(!it.isBegin());
	assert(it.isEnd());
}

int testRadius(std::vector<size_t> size, size_t radius)
{
	shared_ptr<MRImage> testimg;
	unordered_map<std::vector<int64_t>, double, hash_vector<int64_t>> mp;
	unordered_map<std::vector<int64_t>, size_t, hash_vector<int64_t>> count;
	fillRandom<double>(FLOAT64, size, testimg, mp);
	cerr << *testimg << endl << "radius " << radius << endl;
	
	std::vector<int64_t> index(size.size());
	KernelIter<double> it(testimg);
	it.setRadius(radius);
	for(it.goBegin(); !it.eof(); ++it) {
		double v1, v2, v3;
		for(size_t ii=0; ii<it.ksize(); ii++) {
			it.indexK(ii, index.size(), index.data(), false);
			if(mp.count(index) > 0) {
				v1 = it[ii];
				v2 = it.getK(ii);
				v3 = mp[index];
				if(v1 != v2) {
					cerr << "Kernel Offset Error!" << std::endl;
					return -1;
				} else if(v1 != v3){
					cerr << "Map Difference!" << std::endl;
					return -1;
				}

			}
		}
	}

	return 0;
}

int testRadius2(std::vector<size_t> size)
{
	shared_ptr<MRImage> testimg;
	unordered_map<std::vector<int64_t>, double, hash_vector<int64_t>> mp;
	unordered_map<std::vector<int64_t>, size_t, hash_vector<int64_t>> count;
	fillRandom<double>(FLOAT64, size, testimg, mp);
	
	std::list<std::vector<int64_t>> shouldget ({
		{-2,-1, 0}, {-2, 0, 0}, {-2, 1, 0},
		{-1,-1, 0}, {-1, 0, 0}, {-1, 1, 0},
		{ 0,-1, 0}, { 0, 0, 0}, { 0, 1, 0},
		{ 1,-1, 0}, { 1, 0, 0}, { 1, 1, 0},
		{ 2,-1, 0}, { 2, 0, 0}, { 2, 1, 0}});

	std::vector<int64_t> index(size.size());
	std::vector<int64_t> center(size.size());
	KernelIter<double> it(testimg);
	it.setRadius({2,1,0});
	for(it.goBegin(); !it.eof(); ++it) {
		auto check = shouldget;
		double v1, v2, v3;
		for(size_t ii=0; ii<it.ksize(); ii++) {
			it.indexK(ii, index.size(), index.data(), false);
			
			//check value
			if(mp.count(index) > 0) {
				v1 = it[ii];
				v2 = it.getK(ii);
				v3 = mp[index];
				if(v1 != v2) {
					cerr << "Kernel Offset Error!" << std::endl;
					return -1;
				} else if(v1 != v3){
					cerr << "Map Difference!" << std::endl;
					return -1;
				}

			}
			
			// convert index to offset
			it.indexC(center.size(), center.data());
			for(size_t j=0; j<index.size(); j++)
				index[j] -= center[j];

			// check to see if the offset is in check, and if it is, erase it
			bool oneeq = false;
			for(auto it = check.begin(); it != check.end(); ) {
				bool eq = true;
				if(it->size() != index.size()){
					cerr << "Index Size Mismatch" << endl;
					return -1;
				}
				for(size_t jj = 0; jj<index.size(); jj++) {
					if((*it)[jj] != index[jj]) {
						eq = false;
					}
				}
				if(!eq) {
					++it;
				} else  {
					it = check.erase(it);
					oneeq = true;
				}
			}

			if(!oneeq) {
				cerr << "No match for for offset found!" << endl;
				return -1;
			}
		}

		if(check.size() > 0) {
			cerr << "Not all radius neighbors hit!" << endl;
			return -1;
		}
	}

	return 0;
}

int testWindow(std::vector<size_t> size)
{
	shared_ptr<MRImage> testimg;
	unordered_map<std::vector<int64_t>, double, hash_vector<int64_t>> mp;
	unordered_map<std::vector<int64_t>, size_t, hash_vector<int64_t>> count;
	fillRandom<double>(FLOAT64, size, testimg, mp);
	
	std::list<std::vector<int64_t>> shouldget({
			{-2, 0,-2},{-2, 0,-1},{-2, 0, 0},{-2, 0, 1},{-2, 0, 2},
			{-2, 1,-2},{-2, 1,-1},{-2, 1, 0},{-2, 1, 1},{-2, 1, 2},
			{-1, 0,-2},{-1, 0,-1},{-1, 0, 0},{-1, 0, 1},{-1, 0, 2},
			{-1, 1,-2},{-1, 1,-1},{-1, 1, 0},{-1, 1, 1},{-1, 1, 2},
			{ 0, 0,-2},{ 0, 0,-1},{ 0, 0, 0},{ 0, 0, 1},{ 0, 0, 2},
			{ 0, 1,-2},{ 0, 1,-1},{ 0, 1, 0},{ 0, 1, 1},{ 0, 1, 2}
	});

	std::vector<int64_t> index(testimg->ndim());
	std::vector<int64_t> center(testimg->ndim());
	KernelIter<double> it(testimg);
	it.setWindow({{-2,0},{0,1},{-2,2}});
	for(it.goBegin(); !it.eof(); ++it) {
		auto check = shouldget;
		double v1, v2, v3;
		for(size_t ii=0; ii<it.ksize(); ii++) {
			it.indexK(ii, index.size(), index.data(), false);
			
			//check value
			if(mp.count(index) > 0) {
				v1 = it[ii];
				v2 = it.getK(ii);
				v3 = mp[index];
				if(v1 != v2) {
					cerr << "Kernel Offset Error!" << std::endl;
					return -1;
				} else if(v1 != v3){
					cerr << "Map Difference!" << std::endl;
					return -1;
				}

			}
			
			// convert index to offset
			it.indexC(center.size(), center.data());
			for(size_t j=0; j<index.size(); j++)
				index[j] -= center[j];

			// check to see if the offset is in check, and if it is, erase it
			bool oneeq = false;
			for(auto it = check.begin(); it != check.end(); ) {
				bool eq = true;
				if(it->size() != index.size()){
					cerr << "Index Size Mismatch" << endl;
					return -1;
				}
				for(size_t jj = 0; jj<index.size(); jj++) {
					if((*it)[jj] != index[jj]) {
						eq = false;
					}
				}
				if(!eq) {
					++it;
				} else  {
					it = check.erase(it);
					oneeq = true;
				}
			}

			if(!oneeq) {
				cerr << "No match for for offset found!" << endl;
				return -1;
			}
		}

		if(check.size() > 0) {
			cerr << "Not all radius neighbors hit!" << endl;
			return -1;
		}
	}

	return 0;
}

//int testBound(std::vector<size_t> size)
//{
//	shared_ptr<MRImage> testimg;
//	
//	unordered_map<std::vector<int64_t>, double, hash_vector<int64_t>> mp;
//	unordered_map<std::vector<int64_t>, size_t, hash_vector<int64_t>> count;
//	fillRandom<double>(FLOAT64, size, testimg, mp);
//
//	std::vector<int64_t> index;
//	KernelIter<double> it(testimg);
//	it.setRadius(2);
//	
//	KernelIter<double> wit(testimg);
//	wit.setRadius({1,0,2});
//	for(it.goBegin(); !it.eof(); ++it) {
//		double v1, v2, v3;
//		
//		for(size_t ii=0; ii<it.ksize(); ii++) {
//			index = it.offset_index(ii, false);
//			if(mp.count(index) > 0) {
//				v1 = it[ii];
//				v2 = it.offset(ii);
//				v3 = mp[index];
//				if(v1 != v2) {
//					cerr << "Kernel Offset Error!" << std::endl;
//					return -1;
//				} else if(v1 != v3){
//					cerr << "Map Difference!" << std::endl;
//					return -1;
//				}
//
//			}
//		}
//	}
//
//	return 0;
//}


int main()
{
	if(testRadius({10,10,10}, 0) != 0) {
		return -1;
	}
	if(testRadius({10,10,10}, 1) != 0) {
		return -1;
	}
	if(testRadius({10,10,10}, 2) != 0) {
		return -1;
	}

	if(testRadius2({10,10,10}) != 0) {
		return -1;
	}
	
	if(testWindow({10,10,10}) != 0) {
		return -1;
	}
}

//
//	
//	std::vector<std::vector<int64_t>> covered({
//			{-1,-2,-2},{-1,-2,-1},{-1,-2, 0},{-1,-2, 1},{-1,-2, 2},
//			{-1,-1,-2},{-1,-1,-1},{-1,-1, 0},{-1,-1, 1},{-1,-1, 2},
//			{-1, 0,-2},{-1, 0,-1},{-1, 0, 0},{-1, 0, 1},{-1, 0, 2},
//			{-1, 1,-2},{-1, 1,-1},{-1, 1, 0},{-1, 1, 1},{-1, 1, 2},
//			{-1, 2,-2},{-1, 2,-1},{-1, 2, 0},{-1, 2, 1},{-1, 2, 2},
//
//			{ 0,-2,-2},{ 0,-2,-1},{ 0,-2, 0},{ 0,-2, 1},{ 0,-2, 2},
//			{ 0,-1,-2},{ 0,-1,-1},{ 0,-1, 0},{ 0,-1, 1},{ 0,-1, 2},
//			{ 0, 0,-2},{ 0, 0,-1},{ 0, 0, 0},{ 0, 0, 1},{ 0, 0, 2},
//			{ 0, 1,-2},{ 0, 1,-1},{ 0, 1, 0},{ 0, 1, 1},{ 0, 1, 2},
//			{ 0, 2,-2},{ 0, 2,-1},{ 0, 2, 0},{ 0, 2, 1},{ 0, 2, 2},
//			
//			{ 1,-2,-2},{ 1,-2,-1},{ 1,-2, 0},{ 1,-2, 1},{ 1,-2, 2},
//			{ 1,-1,-2},{ 1,-1,-1},{ 1,-1, 0},{ 1,-1, 1},{ 1,-1, 2},
//			{ 1, 0,-2},{ 1, 0,-1},{ 1, 0, 0},{ 1, 0, 1},{ 1, 0, 2},
//			{ 1, 1,-2},{ 1, 1,-1},{ 1, 1, 0},{ 1, 1, 1},{ 1, 1, 2},
//			{ 1, 2,-2},{ 1, 2,-1},{ 1, 2, 0},{ 1, 2, 1},{ 1, 2, 2},
//	});
//
//	std::vector<std::vector<int64_t>> remain({
//			{-2,-2,-2},{-2,-2,-1},{-2,-2, 0},{-2,-2, 1},{-2,-2, 2},
//			{-2,-1,-2},{-2,-1,-1},{-2,-1, 0},{-2,-1, 1},{-2,-1, 2},
//			{-2, 0,-2},{-2, 0,-1},{-2, 0, 0},{-2, 0, 1},{-2, 0, 2},
//			{-2, 1,-2},{-2, 1,-1},{-2, 1, 0},{-2, 1, 1},{-2, 1, 2},
//			{-2, 2,-2},{-2, 2,-1},{-2, 2, 0},{-2, 2, 1},{-2, 2, 2},
//
//			{-1,-2,-2},{-1,-2,-1},
//			{-1,-1,-2},{-1,-1,-1},
//			{-1, 0,-2},{-1, 0,-1},
//
//			{ 0,-2,-2},{ 0,-2,-1},
//			{ 0,-1,-2},{ 0,-1,-1},
//			{ 0, 0,-2},{ 0, 0,-1},
//			
//			{ 1,-2,-2},{ 1,-2,-1},
//			{ 1,-1,-2},{ 1,-1,-1},
//			{ 1, 0,-2},{ 1, 0,-1},
//			
//			{ 2,-2,-2},{ 2,-2,-1},{ 2,-2, 0},{ 2,-2, 1},{ 2,-2, 2},
//			{ 2,-1,-2},{ 2,-1,-1},{ 2,-1, 0},{ 2,-1, 1},{ 2,-1, 2},
//			{ 2, 0,-2},{ 2, 0,-1},{ 2, 0, 0},{ 2, 0, 1},{ 2, 0, 2},
//			{ 2, 1,-2},{ 2, 1,-1},{ 2, 1, 0},{ 2, 1, 1},{ 2, 1, 2},
//			{ 2, 2,-2},{ 2, 2,-1},{ 2, 2, 0},{ 2, 2, 1},{ 2, 2, 2}
//	});
