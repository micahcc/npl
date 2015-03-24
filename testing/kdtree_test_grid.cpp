/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file kdtree_test_grid.cpp
 *
 *****************************************************************************/

#include <iostream>
#include <random>

#include "kdtree.h"

using namespace std;
using namespace npl;

int main()
{
	// add point to tree and build
	std::cerr << "Inserting Points!" << std::endl;
	KDTree<3, 3, float, float> tree;

	std::vector<float> data(3);
	std::vector<float> point(3);
	for(size_t ii=0; ii<10; ii++) {
		for(size_t jj=0; jj<10; jj++) {
			for(size_t kk=0; kk<10; kk++) {
				data[0] = point[0] = ii;
				data[1] = point[1] = jj;
				data[2] = point[2] = kk;
				tree.insert(data, point);

			}
		}
	}
	std::cerr << "Done!" << std::endl;

	std::cerr << "Building!" << std::endl;
	tree.build();
	std::cerr << "Done" << std::endl;

	for(size_t ii=0; ii<10; ii++) {
		for(size_t jj=0; jj<10; jj++) {
			for(size_t kk=0; kk<10; kk++) {
				double treed = INFINITY;
				auto result = tree.nearest(data, treed);

				if(result->m_point[0] != ii) {
					std::cerr << "x mismatch" << std::endl;
				}
				if(result->m_point[1] != jj) {
					std::cerr << "y mismatch" << std::endl;
				}
				if(result->m_point[2] != kk) {
					std::cerr << "z mismatch" << std::endl;
				}

				if(result->m_data[0] != ii) {
					std::cerr << "dx mismatch" << std::endl;
				}
				if(result->m_data[1] != jj) {
					std::cerr << "dy mismatch" << std::endl;
				}
				if(result->m_data[2] != kk) {
					std::cerr << "dz mismatch" << std::endl;
				}
			}
		}
	}

	return 0;
}

