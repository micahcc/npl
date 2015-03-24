/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file kdtree_test.cpp
 *
 *****************************************************************************/

#include <iostream>
#include <random>

#include "kdtree.h"

using namespace std;
using namespace npl;

int main()
{
	std::default_random_engine gen;
	std::normal_distribution<double> dist1(0,10);
	std::normal_distribution<double> dist2(0,100);
	std::normal_distribution<double> dist3(25,10);

	const size_t DIM = 4;
	const size_t DSIZE = 2;
	const size_t NUMSAMPLE = 1000;
	const size_t NUMTEST = 1000;

	// create array of data and corresponding points
	std::vector<std::vector<double>> points(NUMSAMPLE);
	std::vector<std::vector<double>> datas(NUMSAMPLE);

	std::cerr << "Inserting Points!" << std::endl;
	for(size_t ii=0; ii<NUMSAMPLE; ii++) {
		points[ii].resize(DIM);
		datas[ii].resize(DSIZE);
		for(size_t jj=0; jj<DIM; jj++)
			points[ii][jj] = (dist3(gen) < 0 ? dist1(gen) : dist2(gen));


		for(size_t jj=0; jj<DSIZE; jj++)
			datas[ii][jj] = (dist3(gen) < 0 ? dist1(gen) : dist2(gen));
	}
	std::cerr << "Done!" << std::endl;

	// add point to tree and build
	std::cerr << "Inserting Points!" << std::endl;
	KDTree<DIM, DSIZE, double, double> tree;
	for(size_t ii=0; ii<NUMSAMPLE; ii++)
		tree.insert(points[ii], datas[ii]);
	std::cerr << "Done!" << std::endl;

	std::cerr << "Building!" << std::endl;
	tree.build();
	std::cerr << "Done" << std::endl;

	double perror = 0;
	double derror = 0;
	// pick random points to query, then do brute force
	std::vector<double> point(DIM);
	for(size_t ii=0; ii < NUMTEST ; ii++) {

		// choose random point
		cerr << "----------------- " << endl;
		cerr << "Testing: " << endl;
		for(size_t jj=0; jj<DIM; jj++) {
			point[jj] = (dist3(gen) < 0 ? dist1(gen) : dist2(gen));
			cerr << point[jj] << ",";
		}
		cerr << endl;

		// tree search
		double treed = INFINITY;
		auto result = tree.nearest(point, treed);

		// brute force search
		double dist = INFINITY;
		double mind = INFINITY;
		size_t mini = 0;
		for(size_t kk=0; kk<NUMSAMPLE; kk++) {
			dist = 0;
			for(size_t jj=0; jj<DIM; jj++)
				dist += (points[kk][jj]-point[jj])*(points[kk][jj]-point[jj]);

			if(dist < mind) {
				mind = dist;
				mini = kk;
			}
		}

		std::cerr << "Brute Force Verse Tree: " << treed << "|" << sqrt(mind) << endl;
		for(size_t jj=0; jj<DIM; jj++) {
			std::cerr << result->m_point[jj] << ",";
		}
		std::cerr << endl;
		for(size_t jj=0; jj<DIM; jj++) {
			std::cerr << points[mini][jj] << ",";
		}
		std::cerr << endl;

		// compare points
		perror = 0;
		derror = 0;
		for(size_t jj=0; jj<DIM; jj++)
			perror += pow(points[mini][jj]-result->m_point[jj],2);
		for(size_t jj=0; jj<DSIZE; jj++)
			derror += pow(datas[mini][jj]-result->m_data[jj],2);

		if(derror > 0 && perror > 0) {
			cerr << "ERROR! Failed to find closes point ! " << endl;
			return -1;
		}
	}

	return 0;
}
