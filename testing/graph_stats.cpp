/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file graph_stats.cpp Test graph statistics versus known values.
 *
 ******************************************************************************/

#include <iostream>
#include "graph.h"

using namespace npl;
using namespace std;

int testPreRandom()
{
	Graph<double> g1 = Graph<double>::PreRandom();
	double assort = g1.assortativity();

	vector<double> is(g1.nodes());
	vector<double> os(g1.nodes());
	vector<double> ts = g1.strengths(is, os);
	double strength = g1.strength();

	vector<int> id(g1.nodes());
	vector<int> od(g1.nodes());
	vector<int> td = g1.degrees(id, od);
	double degree = g1.degree();

	if(assort != 1) {
		cerr<<"PreRandom assortivity should be 1 (found "<<assort<<")"<<endl;
		return -1;
	}

	if(fabs(strength-767.129) > 0.001) {
		cerr<<"PreRandom strength should be 767.129 (found "<<strength<<")"<<endl;
		return -1;
	}

	const double STRENGTHS[28] = {
		28.79472, 28.30871, 27.45537, 26.5234, 26.69699, 23.56395, 31.84857,
		29.10818, 25.9711, 27.73548, 26.30403, 28.35966, 27.16889, 26.03897,
		30.62232, 25.83918, 28.50973, 29.98089, 27.01239, 29.70356, 27.26695,
		27.56648, 26.71098, 26.12657, 28.39908, 26.62863, 25.19847, 23.68604};

	for(size_t ii=0; ii<g1.nodes(); ii++) {
		double s = STRENGTHS[ii];
		if(fabs(is[ii]-s) > 0.001) {
			cerr << "PreRandom in strength should be "<<s<<" (found "<<is[ii]<<")"<<endl;
			return -1;
		}
		if(fabs(os[ii]-s) > 0.001) {
			cerr << "PreRandom out strength should be "<<s<< " (found "<<os[ii]<<")"<<endl;
			return -1;
		}
		if(fabs(ts[ii]-2*s) > 0.01) {
			cerr << "PreRandom total strength should be "<<2*s<< " (found "<<ts[ii]<<")"<<endl;
			return -1;
		}
	}

	const double betweenness[28] = {
		22.026407, 18.645671, 14.352020, 9.055844, 8.695527, 8.255556,
		21.435281, 15.022078, 12.900433, 14.176623, 15.808297, 15.168975,
		9.457359, 10.672583, 16.200794, 11.047186, 8.951227, 14.752958,
		11.026768, 14.262338, 7.566234, 15.093218, 11.410895, 11.413276,
		12.658369, 13.757287, 9.453319, 8.733478};

	// Convert Weights to Distances
	auto betw = g1.betweenness_centrality();
	for(size_t ii=0; ii<g1.nodes(); ii++) {
		if(fabs(betw[ii] - betweenness[ii]) > 0.0001) {
			cerr<<"PreRandom betweenness centrality shouild be "<<
				betweenness[ii]<<" for node "<<ii<<" but the found value is "<<
				betw[ii]<<endl;
			return -1;
		}
	}

	return 0;
}

int testCoxeter()
{
	Graph<double> g1 = Graph<double>::Coxeter();
	double assort = g1.assortativity();

	vector<double> is(g1.nodes());
	vector<double> os(g1.nodes());
	vector<double> ts = g1.strengths(is, os);
	double strength = g1.strength();

	vector<int> id(g1.nodes());
	vector<int> od(g1.nodes());
	vector<int> td = g1.degrees(id, od);
	double degree = g1.degree();

	if(assort != 1) {
		cerr<<"Coxeter assortivity should be 1 (found "<<assort<<")"<<endl;
		return -1;
	}

	if(strength != 84) {
		cerr<<"Coxeter strength should be 84 (found "<<strength<<")"<<endl;
		return -1;
	}

	for(size_t ii=0; ii<g1.nodes(); ii++) {
		if(is[ii] != 3) {
			cerr << "Coxeter in strength should be 3 (found "<<is[ii]<<")"<<endl;
			return -1;
		}
		if(os[ii] != 3) {
			cerr << "Coxeter out strength should be 3 (found "<<os[ii]<<")"<<endl;
			return -1;
		}
		if(ts[ii] != 6) {
			cerr << "Coxeter total strength should be 3 (found "<<ts[ii]<<")"<<endl;
			return -1;
		}
	}

	if(degree != 84) {
		cerr<<"Coxeter degree should be 84 (found "<<degree<<")"<<endl;
		return -1;
	}

	for(size_t ii=0; ii<g1.nodes(); ii++) {
		if(is[ii] != 3) {
			cerr << "Coxeter in degree be 3 (found "<<id[ii]<<")"<<endl;
			return -1;
		}
		if(os[ii] != 3) {
			cerr << "Coxeter out degree should be 3 (found "<<od[ii]<<")"<<endl;
			return -1;
		}
		if(td[ii] != 6) {
			cerr << "Coxeter total strength should be 3 (found "<<td[ii]<<")"<<endl;
			return -1;
		}
	}

//	// Convert Weights to Distances
//	for(size_t ii=0; ii<g1.nodes(); ii++) {
//		for(size_t jj=0; jj<g1.nodes(); jj++) {
//			if(g1(ii,jj) == 0)
//				g1(ii,jj) = INFINITY;
//		}
//	}
//	auto betw = g1.betweenness_centrality();
//	for(size_t ii=0; ii<g1.nodes(); ii++) {
//		if(betw[ii] != 24) {
//			cerr<<"Coxeter betweenness centrality shouild be 24 for all nodes "
//				"but node "<<ii<<" has betweenness centrality of "<<betw[ii]
//				<<endl;
////			return -1;
//		}
//	}
//
	return 0;
}

int main()
{
	int ret = 0;
	ret |= testCoxeter();
	ret |= testPreRandom();


	if(ret != 0) {
		cerr<<"Error While Running Tests"<<endl;
	}
}
