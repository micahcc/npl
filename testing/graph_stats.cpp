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


	if(ret != 0) {
		cerr<<"Error While Running Tests"<<endl;
	}
}
