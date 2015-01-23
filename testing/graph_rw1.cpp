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
 * @file graph_rw1.cpp This file is a test of graph reading and writing
 ******************************************************************************/

#include <iostream>

#include "graph.h"

using namespace npl;
using namespace std;

// Tests Reading and Writing a basic Graph
int test1()
{
	// Create a Random Graph
	size_t nodes = 13;

	Graph<int> graph(nodes);
	for(size_t ii=0; ii<nodes; ii++) {
		for(size_t jj=0; jj<nodes; jj++) {
			graph(ii, jj) = ii*100+jj;
		}
	}

	// Test Reading
	graph.save("graph_rw1.bgm");
	Graph<int> lgraph1("graph_rw1.bgm");
	for(size_t ii=0; ii<nodes; ii++) {
		for(size_t jj=0; jj<nodes; jj++) {
			if(graph(ii, jj) != lgraph1(ii,jj)) {
				cerr << "Matrix Load Error" << endl;
				cerr<<"Original"<<graph(ii,jj)<<endl;
				cerr<<"Loaded"<<lgraph1(ii,jj)<<endl;
				return -12;
			}
		}
	}

	graph.save("graph_rw1.bgl", G_STORE_LIST);
	Graph<int> lgraph2("graph_rw1.bgl");
	for(size_t ii=0; ii<nodes; ii++) {
		for(size_t jj=0; jj<nodes; jj++) {
			if(graph(ii, jj) != lgraph2(ii,jj)) {
				cerr << "ListLoad Error" << endl;
				cerr<<"Original"<<graph(ii,jj)<<endl;
				cerr<<"Loaded"<<lgraph2(ii,jj)<<endl;
				return -12;
			}
		}
	}

	return 0;
}

int main()
{
	if(test1() != 0)
		return -1;
}
