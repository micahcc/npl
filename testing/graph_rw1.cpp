/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
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
