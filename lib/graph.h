/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file graph.h This file contains the definition for Graph and its derived types.
 ******************************************************************************/

#ifndef GRAPH_H
#define GRAPH_H

#include "npltypes.h"
#include "zlib.h"

#include <algorithm>
#include <string>
#include <vector>

namespace npl {

// Data Types:
// bit 7, 1 if floating, 0 if integer
// bit 6, 1 if complex, 0 if real
// bit 5, 1 if signed, 0 if unsigned
// bit 0-4 size in bytes
enum GraphDataT
{
	G_DT_UNKNOWN = 0,
	G_DT_UCHAR = 1, // 0000 0001
	G_DT_CHAR = 33, // 0010 0001
	G_DT_USHORT = 2,// 0000 0010
	G_DT_SHORT = 34,// 0010 0010
	G_DT_ULONG = 4, // 0000 0100
	G_DT_SLONG = 36,// 0010 0100
	G_DT_ULONGLONG = 8, // 0000 1000
	G_DT_SLONGLONG =40, // 0010 1000
	G_DT_FLOAT = 132,   // 1000 0100
	G_DT_DOUBLE = 136,  // 1000 1000
	G_DT_QUAD = 144,    // 1001 0000
	G_DT_COMPLEX_FLOAT = 228,  // 1110 0100
	G_DT_COMPLEX_DOUBLE = 232, // 1110 1000
	G_DT_COMPLEX_QUAD = 255    // 1111 1111
};

enum GraphStoreT
{
	G_STORE_UNKNOWN = 0,
	G_STORE_FULLMAT = 1,
	G_STORE_UNDIR = 2,
	G_STORE_LIST = 3
};

template <typename T>
GraphDataT getType();

std::string describeType(GraphDataT type);

template <typename T>
class Graph
{
public:
	Graph();
	Graph(std::string filename, bool typefail = true);
	Graph(size_t nodes);
	Graph(Graph&& other);
	Graph(const Graph& other);
	Graph(size_t nodes, void* data,
			std::function<void(void*)> deleter=[](void*){});
	~Graph()
	{
		m_freefunc(m_data);
	};

	void init(size_t nodes);
	void init(size_t nodes, void* data,
			std::function<void(void*)> deleter=[](void*){});

	Graph& operator=(Graph<T>&& other);

	T& operator()(size_t from, size_t to)
	{
		return m_data[m_size*from + to];
	};

	const T& operator()(size_t from, size_t to) const
	{
		return m_data[m_size*from + to];
	};

	size_t nodes() const { return m_size; };

	const std::string& name(size_t ii) const {return m_names[ii]; };
	std::string& name(size_t ii) { return m_names[ii]; };

	void load(std::string filename, bool typefail = true);
	void save(std::string filename, GraphStoreT store = G_STORE_FULLMAT);

	static GraphDataT type() { return getType<T>(); };
	static std::string typestr() { return typeid(T).name(); };

	void normalize();

	/* Famouse Graphs */
	static Graph<T> Coxeter();
	static Graph<T> PreRandom();

	/* Statistics */
	double assortativity() const;
	double assortativity_wei() const;

	T strength() const;
	std::vector<T> strengths() const;
	std::vector<T> strengths(std::vector<T>& is, std::vector<T>& os) const;

	int degree() const;
	std::vector<int> degrees() const;
	std::vector<int> degrees(std::vector<int>& is, std::vector<int>& os) const;

	std::vector<int> betweenness_centrality() const;
	std::vector<int> betweenness_centrality_next() const;
	void shortest(Graph<T>& sdist) const;
	void shortest(Graph<int>& next, Graph<T>& sdist) const;

private:
	size_t m_size;
	T* m_data;
    std::function<void(T*)> m_freefunc;
	std::vector<std::string> m_names;

	void writeCSV(gzFile gz, GraphStoreT store);
	void writeNPL(gzFile gz, GraphStoreT store);
};

} // npl
#endif

