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
 * @file graph.h This file contains the definition for Graph and its derived types.
 ******************************************************************************/

#ifndef GRAPH_H
#define GRAPH_H

#include "npltypes.h"

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
	Graph(std::string filename, bool typefail = true);
	Graph(size_t nodes);
	Graph(Graph&& other);
	Graph(const Graph& other);
	Graph(size_t nodes, void* data,
			std::function<void(void*)> deleter=[](void*){});
	~Graph() { m_freefunc(m_data); };

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
	void Coxeter();

	/* Statistics */
	double assortativity() const;
	double assortativity_wei() const;
	double assortativity(const std::vector<T>& idegree,
			const std::vector<T>& odegree) const;

	T strength() const;
	std::vector<T> strengths() const;
	T strengths(std::vector<T>& is, std::vector<T>& os) const;

	size_t degree() const;
	std::vector<size_t> degrees() const;
	std::vector<size_t> degrees(std::vector<size_t>& is,
			std::vector<size_t>& os) const;
private:
	size_t m_size;
	T* m_data;
    std::function<void(T*)> m_freefunc;
	std::vector<std::string> m_names;
};

} // npl
#endif

