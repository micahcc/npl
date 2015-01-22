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
#include <vectors>

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

enum GraphStore_T
{
	G_STORE_UNKNOWN = 0,
	G_STORE_FULLMAT = 1,
	G_STORE_UNDIR = 2,
	G_STORE_LIST = 3
};

string describeType(GraphDataT type);

template <typename T>
class Graph
{
	Graph(std::string filename)
	{
		load(filename);
	};
	Graph(size_t nodes);
	Graph(Graph&& other);
	Graph(const Graph& other);;
	Graph(size_t nodes, void* data, std::function<void(void*)> deleter);

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

	void load(std::string filename, bool typefail);
	void save(std::string filename);

	GraphDataT type() constexpr { return getType<T>(); };
	string typestr() { return typeid(T).name(); };

private:
	size_t m_size;
	T* m_data;
    std::function<void(void*)> m_freefunc;
	std::vector<string> m_names;
};

template <typename T>
GraphDataT getType() constexpr
{
	if(typeid(T) == typeid(unsigned char))
		return G_DT_UCHAR;
	else if(typeid(T) == typeid(char))
		return G_DT_CHAR;
	else if(typeid(T) == typeid(uint16_t))
		return G_DT_USHORT;
	else if(typeid(T) == typeid(int16_t))
		return G_DT_SHORT;
	else if(typeid(T) == typeid(uint32_t))
		return G_DT_ULONG;
	else if(typeid(T) == typeid(int32_t))
		return G_DT_SLONG;
	else if(typeid(T) == typeid(uint64_t))
		return G_DT_ULONGLONG;
	else if(typeid(T) == typeid(int64_t))
		return G_DT_SLONGLONG;
	else if(typeid(T) == typeid(float))
		return G_DT_FLOAT;
	else if(typeid(T) == typeid(double))
		return G_DT_DOUBLE;
	else if(typeid(T) == typeid(long double))
		return G_DT_QUAD;
	else if(typeid(T) == typeid(complex<float>))
		return G_DT_COMPLEX_FLOAT;
	else if(typeid(T) == typeid(complex<double>))
		return G_DT_COMPLEX_DOUBLE;
	else if(typeid(T) == typeid(complex<long double>))
		return G_DT_COMPLEX_QUAD;
	return G_DT_UNKNOWN;
};

template <typename T>
Graph::Graph(size_t nodes)
{
	m_data = new T[nodes*nodes];
	m_freefunc = [](void* ptr) { delete[] ptr; };
	m_names.resize(nodes);
}

template <typename T>
Graph::Graph(size_t nodes, void* data,
        std::function<void(void*)> deleter)
{
	m_data = data
	m_freefunc = deleter;
	m_names.resize(nodes);
}

template <typename T>
Graph::Graph(Graph<T>&& other)
{
	m_data = other.m_data;
	m_size = other.m_size;
	m_freefunc = std::move(other.m_freefunc);
	m_names = std::move(other.m_names);

	other.m_data = NULL;
	other.m_freefunc = [](void*){ };
}

template <typename T>
Graph::Graph(const Graph<T>& other)
{
	m_size = other.m_size;
	m_data = new T[m_size*m_size];
	m_names = other.m_names;
	std::copy(other.m_data, other.m_data+sizeof(T)*m_size*m_size, m_data);
	m_freefunc = [](void* ptr) { delete[] ptr; };
}

template <typename T>
void Graph::save(std::string filename, GraphStore_T store)
{
	size_t tmp = 0;
	gzFile gz;
	std::string mode = "wb";

	// remove .gz to find the "real" format,
	std::string nogz;
	if(filename.size() >= 3 && filename.substr(filename.size()-3, 3) == ".gz") {
		nogz = filename.substr(0, filename.size()-3);
	} else {
		// if no .gz, then make encoding "transparent" (plain)
		nogz = filename;
		mode += 'T';
	}

	// go ahead and open
	gz = gzopen(filename.c_str(), mode.c_str());
	if(!gz) {
		std::cerr << "Could not open " << filename << " for writing!" << std::endl;
		return -1;
	}

	if(store == G_STORE_FULLMAT) {
		// assymetric (Directed) Adjacency Matrix Type
		// Name     Bytes    Description
		// Magic    0-7      Magic "NPLGDMAT"
		// NumNode  8-15     size_t # of nodes
		// OffMat   16-23    size_t offset from start of file to first matrix
		//                   element (in bytes)
		// OffMeta  24-31    size_t offset from start of file to metadata (in
		//                   bytes)
		// BytePer  32-39    size_t Bytes per data element
		// datatype 40-43    Unsigned char Data type: (See GRAPH_DATATYPES)
		// RESERVE  44-511   Reserved for future
		// MetaData OffMeta- Node Metadata pairs (size_t sz, followed by string
		//                   of bytes of length sz)
		// MatData  OffMat-  Matrix Data, Full Matrix, where source is
		//                   determined by the row, destination of edge is the
		//                   column. Data should be stored in ROW MAJOR order.
		//                   So all the connections leaving node 0 will be
		//                   stored in the first NumNode*BytePer bytes.

		// write header
		char magic[8] = "NPLGDMAT";
		gzwrite(gz, magic, 8); // Magic
		gzwrite(gz, &nodes(), 8); // NumNode

		size_t offmat = 512; // Header Size + All String Data
		for(size_t ii=0; ii<nodes(); ii++)
			offmat += sizeof(size_t)+name(ii).size();
		gzwrite(gz, &offmat, sizeof(size_t)); // offMat

		// Figure Out Offset to Matrix (and Align)
		size_t offmeta = 512;
		gzwrite(gz, &offmeta, sizeof(size_t)); // OffMeta

		// write out bytes per element
		tmp = sizeof(T);
		gzwrite(gz, &tmp, sizeof(size_t)); // byteper

		// write out datatype
		int tmptype = (int)type();
		gzwrite(gz, &tmptype, 4);

		// write filler
		char filler[468];
		gzwrite(gz, filler, sizeof(fillter));

		// write out metadata
		for(size_t ii=0; ii<nodes(); ii++) {
			tmp = name(ii).size();
			gzwrite(gz, &tmp, sizeof(size_t));
			gzwrite(gz, name(ii).c_str(), tmp);
		}

		// Write Data
		gzwrite(gz, m_data, sizeof(T)*nodes()*nodes());
	} else if(store == G_STORE_LIST) {

		// Adjacency List Type (no distnction between directed or undirected)
		// Name     Bytes    Description
		// Magic    0-7      Magic "NPLGLIST"
		// NumNode  8-15     size_t # of nodes
		// OffList  16-23    size_t offset from start of file to first list
		//                   element (in bytes)
		// OffMeta  24-31    size_t offset from start of file to metadata (in
		//                   bytes)
		// BytePer  32-39    size_t Bytes per data element, should be
		//                   sizeof(indtype)*2+sizeo(datatype)
		// indtype  40-43    Unsigned char Index type: (See GRAPH_DATATYPES)
		// datatype 44-47    Unsigned char Data type: (See GRAPH_DATATYPES)
		// RESERVE  48-511   Reserved for future
		// MetaData OffMeta- Node Metadata pairs (size_t sz, followed by string
		//                   of bytes of length sz)
		// MatData  OffList- List Data, where nodes are stored as triplets of
		//                   indtype indtype datatype
		typedef int64_t IndType;
		T thresh = std::numeric_limits<T>::epsilon();

		// write header
		char magic[8] = "NPLGLIST";
		gzwrite(gz, magic, 8); // Magic
		gzwrite(gz, &nodes(), 8); // NumNode

		size_t offlist = 512; // Header Size + All String Data
		for(size_t ii=0; ii<nodes(); ii++)
			offlist += sizeof(size_t)+name(ii).size();
		gzwrite(gz, &offlist, sizeof(size_t));

		size_t offmeta = 512;
		gzwrite(gz, &offmeta, sizeof(size_t)); // OffMeta

		// write out bytes per element
		tmp = 2*sizeof(IndType)+sizeof(T);
		gzwrite(gz, &tmp, sizeof(size_t)); // byteper

		// write out IndType
		int tmptype = (int)getType<IndType>();
		gzwrite(gz, &type, 4);

		// write out datatype
		tmptype = (int)type();
		gzwrite(gz, &type, 4);

		// write filler
		char filler[464];
		gzwrite(gz, filler, sizeof(fillter));

		// write out metadata
		for(size_t ii=0; ii<nodes(); ii++) {
			tmp = name(ii).size();
			gzwrite(gz, &tmp, sizeof(size_t));
			gzwrite(gz, name(ii).c_str(), tmp);
		}

		// Write Data
		for(IndType ii=0; ii<nodes(); ii++) {
			for(IndType jj=0; jj<nodes(); jj++) {
				T v = (*this)(ii, jj);
				if(v > thresh) {
					gzwrite(gz, &ii, sizeof(IndType));
					gzwrite(gz, &jj, sizeof(IndType));
					gzwrite(gz, &v, sizeof(T));
				}
			}
		}
	}

	gzclose(gz);
	return 0;
}

template <typename T>
void Graph::load(std::string filename)
{
	size_t tmp = 0;
	gzFile gz = gzopen(filename.c_str(), "rb");
	if(!gz) {
		std::cerr << "Could not open " << filename << " for writing!" << std::endl;
		return -1;
	}

	char magic[8];
	size_t numnode;
	size_t offdata;
	size_t offmeta;
	size_t bytesper;
	int datatype;

	gzread(gz, magic, 8);
	if(strncmp(magic, "NPLGDMAT", 8) == 0) {
		// assymetric (Directed) Adjacency Matrix Type
		// Name     Bytes    Description
		// Magic    0-7      Magic "NPLGDMAT"
		// NumNode  8-15     size_t # of nodes
		// OffMat   16-23    size_t offset from start of file to first matrix
		//                   element (in bytes)
		// OffMeta  24-31    size_t offset from start of file to metadata (in
		//                   bytes)
		// BytePer  32-39    size_t Bytes per data element
		// datatype 40-43    Unsigned char Data type: (See GRAPH_DATATYPES)
		// RESERVE  44-511   Reserved for future
		// MetaData OffMeta- Node Metadata pairs (size_t sz, followed by string
		//                   of bytes of length sz)
		// MatData  OffMat-  Matrix Data, Full Matrix, where source is
		//                   determined by the row, destination of edge is the
		//                   column. Data should be stored in ROW MAJOR order.
		//                   So all the connections leaving node 0 will be
		//                   stored in the first NumNode*BytePer bytes.

		// header
		gzread(gz, &numnode, 8); // NumNode
		gzread(gz, &offdata, 8); // NumNode
		gzread(gz, &offmeta, 8); // NumNode
		gzread(gz, &bytesper, 8); // NumNode
		gzread(gz, &datatype, 4); // NumNode

		if(datatype != type())
			throw RuntimeError("Error Mismatching type in File");

		gzseek(file, offmeta, SEEK_SET);

		if(nodes() != numnode) {
			m_freefunc(m_data);

			m_size = numnode;
			m_data = new T[numnode*numnode];
			m_names.resize(numnode);
			m_freefunc = [](void* ptr) { delete[] ptr; };
		}

		// read in metadata
		for(size_t ii=0; ii<nodes(); ii++) {
			gzread(gz, &tmp, sizeof(size_t));
			name(ii).resize(tmp);
			gzread(gz, name(ii).c_str(), tmp);
		}

		// Read Data
		gzwrite(gz, m_data, sizeof(T)*nodes()*nodes());
	} else if(strncmp(magic, "NPLGLIST", 8) == 0) {

		// Adjacency List Type (no distnction between directed or undirected)
		// Name     Bytes    Description
		// Magic    0-7      Magic "NPLGLIST"
		// NumNode  8-15     size_t # of nodes
		// OffList  16-23    size_t offset from start of file to first list
		//                   element (in bytes)
		// OffMeta  24-31    size_t offset from start of file to metadata (in
		//                   bytes)
		// BytePer  32-39    size_t Bytes per data element, should be
		//                   sizeof(indtype)*2+sizeo(datatype)
		// indtype  40-43    Unsigned char Index type: (See GRAPH_DATATYPES)
		// datatype 44-47    Unsigned char Data type: (See GRAPH_DATATYPES)
		// RESERVE  48-511   Reserved for future
		// MetaData OffMeta- Node Metadata pairs (size_t sz, followed by string
		//                   of bytes of length sz)
		// MatData  OffList- List Data, where nodes are stored as triplets of
		//                   indtype indtype datatype
		typedef int64_t IndType;

		// read header
		gzread(gz, &numnode, 8); // NumNode
		gzread(gz, &offdata, 8); // NumNode
		gzread(gz, &offmeta, 8); // NumNode
		gzread(gz, &bytesper, 8); // NumNode

		gzread(gz, &datatype, 4); // NumNode
		if(datatype != getType<IndType>())
			throw RuntimeError("Error Mismatching type in File");
		gzread(gz, &datatype, 4); // NumNode
		if(datatype != type())
			throw RuntimeError("Error Mismatching type in File");

		if(nodes() != numnode) {
			m_freefunc(m_data);

			m_size = numnode;
			m_data = new T[numnode*numnode];
			m_names.resize(numnode);
			m_freefunc = [](void* ptr) { delete[] ptr; };
		}

		// read in metadata
		for(size_t ii=0; ii<nodes(); ii++) {
			gzread(gz, &tmp, sizeof(size_t));
			name(ii).resize(tmp);
			gzread(gz, name(ii).c_str(), tmp);
		}

		// Fill with Zeros, then add data
		std::fill(m_data, m_data+nodes()*nodes(), 0);
		IndType ii, jj;
		T v;
		for(size_t ii=0; ii<nodes(); ii++) {
			gzread(gz, &ii, sizeof(IndType));
			gzread(gz, &jj, sizeof(IndType));
			gzread(gz, v&, sizeof(T));
			(*this)(ii,jj) = v;
		}
	}

	gzclose(gz);
	return 0;
}

} // npl
#endif

