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
 * @file graph.cpp This file contains the definition for Graph and its derived types.
 ******************************************************************************/

#include <complex>
#include <iostream>
#include <vector>
#include "graph.h"
#include "zlib.h"
#include "macros.h"
#include "npltypes.h"

using namespace std;

namespace npl
{

template <typename T>
GraphDataT getType()
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
	else if(typeid(T) == typeid(std::complex<float>))
		return G_DT_COMPLEX_FLOAT;
	else if(typeid(T) == typeid(std::complex<double>))
		return G_DT_COMPLEX_DOUBLE;
	else if(typeid(T) == typeid(std::complex<long double>))
		return G_DT_COMPLEX_QUAD;
	return G_DT_UNKNOWN;
};

template <typename T>
Graph<T>::Graph(std::string filename, bool typefail)
{
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](void*) {};
	load(filename, typefail);
}

template <typename T>
Graph<T>::Graph()
{
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](T*) { };
	m_names.clear();
}

template <typename T>
Graph<T>::Graph(size_t nodes)
{
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](T*) { };
	m_names.clear();
	init(nodes);
}

template <typename T>
Graph<T>::Graph(size_t nodes, void* data,
        std::function<void(void*)> deleter)
{
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](T*) { };
	m_names.clear();
	init(nodes, data, deleter);
}

template <typename T>
Graph<T>::Graph(Graph<T>&& other)
{
	m_data = other.m_data;
	m_size = other.m_size;
	m_freefunc = std::move(other.m_freefunc);
	m_names= std::move(other.m_names);

	other.m_data = NULL;
	other.m_freefunc = [](void*){ };
}

template <typename T>
Graph<T>& Graph<T>::operator=(Graph<T>&& other)
{
	// Free Any Data
	m_freefunc(m_data);

	// Take Data from other
	m_data = other.m_data;
	m_size = other.m_size;
	m_freefunc = std::move(other.m_freefunc);
	m_names= std::move(other.m_names);

	other.m_data = NULL;
	other.m_freefunc = [](void*){ };

	return *this;
}

template <typename T>
Graph<T>::Graph(const Graph<T>& other)
{
	m_size = other.m_size;
	m_data = new T[m_size*m_size];
	m_names = other.m_names;
	std::copy(other.m_data, other.m_data+sizeof(T)*m_size*m_size, m_data);
	m_freefunc = [](T* ptr) { delete[] ptr; };
}

template <typename T>
void Graph<T>::init(size_t nodes)
{
	if(nodes != m_size) {
		m_size = nodes;
		m_data = new T[nodes*nodes];
		m_freefunc = [](T* ptr) { delete[] ptr; };
		m_names.resize(nodes);
	}
}

template <typename T>
void Graph<T>::init(size_t nodes, void* data,
			std::function<void(void*)> deleter)
{
	deleter(m_data);
	m_size = nodes;
	m_data = (T*)data;
	m_freefunc = deleter;
	m_names.resize(nodes);
}

template <typename T>
void Graph<T>::save(std::string filename, GraphStoreT store)
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
	if(!gz)
		throw RUNTIME_ERROR("Could not open "+filename+" for writing!");

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
		gzputs(gz, "NPLGDMAT"); // Magic
		size_t N = nodes();
		gzwrite(gz, &N, 8); // NumNode

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
		std::fill(filler, filler+sizeof(filler), 0);
		gzwrite(gz, filler, sizeof(filler));

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
		// listsize 48-55    size_t Number of list elements
		// RESERVE  56-511   Reserved for future
		// MetaData OffMeta- Node Metadata pairs (size_t sz, followed by string
		//                   of bytes of length sz)
		// MatData  OffList- List Data, where nodes are stored as triplets of
		//                   indtype indtype datatype
		typedef int64_t IndType;
		T thresh = std::numeric_limits<T>::epsilon();

		// write header
		gzputs(gz, "NPLGLIST"); // Magic
		size_t N = nodes();
		gzwrite(gz, &N, 8); // NumNode

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
		gzwrite(gz, &tmptype, 4);

		// write out datatype
		tmptype = (int)type();
		gzwrite(gz, &tmptype, 4);

		// Figure out number of non-zero entries, and write it
		size_t listsize = 0;
		for(IndType ii=0; ii<nodes(); ii++) {
			for(IndType jj=0; jj<nodes(); jj++) {
				T v = (*this)(ii, jj);
				if(std::abs(v) > std::abs(thresh))
					listsize++;
			}
		}
		gzwrite(gz, &listsize, sizeof(size_t));

		// write filler
		char filler[456];
		std::fill(filler, filler+sizeof(filler), 0);
		gzwrite(gz, filler, sizeof(filler));

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
				if(std::abs(v) > std::abs(thresh)) {
					gzwrite(gz, &ii, sizeof(IndType));
					gzwrite(gz, &jj, sizeof(IndType));
					gzwrite(gz, &v, sizeof(T));
				}
			}
		}
	}

	gzclose(gz);
}

template <typename T>
void Graph<T>::load(std::string filename, bool)
{
	size_t tmp = 0;
	gzFile gz = gzopen(filename.c_str(), "rb");
	if(!gz)
		throw RUNTIME_ERROR("Could not open "+filename+" for writing!");

	char magic[9] = "        ";
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
			throw RUNTIME_ERROR("Error Mismatching type in File");

		if(nodes() != numnode) {
			m_freefunc(m_data);

			m_size = numnode;
			m_data = new T[numnode*numnode];
			m_names.resize(numnode);
			m_freefunc = [](void* ptr) { delete[] (T*)ptr; };
		}

		// read in metadata
		gzseek(gz, offmeta, SEEK_SET);
		for(size_t ii=0; ii<nodes(); ii++) {
			gzread(gz, &tmp, sizeof(size_t));
			name(ii).resize(tmp);
			gzread(gz, &name(ii)[0], tmp);
		}

		// Read Data
		gzseek(gz, offdata, SEEK_SET);
		gzread(gz, m_data, sizeof(T)*nodes()*nodes());
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
		// indtype  40-43    Unsigned int Index type: (See GRAPH_DATATYPES)
		// datatype 44-47    Unsigned int Data type: (See GRAPH_DATATYPES)
		// listsize 48-55    size_t Number of list elements
		// RESERVE  56-511   Reserved for future
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
			throw RUNTIME_ERROR("Error Mismatching type in File");
		gzread(gz, &datatype, 4); // NumNode
		if(datatype != type())
			throw RUNTIME_ERROR("Error Mismatching type in File");
		size_t listsize = 0;
		gzread(gz, &listsize, 8); // Number of elements in list

		if(nodes() != numnode) {
			m_freefunc(m_data);

			m_size = numnode;
			m_data = new T[numnode*numnode];
			m_names.resize(numnode);
			m_freefunc = [](void* ptr) { delete[] (T*)ptr; };
		}

		// read in metadata
		gzseek(gz, offmeta, SEEK_SET);
		for(size_t ii=0; ii<nodes(); ii++) {
			gzread(gz, &tmp, sizeof(size_t));
			name(ii).resize(tmp);
			gzread(gz, &name(ii)[0], tmp);
		}

		// Fill with Zeros, then add data
		std::fill(m_data, m_data+nodes()*nodes(), (T)0);
		IndType ii, jj;
		T v;
		gzseek(gz, offdata, SEEK_SET);
		for(size_t ll=0; ll<listsize; ll++) {
			gzread(gz, &ii, sizeof(IndType));
			gzread(gz, &jj, sizeof(IndType));
			gzread(gz, &v, sizeof(T));
			(*this)(ii,jj) = v;
		}
	}

	gzclose(gz);
}

template <typename T>
Graph<T> Graph<T>::Coxeter()
{
	static const T DATA[28*28] =
	{
		0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0
	};

	Graph<T> out(28);
	std::copy(DATA, DATA+28*28, out.m_data);
	return out;
}

template <typename T>
Graph<T> Graph<T>::PreRandom()
{
	static const double DATA[28*28] = {
0.000000, 0.390011, 1.052986, 1.183642, 0.458944, 0.426900, 0.918286, 1.086803,
1.169971, 1.353535, 0.818947, 1.036875, 1.514480, 1.244016, 1.501895, 1.012382,
1.413622, 1.353042, 0.665607, 1.515442, 0.533788, 1.707492, 1.298607, 1.147660,
1.251165, 0.728869, 1.550865, 0.458885, 0.390011, 0.000000, 0.084731, 1.396045,
0.695461, 0.415664, 1.664990, 1.070709, 1.173350, 1.129099, 0.297478, 1.406548,
0.469084, 0.560340, 1.393821, 1.316661, 1.581784, 1.167366, 1.932259, 0.898981,
1.425734, 1.264806, 1.475774, 1.379258, 0.970865, 0.657884, 0.792462, 1.297547,
1.052986, 0.084731, 0.000000, 0.561353, 1.155199, 1.362376, 1.452463, 1.609506,
0.931803, 1.309656, 0.986626, 1.663014, 0.839145, 1.054808, 1.135569, 0.648955,
1.708135, 1.211807, 1.859744, 0.893748, 1.289599, 0.462955, 1.076000, 1.089212,
0.482041, 0.704589, 0.380146, 0.449199, 1.183642, 1.396045, 0.561353, 0.000000,
1.250325, 0.837537, 0.974968, 1.605207, 1.034158, 0.772999, 0.426226, 1.162142,
1.197942, 1.644357, 0.711828, 0.913126, 0.659376, 1.340684, 0.316330, 1.064843,
0.859718, 1.504025, 1.144150, 0.329799, 1.332472, 0.955233, 0.410397, 0.934513,
0.458944, 0.695461, 1.155199, 1.250325, 0.000000, 0.594573, 1.385170, 0.772419,
1.468700, 1.705414, 1.323423, 1.115250, 0.639760, 0.657626, 0.975304, 1.711653,
0.357025, 1.496186, 0.680448, 1.766314, 0.017774, 0.782360, 1.366989, 1.161763,
1.184615, 0.514658, 0.861517, 0.598121, 0.426900, 0.415664, 1.362376, 0.837537,
0.594573, 0.000000, 0.411506, 0.779980, 1.643737, 0.905954, 1.489068, 0.374528,
0.862011, 0.410779, 1.493627, 1.106738, 0.325366, 0.524335, 0.975080, 1.316212,
0.598836, 1.288813, 0.897324, 0.989462, 0.281137, 1.071179, 1.116367, 1.064857,
0.918286, 1.664990, 1.452463, 0.974968, 1.385170, 0.411506, 0.000000, 1.490400,
1.552819, 0.614483, 1.571472, 1.814311, 1.021747, 1.352233, 1.148749, 1.401258,
0.869132, 0.671270, 1.383651, 1.694809, 0.641505, 1.015607, 1.713442, 1.741971,
1.064596, 0.535388, 1.065945, 0.676398, 1.086803, 1.070709, 1.609506, 1.605207,
0.772419, 0.779980, 1.490400, 0.000000, 0.882284, 1.600919, 1.029673, 0.424547,
0.936257, 1.306387, 1.444557, 0.035447, 1.375143, 0.749649, 1.385973, 0.791418,
1.283833, 0.984374, 0.598643, 1.163947, 1.602760, 1.246793, 0.521904, 1.328648,
1.169971, 1.173350, 0.931803, 1.034158, 1.468700, 1.643737, 1.552819, 0.882284,
0.000000, 1.231822, 0.382186, 0.037865, 0.785271, 0.411849, 0.303410, 0.855339,
0.986681, 0.832231, 1.004510, 1.081863, 1.207872, 1.269511, 0.612108, 0.771939,
0.787396, 1.597153, 0.890248, 1.065020, 1.353535, 1.129099, 1.309656, 0.772999,
1.705414, 0.905954, 0.614483, 1.600919, 1.231822, 0.000000, 1.300899, 0.615819,
0.792276, 1.850094, 0.458050, 1.021790, 0.719996, 0.914974, 1.038674, 1.387232,
0.891684, 0.877257, 0.761955, 1.077192, 0.600749, 1.034757, 0.738283, 1.029917,
0.818947, 0.297478, 0.986626, 0.426226, 1.323423, 1.489068, 1.571472, 1.029673,
0.382186, 1.300899, 0.000000, 1.250582, 0.732104, 1.050420, 1.142591, 0.628674,
1.314876, 1.158867, 0.375177, 1.302854, 1.746929, 0.419963, 1.168134, 0.354875,
1.879209, 0.366658, 1.158862, 0.627253, 1.036875, 1.406548, 1.663014, 1.162142,
1.115250, 0.374528, 1.814311, 0.424547, 0.037865, 0.615819, 1.250582, 0.000000,
1.111321, 0.710778, 1.142058, 0.596927, 1.721930, 1.771146, 0.848480, 0.565130,
1.337022, 1.161795, 0.950403, 1.397350, 1.401546, 0.724265, 0.198166, 1.819863,
1.514480, 0.469084, 0.839145, 1.197942, 0.639760, 0.862011, 1.021747, 0.936257,
0.785271, 0.792276, 0.732104, 1.111321, 0.000000, 0.867790, 1.666470, 1.314414,
1.527782, 1.413241, 1.183721, 0.908136, 0.761770, 1.346866, 1.094419, 0.850870,
0.813754, 0.631691, 0.876105, 1.010463, 1.244016, 0.560340, 1.054808, 1.644357,
0.657626, 0.410779, 1.352233, 1.306387, 0.411849, 1.850094, 1.050420, 0.710778,
0.867790, 0.000000, 1.620595, 0.726216, 0.567926, 0.378013, 1.566222, 0.273149,
1.402059, 1.061667, 0.210727, 0.792448, 1.377705, 1.065137, 1.051828, 0.823804,
1.501895, 1.393821, 1.135569, 0.711828, 0.975304, 1.493627, 1.148749, 1.444557,
0.303410, 0.458050, 1.142591, 1.142058, 1.666470, 1.620595, 0.000000, 0.940144,
1.375832, 1.437277, 0.433555, 1.732067, 1.247444, 0.911352, 0.918343, 1.123740,
1.478902, 1.384285, 0.710192, 0.790660, 1.012382, 1.316661, 0.648955, 0.913126,
1.711653, 1.106738, 1.401258, 0.035447, 0.855339, 1.021790, 0.628674, 0.596927,
1.314414, 0.726216, 0.940144, 0.000000, 1.086517, 0.998755, 0.780698, 1.233942,
0.633123, 0.987930, 1.204972, 0.297059, 1.209177, 0.898247, 0.972289, 1.306744,
1.413622, 1.581784, 1.708135, 0.659376, 0.357025, 0.325366, 0.869132, 1.375143,
0.986681, 0.719996, 1.314876, 1.721930, 1.527782, 0.567926, 1.375832, 1.086517,
0.000000, 0.797255, 1.349439, 0.912631, 0.122292, 0.729573, 0.688910, 1.418608,
1.630246, 1.778246, 0.789769, 0.701635, 1.353042, 1.167366, 1.211807, 1.340684,
1.496186, 0.524335, 0.671270, 0.749649, 0.832231, 0.914974, 1.158867, 1.771146,
1.413241, 0.378013, 1.437277, 0.998755, 0.797255, 0.000000, 1.562581, 1.195909,
1.598046, 1.288781, 0.500864, 1.426561, 1.560956, 1.266956, 0.790686, 0.573452,
0.665607, 1.932259, 1.859744, 0.316330, 0.680448, 0.975080, 1.383651, 1.385973,
1.004510, 1.038674, 0.375177, 0.848480, 1.183721, 1.566222, 0.433555, 0.780698,
1.349439, 1.562581, 0.000000, 0.672611, 0.982461, 0.259162, 0.649939, 0.349399,
1.274178, 1.787739, 1.132385, 0.562368, 1.515442, 0.898981, 0.893748, 1.064843,
1.766314, 1.316212, 1.694809, 0.791418, 1.081863, 1.387232, 1.302854, 0.565130,
0.908136, 0.273149, 1.732067, 1.233942, 0.912631, 1.195909, 0.672611, 0.000000,
1.131397, 1.285046, 1.549435, 1.385277, 0.625911, 0.745100, 0.799479, 0.974626,
0.533788, 1.425734, 1.289599, 0.859718, 0.017774, 0.598836, 0.641505, 1.283833,
1.207872, 0.891684, 1.746929, 1.337022, 0.761770, 1.402059, 1.247444, 0.633123,
0.122292, 1.598046, 0.982461, 1.131397, 0.000000, 1.468107, 0.924488, 1.482941,
0.574857, 1.547424, 0.730650, 0.825593, 1.707492, 1.264806, 0.462955, 1.504025,
0.782360, 1.288813, 1.015607, 0.984374, 1.269511, 0.877257, 0.419963, 1.161795,
1.346866, 1.061667, 0.911352, 0.987930, 0.729573, 1.288781, 0.259162, 1.285046,
1.468107, 0.000000, 0.688529, 1.047796, 0.329903, 1.192382, 1.653906, 0.576523,
1.298607, 1.475774, 1.076000, 1.144150, 1.366989, 0.897324, 1.713442, 0.598643,
0.612108, 0.761955, 1.168134, 0.950403, 1.094419, 0.210727, 0.918343, 1.204972,
0.688910, 0.500864, 0.649939, 1.549435, 0.924488, 0.688529, 0.000000, 0.975230,
1.184997, 1.195622, 1.121823, 0.739153, 1.147660, 1.379258, 1.089212, 0.329799,
1.161763, 0.989462, 1.741971, 1.163947, 0.771939, 1.077192, 0.354875, 1.397350,
0.850870, 0.792448, 1.123740, 0.297059, 1.418608, 1.426561, 0.349399, 1.385277,
1.482941, 1.047796, 0.975230, 0.000000, 0.278306, 0.379372, 1.092345, 0.622192,
1.251165, 0.970865, 0.482041, 1.332472, 1.184615, 0.281137, 1.064596, 1.602760,
0.787396, 0.600749, 1.879209, 1.401546, 0.813754, 1.377705, 1.478902, 1.209177,
1.630246, 1.560956, 1.274178, 0.625911, 0.574857, 0.329903, 1.184997, 0.278306,
0.000000, 0.652962, 1.794140, 0.774540, 0.728869, 0.657884, 0.704589, 0.955233,
0.514658, 1.071179, 0.535388, 1.246793, 1.597153, 1.034757, 0.366658, 0.724265,
0.631691, 1.065137, 1.384285, 0.898247, 1.778246, 1.266956, 1.787739, 0.745100,
1.547424, 1.192382, 1.195622, 0.379372, 0.652962, 0.000000, 0.954847, 1.011197,
1.550865, 0.792462, 0.380146, 0.410397, 0.861517, 1.116367, 1.065945, 0.521904,
0.890248, 0.738283, 1.158862, 0.198166, 0.876105, 1.051828, 0.710192, 0.972289,
0.789769, 0.790686, 1.132385, 0.799479, 0.730650, 1.653906, 1.121823, 1.092345,
1.794140, 0.954847, 0.000000, 1.042865, 0.458885, 1.297547, 0.449199, 0.934513,
0.598121, 1.064857, 0.676398, 1.328648, 1.065020, 1.029917, 0.627253, 1.819863,
1.010463, 0.823804, 0.790660, 1.306744, 0.701635, 0.573452, 0.562368, 0.974626,
0.825593, 0.576523, 0.739153, 0.622192, 0.774540, 1.011197, 1.042865,
0.000000};

	Graph<T> out(28);
	for(size_t ii=0; ii<28; ii++){
		for(size_t jj=0; jj<28; jj++){
			out(ii,jj) = (T)DATA[ii*28+jj];
		}
	}
	return out;
}

/*
 * Computes assortativity for a directed or undirected graph.  In
 * an undirected graph istrength = ostrength , and in an unweighted
 * graph strength = degree
 */
template<typename T>
double Graph<T>::assortativity() const
{
	const Graph<T>& CIJ = *this;
	const double thresh = std::numeric_limits<double>::epsilon();
	vector<T> is(nodes(), 0);
	vector<T> os(nodes(), 0);
	T total = 0;

	// Compute Input Strengths and Output Strengths
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			is[ii] += (std::abs(CIJ(ii, jj)) > thresh);
			os[jj] += (std::abs(CIJ(ii, jj)) > thresh);
			total  += (std::abs(CIJ(ii, jj)) > thresh);
		}
	}

	// Correlate Strengths Among Connected Nodes
	size_t ecount = 0;
	T num1 = 0;
	T num2 = 0;
	T den1 = 0;
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			if(std::abs(CIJ(ii, jj)) > thresh) {
				ecount++;

				num1 += is[ii]*os[jj];
				num2 += is[ii]+os[jj];
				den1 += is[ii]*is[ii] + os[ii]*os[ii];
			}
		}
	}

	num1 /= (T)ecount;
	den1 /= (T)2*(T)ecount;
	num2 = std::pow(num2/((T)2*(T)ecount), 2);

	if((num1-num2) == (den1-num2))
		return 1;
	else
		return (double)std::abs((num1-num2)/(den1-num2));
}

template<typename T>
double Graph<T>::assortativity_wei() const
{
	const Graph<T>& CIJ = *this;
	const double thresh = std::numeric_limits<double>::epsilon();
	vector<T> is(nodes(), 0);
	vector<T> os(nodes(), 0);
	T total = 0;

	// Compute Input Strengths and Output Strengths
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			is[ii] += CIJ(ii, jj);
			os[jj] += CIJ(ii, jj);
			total += CIJ(ii, jj);
		}
	}

	// Correlate Strengths Among Connected Nodes
	size_t ecount = 0;
	T num1 = 0;
	T num2 = 0;
	T den1 = 0;
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			if(std::abs(CIJ(ii, jj)) > thresh) {
				ecount++;

				num1 += is[ii]*os[jj];
				num2 += is[ii]+os[jj];
				den1 += is[ii]*is[ii] + os[ii]*os[ii];
			}
		}
	}

	num1 /= (T)ecount;
	den1 /= (T)2*(T)ecount;
	num2 = std::pow(num2/((T)2*(T)ecount), 2.);

	return (double)std::abs((num1-num2)/(den1-num2));
}

/*
 * Computes degrees, in-degrees, and out-degrees (and the average of the 2)
 */
template <typename T>
vector<int> Graph<T>::degrees(vector<int>& is,
		vector<int>& os) const
{
	const Graph<T>& CIJ = *this;
	is.resize(nodes()); // in degree
	os.resize(nodes()); // out degree
	vector<int> out(nodes(), 0); // total degree
	std::fill(is.begin(), is.end(), 0);
	std::fill(os.begin(), os.end(), 0);
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			is[ii] += (std::abs(CIJ(ii, jj))!=0);
			os[jj] += (std::abs(CIJ(ii, jj))!=0);
			out[ii] += (std::abs(CIJ(ii, jj))!=0);
			out[jj] += (std::abs(CIJ(ii, jj))!=0);
		}
	}
	return out;
}

template <typename T>
vector<int> Graph<T>::degrees() const
{
	const Graph<T>& CIJ = *this;
	vector<int> out(nodes(), 0); // total degree
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			out[ii] += (std::abs(CIJ(ii, jj))!=0);
			out[jj] += (std::abs(CIJ(ii, jj))!=0);
		}
	}
	return out;
}

template <typename T>
int Graph<T>::degree() const
{
	int total = 0;
	const Graph<T>& CIJ = *this;
	//this works out so that an undirected graph double counts cycles, and single
	//counts undirected paths. Simultaneously in a directed graph, self connections
	//will be double counted, but this number has little meaning in a directed graph
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			total += (std::abs(CIJ(ii, jj))!=0);
		}
	}

	return total;
}

/*
 * Computes strength, in-strength, and out-strength (and the average of the 2)
 */
template <typename T>
vector<T> Graph<T>::strengths(vector<T>& is, vector<T>& os) const
{
	const Graph<T>& CIJ = *this;
	vector<T> total(nodes(), 0);
	is.resize(nodes());
	os.resize(nodes());
	std::fill(is.begin(), is.end(), (T)0);
	std::fill(os.begin(), os.end(), (T)0);

	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			is[ii] += CIJ(ii, jj);
			os[jj] += CIJ(ii, jj);
			total[ii] += CIJ(ii, jj);
			total[jj] += CIJ(ii, jj);
		}
	}

	return total;
}

template <typename T>
vector<T> Graph<T>::strengths() const
{
	const Graph<T>& CIJ = *this;
	vector<T> out(nodes(), (T)0);
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			out[jj] += CIJ(ii, jj);
			out[ii] += CIJ(ii, jj);
		}
	}
	return out;
}

template <typename T>
T Graph<T>::strength() const
{
	const Graph<T>& CIJ = *this;
	T out = 0;
	for(size_t ii = 0 ; ii < CIJ.nodes() ; ii++) {
		for(size_t jj = 0 ; jj < CIJ.nodes() ; jj++)
			out += CIJ(ii, jj);
	}

	return out;
}

/**
* @brief Calculates betweenness centrality given a distance matrix.
*
* This first calculates the shortest path/next matrices so it is
* slower than the next methods if run repeatedly.
*
* @return	betweennes centrality of each vertex, the number of shortest
*			paths that pass through the vertex.
*/
template <typename T>
vector<int> Graph<T>::betweenness_centrality() const
{
	assert(nodes() < INT_MAX);
	Graph<int> nextmat;
	Graph<T> shortmat;
	shortest(nextmat, shortmat);
	return nextmat.betweenness_centrality_next();
}

/*
 * Computes node betweenness for a graph given the next matrix
 * provided by distance(). this should be a next matrix
 */
template <typename T>
vector<int> Graph<T>::betweenness_centrality_next() const
{
	if(type() != G_DT_SLONG) {
		throw RUNTIME_ERROR("Error calling betweenness_centrality_next on a "
				"matrix this is not an integer, so it can't be a matrix of "
				"next values. Consider the between_centrality function "
				"instead");
	}
	const Graph<int>& next = *(Graph<int>*)this;
	vector<int> out(nodes(), 0);

	//traverse all pairs of starting and ending points
	for(int ss = 0 ; ss < nodes(); ss++) {
		for(int tt = 0 ; tt < nodes(); tt++) {
			//no path (next(ss,tt) == ss) or direct path
			// (next(ss,tt) == tt) are skipped
			if(ss != tt && next(ss,tt) != tt && next(ss,tt) != ss) {
				int pos = ss;
				//follow path between ss and tt, and increment every vertex visited
				while((pos = next(pos, tt)) != tt)
					out[pos]++;
			}
		}
	}

	return out;
}

/**
 * @brief Compute shortest paths between all nodes without saving the next
 * graph for each node.
 *
 * @tparam T
 * @param sdist
 */
template <typename T>
void Graph<T>::shortest(Graph<T>& sdist) const
{
	// Realloc sdist if necessary
	if(sdist.nodes() != nodes()) {
		sdist.m_freefunc(sdist.m_data);
		sdist.m_data = new T[nodes()*nodes()];
		sdist.m_size = nodes();
		sdist.m_names.resize(nodes());
	}

	// Initialize distances to the direct distances, non existent connections
	// should already have been set to max or infinity
	for(size_t ii = 0 ; ii < nodes(); ii++) {
		for(size_t jj = 0 ; jj < nodes(); jj++) {
			if(ii == jj)
				sdist(ii,jj) = (T)0;
			else
				sdist(ii,jj) = (*this)(ii,jj);
		}
	}

	// If going through ii->kk->jj is shorter than ii->jj make next
	// and replace old shortest distance (ii->jj) with ii->kk->jj
	for(size_t kk = 0; kk < nodes(); kk++) {
		for(size_t ii = 0 ; ii < nodes(); ii++) {
			for(size_t jj = 0 ; jj < nodes(); jj++) {
				if(std::abs(sdist(ii,kk)) + std::abs(sdist(kk, jj))
							< std::abs(sdist(ii, jj))) {
					sdist(ii,jj) = sdist(ii,kk) + sdist(kk,jj);
				}
			}
		}
	}
}

/**
 * @brief Compute shortest paths between all nodes. The next Graph provides the
 * next node that is the fastest way to target. So G(i,j) = k means that
 * fastest way to get from i to j is to go through k. Then G(k,j) = l is the
 * next step and so on. G(i,j) = i is naturally invalid, so that is used to
 * indicate that no path exists. G(i,j) = j means that there is a direct link.
 *
 * @tparam T
 * @param next
 * @param sdist
 */
template <typename T>
void Graph<T>::shortest(Graph<int>& next, Graph<T>& sdist) const
{
	// Realloc sdist if necessary
	if(sdist.nodes() != nodes())
		sdist.init(nodes());

	// Realloc next if necessary
	if(next.nodes() != nodes())
		next.init(nodes());

	// Initialize distances to the direct distances, non existent connections
	// should already have been set to max or infinity
	for(size_t ii = 0 ; ii < nodes(); ii++) {
		for(size_t jj = 0 ; jj < nodes(); jj++) {
			if(ii == jj)
				sdist(ii,jj) = (T)0;
			else
				sdist(ii,jj) = (*this)(ii,jj);
		}
	}

	// Default to unconnected for all nodes (points back to source)
	for(size_t ii = 0; ii < nodes(); ii++) {
		for(size_t jj = 0; jj < nodes(); jj++) {
			next(ii,jj) = jj;
		}
	}

	// If going through ii->kk->jj is shorter than ii->jj make next
	// and replace old shortest distance (ii->jj) with ii->kk->jj
	for(size_t kk = 0; kk < nodes(); kk++) {
		for(size_t ii = 0 ; ii < nodes(); ii++) {
			for(size_t jj = 0 ; jj < nodes(); jj++) {
				if(std::abs(sdist(ii,kk)) + std::abs(sdist(kk, jj))
							< std::abs(sdist(ii, jj))) {
					sdist(ii,jj) = sdist(ii,kk) + sdist(kk,jj);
					next(ii,jj) = kk;
				}
			}
		}
	}
}

/*
 * Normalize all weights by the total weight
 */
template<typename T>
void Graph<T>::normalize()
{
	Graph<T>& CIJ = *this;
	double scale = strength();
	Graph<T> out = CIJ;

	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++)
			CIJ(ii,jj) /= scale;
	}
}

template class Graph<double>;
template class Graph<long double>;
template class Graph<cdouble_t>;
template class Graph<cquad_t>;
template class Graph<float>;
template class Graph<cfloat_t>;
template class Graph<int64_t>;
template class Graph<uint64_t>;
template class Graph<int32_t>;
template class Graph<uint32_t>;
template class Graph<int16_t>;
template class Graph<uint16_t>;
template class Graph<int8_t>;
template class Graph<uint8_t>;

}
