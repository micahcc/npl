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

using std::vector;
using std::cerr;
using std::endl;

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
	cerr<<"Load Constructor"<<endl;
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](void*) {};
	load(filename, typefail);
}

template <typename T>
Graph<T>::Graph()
{
	cerr<<"Default Constructor"<<endl;
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](T*) { };
	m_names.clear();
}

template <typename T>
Graph<T>::Graph(size_t nodes)
{
	cerr<<"Basic Constructor"<<endl;
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
	cerr<<"Graph Constructor"<<endl;
	m_size = 0;
	m_data = NULL;
	m_freefunc = [](T*) { };
	m_names.clear();
	init(nodes, data, deleter);
}

template <typename T>
Graph<T>::Graph(Graph<T>&& other)
{
	cerr<<"Move Constructor"<<endl;
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
	cerr<<"Move Assignment"<<endl;
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
	cerr<<"Copy Constructor"<<endl;
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
			cerr << " next("<<ss<<","<<tt<<") = " << next(ss,tt)<<endl;
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
