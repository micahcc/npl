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
Graph<T>::Graph(size_t nodes)
{
	m_size = nodes;
	m_data = new T[nodes*nodes];
	m_freefunc = [](T* ptr) { delete[] ptr; };
	m_names.resize(nodes);
}

template <typename T>
Graph<T>::Graph(size_t nodes, void* data,
        std::function<void(void*)> deleter)
{
	m_size = nodes;
	m_data = (T*)data;
	m_freefunc = deleter;
	m_names.resize(nodes);
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
Graph<T>::Graph(const Graph<T>& other)
{
	m_size = other.m_size;
	m_data = new T[m_size*m_size];
	m_names = other.m_names;
	std::copy(other.m_data, other.m_data+sizeof(T)*m_size*m_size, m_data);
	m_freefunc = [](T* ptr) { delete[] ptr; };
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
void Graph<T>::Coxeter()
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

	if(m_size != 28) {
		m_freefunc(m_data);
		m_data = new T[28*28];
		m_freefunc = [](T* ptr){delete[] ptr;};
	}
	m_size = 28;
	std::copy(DATA, DATA+28*28, m_data);
	m_names.resize(28);
}

/*
 * Computes assortativity for a directed or undirected graph.  In
 * an undirected graph istrength = ostrength , and in an unweighted
 * graph strength = degree
 */
template<typename T>
double assortativity(const Graph<T>& CIJ)
{
	std::vector<T> idegree;
	vector<T> odegree;
	degrees(CIJ, idegree, odegree);
	return assortativity(CIJ, idegree, odegree);
}

template<typename T>
double Graph<T>::assortativity_wei() const
{
	vector<T> idegree;
	vector<T> odegree;
	this->strengths(idegree, odegree);
	return assortativity(idegree, odegree);
}

template<typename T>
double Graph<T>::assortativity(const vector<T>& idegree,
		const vector<T>& odegree) const
{
	const Graph<T>& CIJ = *this;
	size_t ecount = 0;
	double num1 = 0;
	double num2 = 0;
	double den1 = 0;
	for(size_t ii = 0 ; ii < CIJ.nodes() ; ii++) {
		for(size_t jj = 0 ; jj < CIJ.nodes() ; jj++) {
			if(CIJ(ii, jj) != (T)0) {
				ecount++;

				num1 += std::abs(idegree[ii]*odegree[jj]);
				num2 += std::abs(idegree[ii]+odegree[jj]);
				den1 += std::abs(std::pow(idegree[ii], 2) + std::pow(odegree[jj] , 2));
			}
		}
	}

	num1 /= ecount;
	den1 /= 2*ecount;
	num2 = std::pow(num2 / (2*ecount), 2);

	return (num1 - num2) / (den1 - num2);
}

/*
 * Computes degrees, in-degrees, and out-degrees (and the average of the 2)
 */
template <typename T>
vector<size_t> Graph<T>::degrees(vector<size_t>& is,
		vector<size_t>& os) const
{
	const Graph<T>& CIJ = *this;
	is.resize(nodes()); // in degree
	os.resize(nodes()); // out degree
	vector<size_t> out(nodes(), 0); // total degree
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
vector<size_t> Graph<T>::degrees() const
{
	const Graph<T>& CIJ = *this;
	vector<size_t> out(nodes(), 0); // total degree
	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			out[ii] += (std::abs(CIJ(ii, jj))!=0);
			out[jj] += (std::abs(CIJ(ii, jj))!=0);
		}
	}
	return out;
}

template <typename T>
size_t Graph<T>::degree() const
{
	size_t total = 0;
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
T Graph<T>::strengths(vector<T>& is, vector<T>& os) const
{
	const Graph<T>& CIJ = *this;
	T total = 0;
	is.resize(nodes());
	os.resize(nodes());
	std::fill(is.begin(), is.end(), (T)0);
	std::fill(os.begin(), os.end(), (T)0);

	for(size_t ii = 0 ; ii < nodes() ; ii++) {
		for(size_t jj = 0 ; jj < nodes() ; jj++) {
			is[ii] += CIJ(ii, jj);
			os[jj] += CIJ(ii, jj);
			total += CIJ(ii, jj);
		}
	}

	return 0;
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

}
