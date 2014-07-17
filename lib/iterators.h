/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries are free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#ifndef ITERATOR_H
#define ITERATOR_H

#include <stdexcept>
#include <memory>
#include "npltypes.h"
#include "ndarray.h"
#include "slicer.h"

namespace npl {

template <typename T>
class FlatIter
{
public:
	FlatIter(std::shared_ptr<NDArray> in) : i(0), parent(in)
	{
		switch(in->type()) {
			case UINT8:
				castfunc = castor<uint8_t>;
				break;
			case INT8:
				castfunc = castor<int8_t>;
				break;
			case UINT16:
				castfunc = castor<uint16_t>;
				break;
			case INT16:
				castfunc = castor<int16_t>;
				break;
			case UINT32:
				castfunc = castor<uint32_t>;
				break;
			case INT32:
				castfunc = castor<int32_t>;
				break;
			case UINT64:
				castfunc = castor<uint64_t>;
				break;
			case INT64:
				castfunc = castor<int64_t>;
				break;
			case FLOAT32:
				castfunc = castor<float>;
				break;
			case FLOAT64:
				castfunc = castor<double>;
				break;
			case FLOAT128:
				castfunc = castor<long double>;
				break;
			case COMPLEX64:
				castfunc = castor<cfloat_t>;
				break;
			case COMPLEX128:
				castfunc = castor<cdouble_t>;
				break;
			case COMPLEX256:
				castfunc = castor<cquad_t>;
				break;
			case RGB24:
				castfunc = castor<rgb_t>;
				break;
			case RGBA32:
				castfunc = castor<rgba_t>;
				break;
			case UNKNOWN_TYPE:
				throw std::invalid_argument("Unknown type to BasicIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() { assert(!isEnd()); return castfunc(parent->getAddr(++i)); };

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) { assert(!isEnd()); return castfunc(parent->getAddr(i++)); };

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() { assert(!isBegin()); return castfunc(parent->getAddr(--i)); };
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) { assert(!isBegin()); return castfunc(parent->getAddr(i--)); };

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() { assert(!isEnd()); return castfunc(parent->getAddr(i)); };

	/**
	 * @brief Go to beginning of iteration
	 */
	void goBegin() { i = 0; };

	/**
	 * @brief Go to end of iteration
	 */
	void goEnd() { i = parent->elements(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	void isEnd() { i == parent->elements(); };
	
	/**
	 * @brief Are we at the first element
	 */
	void isBegin() { i == 0; };
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const BasicIter other) 
	{ 
		return parent == other.parent && i == other.i; 
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const BasicIter other) 
	{ 
		return parent != other.parent || i != other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<(const BasicIter other) 
	{ 
		return parent == other.parent && i < other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const BasicIter other) 
	{ 
		return parent == other.parent && i > other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const BasicIter other) 
	{ 
		return parent == other.parent && i <= other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const BasicIter other) 
	{ 
		return parent == other.parent && i >= other.i; 
	};

private:
	template <typename U>
	static T castor(void* ptr)
	{
		return (T)(*((U*)ptr));
	};


	size_t i;

	std::shared_ptr<NDArray> parent;
	T (*castfunc)(void* ptr);
};

template <typename T>
class NDIter : protected Slicer 
{
public:
	NDIter(std::shared_ptr<NDArray> in, std::initializer_list<size_t> order)
				: i(0), parent(in), Slicer(in->ndim(), in->dim(), order)

	{
		switch(in->type()) {
			case UINT8:
				castfunc = castor<uint8_t>;
				break;
			case INT8:
				castfunc = castor<int8_t>;
				break;
			case UINT16:
				castfunc = castor<uint16_t>;
				break;
			case INT16:
				castfunc = castor<int16_t>;
				break;
			case UINT32:
				castfunc = castor<uint32_t>;
				break;
			case INT32:
				castfunc = castor<int32_t>;
				break;
			case UINT64:
				castfunc = castor<uint64_t>;
				break;
			case INT64:
				castfunc = castor<int64_t>;
				break;
			case FLOAT32:
				castfunc = castor<float>;
				break;
			case FLOAT64:
				castfunc = castor<double>;
				break;
			case FLOAT128:
				castfunc = castor<long double>;
				break;
			case COMPLEX64:
				castfunc = castor<cfloat_t>;
				break;
			case COMPLEX128:
				castfunc = castor<cdouble_t>;
				break;
			case COMPLEX256:
				castfunc = castor<cquad_t>;
				break;
			case RGB24:
				castfunc = castor<rgb_t>;
				break;
			case RGBA32:
				castfunc = castor<rgba_t>;
				break;
			case UNKNOWN_TYPE:
				throw std::invalid_argument("Unknown type to BasicIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() { return castfunc(parent->getAddr(++Slicer)); };

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) { return castfunc(parent->getAddr(Slicer++)); };

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() { return castfunc(parent->getAddr(--Slicer)); };
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) { return castfunc(parent->getAddr(Slicer--)); };

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() { return castfunc(parent->getAddr(*Slicer)); };

	/**
	 * @brief Go to beginning of iteration
	 */
	void goBegin() { Slicer::goBegin(); };

	/**
	 * @brief Go to end of iteration
	 */
	void goEnd() { Slicer::goEnd(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	void isEnd() { Slicer::isEnd(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	void eof() { Slicer::isEnd(); };
	
	/**
	 * @brief Are we at the first element
	 */
	void isBegin() { Slicer::isBegin(); };
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const NDArray& other) 
	{ 
		return parent == other.parent && this->m_linpos == other.m_linpos;
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const BasicIter other) 
	{ 
		return parent != other.parent || this->m_linpos != other.m_linpos;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<(const BasicIter other) 
	{ 
		for(size_t dd=0; dd<m_dim; dd++)
		return parent == other.parent && i < other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const BasicIter other) 
	{ 
		return parent == other.parent && i > other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const BasicIter other) 
	{ 
		return parent == other.parent && i <= other.i; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const BasicIter other) 
	{ 
		return parent == other.parent && i >= other.i; 
	};

private:
	template <typename U>
	static T castor(void* ptr)
	{
		return (T)(*((U*)ptr));
	};


	size_t i;

	std::shared_ptr<NDArray> parent;
	T (*castfunc)(void* ptr);
};

}

#endif //ITERATOR_H
