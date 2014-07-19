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
#include "kernel_slicer.h"

namespace npl {

/**
 * @brief Flat iterator for NDArray. No information is kept about 
 * the current ND index. Just goes through all data. This casts the output to the
 * type specified using T.
 *
 * @tparam T
 */
template <typename T>
class FlatIter 
{
public:
	FlatIter(std::shared_ptr<NDArray> in)
				: parent(in), m_linpos(0)

	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				castset = castsetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				castset = castsetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				castset = castsetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				castset = castsetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				castset = castsetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				castset = castsetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				castset = castsetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				castset = castsetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				castset = castsetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				castset = castsetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				castset = castsetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				castset = castsetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				castset = castsetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				castset = castsetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				castset = castsetStatic<rgba_t>;
				break;
			default:
			case UNKNOWN_TYPE:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to FlatIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() 
	{ 
		return castget(parent->__getAddr(++m_linpos)); 
	};

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) 
	{ 
		return castget(parent->__getAddr(m_linpos++)); 
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() 
	{ 
		return castget(parent->__getAddr(--m_linpos)); 
	};
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) 
	{ 
		return castget(parent->__getAddr(m_linpos--)); 
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{ 
		return castget(parent->__getAddr(m_linpos)); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{ 
		return castget(parent->__getAddr(m_linpos)); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	void set(T v) const
	{ 
		castset(parent->__getAddr(m_linpos), v); 
	};
	
	/**
	 * @brief Go to beginning of iteration
	 */
	void goBegin() { m_linpos=0; };

	/**
	 * @brief Go to end of iteration
	 */
	void goEnd() { m_linpos=parent->elements(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	bool isEnd() const { return m_linpos==parent->elements(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	bool eof() const { return m_linpos==parent->elements(); };
	
	/**
	 * @brief Are we at the first element
	 */
	bool isBegin() const { return m_linpos == 0; };
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const FlatIter& other) const
	{ 
		return parent == other.parent && m_linpos == other.m_linpos;
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const FlatIter& other) const
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
	bool operator<(const FlatIter& other) const
	{ 
		return parent == other.parent && m_linpos < other.m_linpos;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const FlatIter& other) const
	{ 
		return parent == other.parent && m_linpos > other.m_linpos;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const FlatIter& other) const
	{ 
		return parent == other.parent && m_linpos <= other.m_linpos; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const FlatIter& other) const
	{ 
		return parent == other.parent && m_linpos >= other.m_linpos; 
	};

private:
	FlatIter();
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*((U*)ptr));
	};
	
	template <typename U>
	static void castsetStatic(void* ptr, const T& val)
	{
		(*((U*)ptr)) = (U)val;
	};

	
	std::shared_ptr<NDArray> parent;

	T (*castget)(void* ptr);
	void (*castset)(void* ptr, const T& v);

	int64_t m_linpos;
};

/**
 * @brief Flat iterator iterator for NDArray. No information is kept about 
 * the current index. Just goes through all data. This casts the output to the
 * type specified using T.
 *
 * @tparam T
 */
template <typename T>
class FlatConstIter 
{
public:
	FlatConstIter(std::shared_ptr<const NDArray> in)
				: parent(in), m_linpos(0)

	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				break;
			case UNKNOWN_TYPE:
			default:
				castget = castgetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to FlatIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() 
	{ 
		return castget(parent->__getAddr(++m_linpos)); 
	};

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) 
	{ 
		return castget(parent->__getAddr(m_linpos++)); 
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() 
	{ 
		return castget(parent->__getAddr(--m_linpos)); 
	};
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) 
	{ 
		return castget(parent->__getAddr(m_linpos--)); 
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{ 
		return castget(parent->__getAddr(m_linpos)); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{ 
		return castget(parent->__getAddr(m_linpos)); 
	};
	
	/**
	 * @brief Go to beginning of iteration
	 */
	void goBegin() { m_linpos=0; };

	/**
	 * @brief Go to end of iteration
	 */
	void goEnd() { m_linpos=parent->elements(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	bool isEnd() const { return m_linpos==parent->elements(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	bool eof() const { return m_linpos==parent->elements(); };
	
	/**
	 * @brief Are we at the first element
	 */
	bool isBegin() const { return m_linpos == 0; };
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const FlatConstIter& other) const
	{ 
		return parent == other.parent && m_linpos == other.m_linpos;
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const FlatConstIter& other) const
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
	bool operator<(const FlatConstIter& other) const
	{ 
		return parent == other.parent && m_linpos < other.m_linpos;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const FlatConstIter& other) const
	{ 
		return parent == other.parent && m_linpos > other.m_linpos;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const FlatConstIter& other) const
	{ 
		return parent == other.parent && m_linpos <= other.m_linpos; 
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const FlatConstIter& other) const
	{ 
		return parent == other.parent && m_linpos >= other.m_linpos; 
	};

private:
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*((U*)ptr));
	};
	
	std::shared_ptr<const NDArray> parent;

	T (*castget)(void* ptr);

	int64_t m_linpos;
};


/**
 * @brief Constant iterator for NDArray. Typical usage calls for OrderConstIter it(array); it++; *it
 *
 * @tparam T
 */
template <typename T>
class OrderConstIter : public Slicer 
{
public:
	OrderConstIter(std::shared_ptr<const NDArray> in, 
				std::vector<size_t> order = {}, bool revorder = false)
				: Slicer(in->ndim(), in->dim(), order, revorder), parent(in)
	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				break;
			case UNKNOWN_TYPE:
			default:
				castget = castgetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to OrderConstIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() 
	{ 
		return castget(parent->__getAddr(Slicer::operator++())); 
	};

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) 
	{ 
		return castget(parent->__getAddr(Slicer::operator++(0))); 
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() 
	{ 
		return castget(parent->__getAddr(Slicer::operator--())); 
	};
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) 
	{ 
		return castget(parent->__getAddr(Slicer::operator--(0))); 
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{ 
		return castget(parent->__getAddr(Slicer::operator*())); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{ 
		return castget(parent->__getAddr(Slicer::operator*())); 
	};
	
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
	bool isEnd() const { return Slicer::isEnd(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	bool eof() const { return Slicer::isEnd(); };
	
	/**
	 * @brief Are we at the first element
	 */
	bool isBegin() const { return Slicer::isBegin(); };
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const OrderConstIter& other) const
	{ 
		return parent == other.parent && m_linpos == other.m_linpos;
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const OrderConstIter& other) const
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
	bool operator<(const OrderConstIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		for(size_t dd=0; dd<this->m_dim; dd++) {
			if(this->m_pos[dd] < other.m_pos[dd])
				return true;
		}

		return false;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const OrderConstIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		for(size_t dd=0; dd<this->m_dim; dd++) {
			if(this->m_pos[dd] > other.m_pos[dd])
				return true;
		}

		return false;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const OrderConstIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this < other;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const OrderConstIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this > other;
	};

private:
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*((U*)ptr));
	};
	
	std::shared_ptr<const NDArray> parent;

	T (*castget)(void* ptr);
};
/**
 * @brief This class is used to iterate through an N-Dimensional array. 
 *
 * @tparam T
 */
template <typename T>
class OrderIter : public Slicer 
{
public:
	OrderIter(std::shared_ptr<NDArray> in, 
				std::initializer_list<size_t> order = {}, bool revorder = false)
				: Slicer(in->ndim(), in->dim(), order, revorder), parent(in)

	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				castset = castsetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				castset = castsetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				castset = castsetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				castset = castsetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				castset = castsetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				castset = castsetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				castset = castsetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				castset = castsetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				castset = castsetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				castset = castsetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				castset = castsetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				castset = castsetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				castset = castsetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				castset = castsetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				castset = castsetStatic<rgba_t>;
				break;
			default:
			case UNKNOWN_TYPE:
				castget = castgetStatic<uint8_t>;
				castset = castsetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to OrderIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() 
	{ 
		return castget(parent->__getAddr(Slicer::operator++())); 
	};

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) 
	{ 
		return castget(parent->__getAddr(Slicer::operator++(0))); 
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() 
	{ 
		return castget(parent->__getAddr(Slicer::operator--())); 
	};
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) 
	{ 
		return castget(parent->__getAddr(Slicer::operator--(0))); 
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{ 
		return castget(parent->__getAddr(Slicer::operator*())); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{ 
		return castget(parent->__getAddr(Slicer::operator*())); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	void set(const T& v) 
	{ 
		this->castset(parent->__getAddr(Slicer::operator*()), v); 
	};

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
	bool isEnd() const { return Slicer::isEnd(); };
	
	/**
	 * @brief Are we one past the last element?
	 */
	bool eof() const { return Slicer::isEnd(); };
	
	/**
	 * @brief Are we at the first element
	 */
	bool isBegin() const { return Slicer::isBegin(); };
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const OrderIter& other) const
	{ 
		return parent == other.parent && m_linpos == other.m_linpos;
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const OrderIter other) const
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
	bool operator<(const OrderIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		for(size_t dd=0; dd<this->m_dim; dd++) {
			if(this->m_pos[dd] < other.m_pos[dd])
				return true;
		}

		return false;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const OrderIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		for(size_t dd=0; dd<this->m_dim; dd++) {
			if(this->m_pos[dd] > other.m_pos[dd])
				return true;
		}

		return false;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const OrderIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this < other;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const OrderIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this > other;
	};

private:
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*((U*)ptr));
	};
	
	template <typename U>
	static void castsetStatic(void* ptr, const T& val)
	{
		(*((U*)ptr)) = (U)val;
	};

	
	std::shared_ptr<NDArray> parent;

	T (*castget)(void* ptr);
	void (*castset)(void* ptr, const T& val);
};


/**
 * @brief Iterator for an image, that allows for easy access to the neighbors
 * of the current element/pixel. Neighbors can be accessed through offset(i)
 * which simply provides the value of the i'th neighbor. To get that neighbors
 * index you can use it.offset_index(i). For the center use
 * it.center()/it.center_index(). [] may also be used in place of offset. To 
 * get the number of neighbors in the kernel use ksize(), so
 * it.offset(0), it.offset(1), ..., it.offset(ksize()-1) are valid calls.
 * 
 * @tparam T
 */
template <typename T>
class KernelIter : public KSlicer 
{
public:
	KernelIter(std::shared_ptr<const NDArray> in)
				: KSlicer(in->ndim(), in->dim()), parent(in)
	{
		switch(in->type()) {
			case UINT8:
				castget = castgetStatic<uint8_t>;
				break;
			case INT8:
				castget = castgetStatic<int8_t>;
				break;
			case UINT16:
				castget = castgetStatic<uint16_t>;
				break;
			case INT16:
				castget = castgetStatic<int16_t>;
				break;
			case UINT32:
				castget = castgetStatic<uint32_t>;
				break;
			case INT32:
				castget = castgetStatic<int32_t>;
				break;
			case UINT64:
				castget = castgetStatic<uint64_t>;
				break;
			case INT64:
				castget = castgetStatic<int64_t>;
				break;
			case FLOAT32:
				castget = castgetStatic<float>;
				break;
			case FLOAT64:
				castget = castgetStatic<double>;
				break;
			case FLOAT128:
				castget = castgetStatic<long double>;
				break;
			case COMPLEX64:
				castget = castgetStatic<cfloat_t>;
				break;
			case COMPLEX128:
				castget = castgetStatic<cdouble_t>;
				break;
			case COMPLEX256:
				castget = castgetStatic<cquad_t>;
				break;
			case RGB24:
				castget = castgetStatic<rgb_t>;
				break;
			case RGBA32:
				castget = castgetStatic<rgba_t>;
				break;
			case UNKNOWN_TYPE:
			default:
				castget = castgetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to OrderConstIter");
				break;
		}
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	T operator++() 
	{ 
		return castget(parent->__getAddr(KSlicer::operator++())); 
	};

	/**
	 * @brief Postfix increment operator
	 *
	 * @return old value
	 */
	T operator++(int) 
	{ 
		return castget(parent->__getAddr(KSlicer::operator++(0))); 
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	T operator--() 
	{ 
		return castget(parent->__getAddr(KSlicer::operator--())); 
	};
	
	/**
	 * @brief Postfix decrement operator
	 *
	 * @return old value
	 */
	T operator--(int) 
	{ 
		return castget(parent->__getAddr(KSlicer::operator--(0))); 
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{ 
		return castget(parent->__getAddr(KSlicer::operator*())); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T center() const
	{ 
		return castget(parent->__getAddr(KSlicer::center())); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value at offset k
	 */
	T offset(int64_t k) const
	{ 
		return castget(parent->__getAddr(KSlicer::offset(k))); 
	};
	
	/**
	 * @brief Dereference operator
	 *
	 * @return current value at offset k
	 */
	T operator[](int64_t k) const
	{ 
		return castget(parent->__getAddr(KSlicer::offset(k))); 
	};
	
	/**
	 * @brief Whether the position and parent are the same as another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator==(const KernelIter& other) const
	{ 
		return parent == other.parent && m_linpos == other.m_linpos;
	};

	/**
	 * @brief Whether the position and parent are different from another 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator!=(const KernelIter& other) const
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
	bool operator<(const KernelIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		for(size_t dd=0; dd<this->m_dim; dd++) {
			if(this->m_pos[dd] < other.m_pos[dd])
				return true;
		}

		return false;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>(const KernelIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		for(size_t dd=0; dd<this->m_dim; dd++) {
			if(this->m_pos[dd] > other.m_pos[dd])
				return true;
		}

		return false;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or before the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator<=(const KernelIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this < other;
	};
	
	/**
	 * @brief If the parents are different then false, if they are the same, 
	 * returns whether this iterator is the same or after the other. 
	 *
	 * @param other
	 *
	 * @return 
	 */
	bool operator>=(const KernelIter& other) const
	{ 
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this > other;
	};

private:
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*((U*)ptr));
	};
	
	std::shared_ptr<const NDArray> parent;

	T (*castget)(void* ptr);
};


}

#endif //ITERATOR_H
