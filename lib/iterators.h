/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file iterators.h Iterators for images. Each is templated by the type that
 * the pixels are viewed as (rather than the actual stored pixel type).
 *
 *****************************************************************************/

#ifndef ITERATOR_H
#define ITERATOR_H

#include <stdexcept>
#include <memory>
#include "ndarray.h"
#include "slicer.h"
#include "npltypes.h"

namespace npl {


 /** \defgroup Iterators Iterators for NDarray/Image
 *
 * Iterators are similar to accessors in that they perform casting, however
 * they also advance through pixels. Thus they are designed to walk over the
 * image or array space.
 *
 * A simple example:
 * \code{.cpp}
 * double sum = 0;
 * for(FlatIter<double> it(img); !it.isEnd(); ++it) {
 *   sum += *it;
 * }
 * \endcode
 *
 * FlatIter doesn't maintain an ND-index however, so if you need index
 * information try NDIter. So for instance the code below would set pixels
 * to their x index:
 * \code{.cpp}
 * vector<int64_t> ind(img->ndim());
 * for(NDIter<double> it(img); !it.eof(); ++it) {
 *    it.index(ind);
 *    it.set(ind[0]);
 * }
 * \endcode
 *
 * @{
 */

/**
 * @brief Flat iterator for NDArray. No information is kept about
 * the current ND index. Just goes through all data. This casts the output to the
 * type specified using T.
 *
 * @tparam T
 */
template <typename T = double>
class FlatIter
{
public:
    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    FlatIter() {};

	FlatIter(std::shared_ptr<NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<NDArray> in)
    {
        parent = in;
        m_linpos = 0;
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
    }

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	FlatIter& operator++()
	{
		++m_linpos;
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	FlatIter& operator--()
	{
		--m_linpos;
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(m_linpos);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{
		auto ptr = parent->__getAddr(m_linpos);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	void set(T v) const
	{
		auto ptr = parent->__getAddr(m_linpos);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		castset(ptr, v);
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
template <typename T = double>
class FlatConstIter
{
public:
    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    FlatConstIter() {};

	FlatConstIter(std::shared_ptr<const NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<const NDArray> in)
    {
        parent = in;
        m_linpos = 0;
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
    }

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	FlatConstIter& operator++()
	{
		++m_linpos;
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	FlatConstIter& operator--()
	{
		--m_linpos;
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(m_linpos);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{
		auto ptr = parent->__getAddr(m_linpos);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
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
 * @brief Constant iterator for NDArray. Typical usage calls for NDConstIter it(array); it++; *it
 *
 * @tparam T
 */
template <typename T = double>
class NDConstIter : public Slicer
{
public:
    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    NDConstIter() {};

	NDConstIter(std::shared_ptr<const NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<const NDArray> in)
    {
		setDim(in->ndim(), in->dim());
        parent = in;
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
				throw std::invalid_argument("Unknown type to NDConstIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	NDConstIter& operator++()
	{
		Slicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	NDConstIter& operator--()
	{
		Slicer::operator--();
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
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
	bool operator==(const NDConstIter& other) const
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
	bool operator!=(const NDConstIter& other) const
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
	bool operator<(const NDConstIter& other) const
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
	bool operator>(const NDConstIter& other) const
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
	bool operator<=(const NDConstIter& other) const
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
	bool operator>=(const NDConstIter& other) const
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
template <typename T = double>
class NDIter : public Slicer
{
public:
    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    NDIter() {};

	NDIter(std::shared_ptr<NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<NDArray> in)
    {
        parent = in;
        setDim(in->ndim(), in->dim());
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
				throw std::invalid_argument("Unknown type to NDIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	NDIter& operator++()
	{
		Slicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	NDIter& operator--()
	{
		Slicer::operator--();
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	void set(T v)
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		this->castset(ptr, v);
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
	bool operator==(const NDIter& other) const
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
	bool operator!=(const NDIter& other) const
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
	bool operator<(const NDIter& other) const
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
	bool operator>(const NDIter& other) const
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
	bool operator<=(const NDIter& other) const
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
	bool operator>=(const NDIter& other) const
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
 * @brief To maintain backward compatability, I have saved the OrderIter
 * and OrderConstIter name, but eventually they may be deprecated.
 *
 * @tparam T PixelType to read out
 */
template<class T> using OrderIter = NDIter<T>;

/**
 * @brief To maintain backward compatability, I have saved the OrderIter
 * and OrderConstIter name, but eventually they may be deprecated.
 *
 * @tparam T PixelType to read out
 */
template<class T> using OrderConstIter = NDConstIter<T>;

/**
 * @brief Constant iterator for NDArray. This is slightly different from order
 * iterator in that the ROI may be broken down into chunks. When the end of a
 * chunk is reached, no more iteration can be performed until nextChunk() is
 * called. isEnd() will return false until nextChunk() is called while the
 * current chunk is at the last available. Note that if setBreaks
 * uses an array that is smaller than input dimension, then the whole of
 * the dimension will be used.
 *
 * Usage:
 *
 * ChunkConstIter it(imag);
 * it.setBreaks({0,0,10}); // stop every 10 values of z
 * it.setBreaks({0,0,0,1}); // stop at the end of each volume
 * it.setBreaks({1,1,1,0}); // stop at the end of each time-series
 * it.setBreaks({1,1,1}); // same as above (default = 0)
 * it.setLineChunk(0); // iterates along lines in dimension 0
 *
 * for(it.goBegin(); !it.isEnd(); it.nextChunk()) {
 * 	for(it.goBegin(); !it.isChunkEnd(); ++it) {
 *		it.get();
 *		...
 * 	}
 * }
 *
 * @tparam T
 */
template <typename T = double>
class ChunkConstIter : public ChunkSlicer
{
public:

    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    ChunkConstIter() {};

	ChunkConstIter(std::shared_ptr<const NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<const NDArray> in)
    {
		setDim(in->ndim(), in->dim());
        parent = in;
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
				throw std::invalid_argument("Unknown type to ChunkConstIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	ChunkConstIter& operator++()
	{
		ChunkSlicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	ChunkConstIter& operator--()
	{
		ChunkSlicer::operator--();
		return *this;
	};

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	ChunkConstIter& nextChunk()
	{
		ChunkSlicer::nextChunk();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	ChunkConstIter& prevChunk()
	{
		ChunkSlicer::prevChunk();
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(ChunkSlicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{
		auto ptr = parent->__getAddr(ChunkSlicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Go to beginning of iteration
	 */
	void goBegin() { ChunkSlicer::goBegin(); };

	/**
	 * @brief Go to end of iteration
	 */
	void goEnd() { ChunkSlicer::goEnd(); };

	/**
	 * @brief Are we one past the last element?
	 */
	bool isEnd() const { return ChunkSlicer::isEnd(); };

	/**
	 * @brief Are we one past the last element?
	 */
	bool eof() const { return ChunkSlicer::isEnd(); };

	/**
	 * @brief Are we at the first element
	 */
	bool isBegin() const { return ChunkSlicer::isBegin(); };

	/**
	 * @brief Whether the position and parent are the same as another
	 *
	 * @param other
	 *
	 * @return
	 */
	bool operator==(const ChunkConstIter& other) const
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
	bool operator!=(const ChunkConstIter& other) const
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
	bool operator<(const ChunkConstIter& other) const
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
	bool operator>(const ChunkConstIter& other) const
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
	bool operator<=(const ChunkConstIter& other) const
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
	bool operator>=(const ChunkConstIter& other) const
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
 * Unlike NDIter it breaks the NDArray into chunks. Iteration
 * stops at the end of each chunk until nextChunk is called. By default only
 * one chunk is used, equal to the entire image. Use setBreaks() to set the
 * frequency of chunks. The input to setBreaks() is an array of integers, where
 * the next chunk occurs when dist%break == 0, with the special property that
 * break=0 indicates no breaks for the dimension. So for a 3D image,
 * setBreaks({1,0,0}) will stop every time a new x-values is reached. Note that
 * the affects the order of iteration, so x cannot be the fastest iterator.
 * setChunkSize() is an alias for setBreaks. Note that in cases where
 * break%size != 0, for example image size = 5 and break = 3, chunk sizes will
 * differ during the course of iteration!
 *
 * Order may still be set to decide the order of iteration. The first member
 * of setOrder will be the fastest moving, and the last will be the slowest.
 * Any dimensions not included in the order vector will be slower than the last
 * member of order. Note that order will not necessarily be strictly obeyed
 * when more than one chunk size is > 0. For isntance if setBreaks({1,1,0}),
 * setOrder({0,1,2}) are used, then all of 2 will visited before we iterate in
 * 0 or 1, because otherwise we would be stepping across chunks.
 *
 * @tparam T
 */
template <typename T = double>
class ChunkIter : public ChunkSlicer
{
public:

    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    ChunkIter() {};

	ChunkIter(std::shared_ptr<NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<NDArray> in)
    {
        parent = in;
        setDim(in->ndim(), in->dim());
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
				throw std::invalid_argument("Unknown type to ChunkIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	ChunkIter& operator++()
	{
		ChunkSlicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	ChunkIter& operator--()
	{
		ChunkSlicer::operator--();
		return *this;
	};

	/**
	 * @brief NextChunk
	 *
	 * @return new value
	 */
	ChunkIter& nextChunk()
	{
		ChunkSlicer::nextChunk();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	ChunkIter& prevChunk()
	{
		ChunkSlicer::prevChunk();
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(ChunkSlicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	T get() const
	{
		auto ptr = parent->__getAddr(ChunkSlicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return current value
	 */
	void set(T v)
	{
		auto ptr = parent->__getAddr(ChunkSlicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		this->castset(ptr, v);
	};

	/**
	 * @brief Go to beginning of iteration
	 */
	void goBegin() { ChunkSlicer::goBegin(); };

	/**
	 * @brief Go to end of iteration
	 */
	void goEnd() { ChunkSlicer::goEnd(); };

	/**
	 * @brief Are we one past the last element?
	 */
	bool isEnd() const { return ChunkSlicer::isEnd(); };

	/**
	 * @brief Are we one past the last element?
	 */
	bool eof() const { return ChunkSlicer::isEnd(); };

	/**
	 * @brief Are we at the first element
	 */
	bool isBegin() const { return ChunkSlicer::isBegin(); };

	/**
	 * @brief Whether the position and parent are the same as another
	 *
	 * @param other
	 *
	 * @return
	 */
	bool operator==(const ChunkIter& other) const
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
	bool operator!=(const ChunkIter& other) const
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
	bool operator<(const ChunkIter& other) const
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
	bool operator>(const ChunkIter& other) const
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
	bool operator<=(const ChunkIter& other) const
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
	bool operator>=(const ChunkIter& other) const
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
 * The primary functions are operator[] which allows you to get the k'th offset
 * value, indexK() which returns the index of the k'th offset, and offsetK()
 * which returns the offset of k'th offset, IE where it is in relation to the
 * center.
 *
 * Note that order is the not the default order. If you use using this iterator
 * with another iterator, be sure to setOrder(it.getOrder()) so that both
 * iterators have the same directions.
 *
 * There are no 'set' functions so this is const iterator.
 *
 * @tparam T
 */
template <typename T = double>
class KernelIter : public KSlicer
{
public:

    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    KernelIter() {};

    KernelIter(std::shared_ptr<const NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<const NDArray> in)
    {
		setDim(in->ndim(), in->dim());
        parent = in;
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
				throw std::invalid_argument("Unknown type to NDConstIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value
	 */
	KernelIter& operator++()
	{
		KSlicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value
	 */
	KernelIter& operator--()
	{
		KSlicer::operator--();
		return *this;
	};

	/**
	 * @brief Dereference operator, get center pixel value
	 *
	 * @return current value
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(KSlicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Same as dereference operator, get center pixel value
	 *
	 * @return current value
	 */
	T getC() const
	{
		auto ptr = parent->__getAddr(KSlicer::getC());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference (get) the pixel at the k'th offset position. To
     * figure out WHERE this pixel is in relation to the center use offsetK
	 *
	 * @return current value at offset k
	 */
	T getK(int64_t k) const
	{
		auto ptr = parent->__getAddr(KSlicer::getK(k));
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Dereference (get) the pixel at the k'th offset position. To
     * figure out WHERE this pixel is in relation to the center use offsetK
	 *
	 * @return current value at offset k
	 */
	T operator[](int64_t k) const
	{
		auto ptr = parent->__getAddr(KSlicer::getK(k));
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
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

/**
 * @brief This class is used to iterate through an 3D array, where each point
 * then has has multiple higher dimensional variable. This is analogous to
 * Vector3DView, where even if there are multiple higher dimensions they are all
 * alligned into a single vector at each 3D point. This makes them easier to
 * than simple iteration in N-dimensions.
 *
 * @tparam T
 */
template <typename T = double>
class Vector3DIter : public Slicer
{
public:

    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
    Vector3DIter() {};

	Vector3DIter(std::shared_ptr<NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<NDArray> in)
    {
		setDim(in->ndim(), in->dim());
        parent = in;

		// iterate through the first 3 dimensions
		std::vector<std::pair<int64_t,int64_t>> roi(in->ndim());
		for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
			roi[ii].first = 0;
			roi[ii].second = in->dim(ii)-1;
		}

		// don't iterate in higher dimensions
		for(size_t ii=3; ii<in->ndim(); ii++) {
			roi[ii].first = 0;
			roi[ii].second = 0;
		}
		this->setROI(roi);

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
				throw std::invalid_argument("Unknown type to NDIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value in 0th element of vector
	 */
	Vector3DIter& operator++()
	{
		Slicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value in 0th element of vector
	 */
	Vector3DIter& operator--()
	{
		Slicer::operator--();
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return Value in 0th element of vector
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Get value at ith element of vector
	 *
	 * @return current value
	 */
	T get(int64_t i = 0) const
	{
		auto ptr = parent->__getAddr(Slicer::operator*()+i);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Get value at ith element of vector
	 *
	 * @return current value
	 */
	T operator[](int64_t i) const
	{
		auto ptr = parent->__getAddr(Slicer::operator*()+i);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Set the value at the ith element of thevector
	 *
	 */
	void set(int64_t i, T v)
	{
		auto ptr = parent->__getAddr(Slicer::operator*()+i);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		this->castset(ptr, v);
	};

	/**
	 * @brief Set the value at the 0th element of the vector
	 *
	 */
	void set(T v)
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		this->castset(ptr, v);
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
	bool operator==(const Vector3DIter& other) const
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
	bool operator!=(const Vector3DIter& other) const
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
	bool operator<(const Vector3DIter& other) const
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
	bool operator>(const Vector3DIter& other) const
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
	bool operator<=(const Vector3DIter& other) const
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
	bool operator>=(const Vector3DIter& other) const
	{
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this > other;
	};

	size_t tlen() const { return this->parent->tlen(); };

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
 * @brief This class is used to iterate through an 3D array, where each point
 * then has has multiple higher dimensional variable. This is analogous to
 * Vector3DView, where even if there are multiple higher dimensions they are all
 * alligned into a single vector at each 3D point. This makes them easier to
 * than simple iteration in N-dimensions. This is the constant version
 *
 * @tparam T
 */
template <typename T = double>
class Vector3DConstIter : public Slicer
{
public:

    /**
     * @brief Default constructor. Note, this will segfault if you don't use
     * setArray to set the target NDArray/Image.
     */
	Vector3DConstIter() {};

	Vector3DConstIter(std::shared_ptr<const NDArray> in)
	{
        setArray(in);
	};

    void setArray(ptr<const NDArray> in)
    {
		setDim(in->ndim(), in->dim());
        parent = in;

		// iterate through the first 3 dimensions
		std::vector<std::pair<int64_t,int64_t>> roi(in->ndim());
		for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
			roi[ii].first = 0;
			roi[ii].second = in->dim(ii)-1;
		}

		// don't iterate in higher dimensions
		for(size_t ii=3; ii<in->ndim(); ii++) {
			roi[ii].first = 0;
			roi[ii].second = 0;
		}
		this->setROI(roi);

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
			default:
			case UNKNOWN_TYPE:
				castget = castgetStatic<uint8_t>;
				throw std::invalid_argument("Unknown type to NDIter");
				break;
		}
    };

	/**
	 * @brief Prefix increment operator
	 *
	 * @return new value in 0th element of vector
	 */
	Vector3DConstIter& operator++()
	{
		Slicer::operator++();
		return *this;
	};

	/**
	 * @brief Prefix decrement operator
	 *
	 * @return new value in 0th element of vector
	 */
	Vector3DConstIter& operator--()
	{
		Slicer::operator--();
		return *this;
	};

	/**
	 * @brief Dereference operator
	 *
	 * @return Value in 0th element of vector
	 */
	T operator*() const
	{
		auto ptr = parent->__getAddr(Slicer::operator*());
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Get value at ith element of vector
	 *
	 * @return current value
	 */
	T get(int64_t i = 0) const
	{
		auto ptr = parent->__getAddr(Slicer::operator*()+i);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
	};

	/**
	 * @brief Get value at ith element of vector
	 *
	 * @return current value
	 */
	T operator[](int64_t i) const
	{
		auto ptr = parent->__getAddr(Slicer::operator*()+i);
		assert(ptr >= this->parent->__getAddr(0) &&
				ptr < this->parent->__getAddr(this->parent->elements()));
		return castget(ptr);
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
	bool operator==(const Vector3DConstIter& other) const
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
	bool operator!=(const Vector3DConstIter& other) const
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
	bool operator<(const Vector3DConstIter& other) const
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
	bool operator>(const Vector3DConstIter& other) const
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
	bool operator<=(const Vector3DConstIter& other) const
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
	bool operator>=(const Vector3DConstIter& other) const
	{
		if(parent != other.parent)
			return false;

		if(*this == other)
			return true;

		return *this > other;
	};

	size_t tlen() const { return this->parent->tlen(); };

private:
	template <typename U>
	static T castgetStatic(void* ptr)
	{
		return (T)(*((U*)ptr));
	};

	std::shared_ptr<const NDArray> parent;

	T (*castget)(void* ptr);
};

/** @} */

}

#endif //ITERATOR_H
