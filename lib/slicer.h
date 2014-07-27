/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#ifndef SLICER_H
#define SLICER_H

#include <vector>
#include <cstdint>
#include <cstddef>

using namespace std;

namespace npl {

/**
 * @brief This class is used to slice an image in along a dimension, and to 
 * step an arbitrary direction in an image. Order may be any size from 0 to the
 * number of dimensions. The first member of order will be the fastest moving,
 * and the last will be the slowest. Any not dimensions not included in the order
 * vector will be slower than the last member of order
 */
class Slicer 
{
public:
	
	/****************************************
	 *
	 * Constructors
	 *
	 ****************************************/

	/**
	 * @brief Default Constructor, max a length 1, dimension 1 slicer
	 */
	Slicer();

	/**
	 * @brief Constructor, takses the number of dimensions and the size of the
	 * image.
	 *
	 * @param ndim	size of ND array
	 * @param dim	array providing the size in each dimension 
	 */
	Slicer(size_t ndim, const size_t* dim);
	
	/****************************************
	 *
	 * Query Location
	 *
	 ****************************************/
	
	/**
	 * @brief Are we at the begining of iteration?
	 *
	 * @return true if we are at the begining
	 */
	bool isBegin() const { return m_linpos==m_linfirst; };

	/**
	 * @brief Are we at the end of iteration? Note that this will be 1 past the
	 * end, as typically is done in c++
	 *
	 * @return true if we are at the end
	 */
	bool isEnd() const { return m_end; };
	
	/**
	 * @brief Are we at the end of iteration? Note that this will be 1 past the
	 * end, as typically is done in c++
	 *
	 * @return true if we are at the end
	 */
	bool eof() const { return m_end; };

	/*************************************
	 * Movement
	 ***********************************/

	/**
	 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
	 * order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	Slicer& operator++();
	
	/**
	 * @brief Prefix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	Slicer& operator--();

	/**
	 * @brief Are we at the begining of iteration?
	 *
	 */
	void goBegin();

	/**
	 * @brief Jump to the end of iteration.
	 *
	 */
	void goEnd();

	/**
	 * @brief Jump to the given position, additional values in newpos beyond dim
	 * will be ignored. Any values missing due to ndim > len will be treated as
	 * zeros.
	 *
	 * @param newpos	location to move to
	 */
	void goIndex(size_t len, int64_t* newpos);
	
	/**
	 * @brief Jump to the given position
	 *
	 * @param newpos	location to move to
	 */
	void goIndex(std::vector<int64_t> newpos);

	/****************************************
	 *
	 * Actually get the linear location
	 *
	 ***************************************/

	/**
	 * @brief dereference operator. Returns the linear position in the array 
	 * given the n-dimensional position.
	 *
	 * @return 
	 */
	inline
	int64_t operator*() const { return m_linpos; };
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be 
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ. 
	 *
	 * @param ndim size of index 
	 * @param index output index variable
	 */
	void index(size_t len, int64_t* index) const;

	/***********************************************
	 *
	 * Modification
	 *
	 **********************************************/

	/**
	 * @brief Sets the region of interest. During iteration or any motion the
	 * position will not move outside the specified range. Extra elements in 
	 * roi beyond the number of dimensions, are ignored
	 *
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(const std::vector<std::pair<int64_t, int64_t>>& roi);

	/**
	 * @brief Sets the region of interest. During iteration or motion the
	 * position will not move outside the specified range
	 *
	 * @param len	Length of both lower and upper arrays.
	 * @param lower	Coordinate at lower bound of bounding box. 
	 * @param upper	Coordinate at upper bound of bounding box. 
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(size_t len, const int64_t* lower, const int64_t* upper);

	/**
	 * @brief Sets the order of iteration from ++/-- operators
	 *
	 * @param order	vector of priorities, with first element being the fastest
	 * iteration and last the slowest. All other dimensions not used will be 
	 * slower than the last
	 * @param revorder	Reverse order, in which case the first element of order
	 * 					will have the slowest iteration, and dimensions not
	 * 					specified in order will be faster than those included.
	 */
	void setOrder(const std::vector<size_t>& order, bool revorder = false);

	/**
	 * @brief Returns the array giving the order of dimension being traversed.
	 * So 3,2,1,0 would mean that the next point in dimension 3 will be next,
	 * when wrapping the next point in 2 is visited, when that wraps the next
	 * in one and so on. 
	 *
	 * @return Order of dimensions
	 */
	const std::vector<size_t>& getOrder() const { return m_order; } ;

protected:
	
	/******************************************
	 *
	 * Offset, useful to kernel processing
	 *
	 ******************************************/
	
	size_t m_linpos;
	size_t m_linfirst;
	size_t m_linlast;
	bool m_end;
	std::vector<size_t> m_order;
	std::vector<int64_t> m_pos;
	std::vector<std::pair<int64_t,int64_t>> m_roi;

	// these might benefit from being constant
	std::vector<size_t> m_sizes;
	std::vector<size_t> m_strides;
	size_t m_total;

	void updateLinRange();

	/**
	 * @brief Updates dimensions of target nd array
	 *
	 * @param dim	size of nd array, number of dimesions given by dim.size()
	 */
	void updateDim(size_t ndim, const size_t* dim);
};

} // npl

#endif //SLICER_H

