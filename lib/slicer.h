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
	 * @brief Simple (no ROI, no order) Constructor
	 *
	 * @param dim	size of ND array
	 */
	Slicer(size_t ndim, const size_t* dim);
	
	/**
	 * @brief Constructor that takes a dimension and order of ++/-- iteration
	 *
	 * @param dim	size of ND array
	 * @param order	iteration direction, steps will be fastest in the direction
	 * 				of order[0] and slowest in order.back()
	 * @param revorder	Reverse order, in which case the first element of order
	 * 					will have the slowest iteration, and dimensions not
	 * 					specified in order will be faster than those included.
	 */
	Slicer(size_t ndim, const size_t* dim, const std::vector<size_t>& order, 
			bool revorder = false);
	
	/****************************************
	 *
	 * Query Location
	 *
	 ****************************************/

	/**
	 * @brief Are we at the end in a particular dimension
	 *
	 * @param dim	dimension to check
	 *
	 * @return whether we are at the tail end of the particular dimension
	 */
	bool isLineBegin(size_t dim) const { return m_pos[dim] == m_roi[dim].first; };
	
	/**
	 * @brief Are we at the begin in a particular dimension
	 *
	 * @param dim	dimension to check
	 *
	 * @return whether we are at the start of the particular dimension
	 */
	bool isBegin(size_t dim) const { return m_pos[dim] == m_roi[dim].first; };
	
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
	 * @brief Postfix iterator. Iterates in the order dictatored by the dimension
	 * order passsed during construction or by setOrder
	 *
	 * @param int	unused
	 *
	 * @return 	old value of linear position
	 */
	int64_t operator++(int);
	

	/**
	 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
	 * order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	int64_t operator++();
	
	/**
	 * @brief Postfix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	old value of linear position
	 */
	int64_t operator--(int);
	
	/**
	 * @brief Prefix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	int64_t operator--();

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
	 * @brief Jump to the given position
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
	 * @brief Get both ND position and linear position. Same as get(vector) but
	 * easier name to remember.
	 *
	 * @param ndpos	Output, ND position
	 *
	 * @return linear position
	 */
	std::vector<int64_t> index() const
	{
		return m_pos;
	};

	/***********************************************
	 *
	 * Modification
	 *
	 **********************************************/

	/**
	 * @brief Updates dimensions of target nd array
	 *
	 * @param dim	size of nd array, number of dimesions given by dim.size()
	 */
	void updateDim(size_t ndim, const size_t* dim);

	/**
	 * @brief Sets the region of interest. During iteration or any motion the
	 * position will not move outside the specified range
	 *
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(const std::vector<std::pair<int64_t, int64_t>>& roi);

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
};

} // npl

#endif //SLICER_H

