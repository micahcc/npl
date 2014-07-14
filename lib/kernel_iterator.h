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
#include <stdexcept>
#include <cassert>

using namespace std;

namespace npl {

enum BoundaryMethodT {OVER_ROI, WRAP, CLAMP};

/**
 * @brief This class is used to slice an image in along a dimension, and to 
 * step an arbitrary direction in an image. Order may be any size from 0 to the
 * number of dimensions. The first member of order will be the fastest moving,
 * and the last will be the slowest. Any not dimensions not included in the order
 * vector will be slower than the last member of order. 
 *
 * Key variables are
 *
 * dim 		Dimension (size) of memory block.
 * krange 	Range of offset values. Each pair indicats a min and max
 * 					in the i'th dimension. So {{-3,0}, {-3,3},{0,3}} would
 * 					indicate a kernel from (X-3,Y-3,Z+0) to (X+0,Y+3,Z+3).
 * 					Ranges must include zero.
 * roi 		Range of region of interest. Pairs indicates the range 
 * 					in i'th dimension, so krange = {{1,5},{0,9},{32,100}}
 * 					would cause the iterator to range from (1,0,32) to
 * 					(5,9,100)
 */
class kernel_iterator 
{
public:
	
	/****************************************
	 *
	 * Constructors
	 *
	 ****************************************/

	/**
	 * @brief Default Constructor, size 1, dimension 1 slicer, krange = 0
	 */
	kernel_iterator();
	
	/**
	 * @brief Constructs a iterator with the given dimensions and bounding 
	 * box over the full area. Kernel will range from 
	 * [kRange[0].first, kRange[0].second] 
	 * [kRange[1].first, kRange[1].second] 
	 * ....
	 *
	 *
	 * @param dim	size of ND array
	 * @param kRange Range to iterate over. This determines the offset from 
	 * center that will will traverse.
	 */
	kernel_iterator(const std::vector<size_t>& dim, 
			const std::vector<std::pair<int64_t, int64_t>>& krange);
	
	/**
	 * @brief Constructs a iterator with the given dimensions and bounding 
	 * box over the full area. Kernel will range from 
	 * [kRange[0].first, kRange[0].second] 
	 * [kRange[1].first, kRange[1].second] 
	 * ....
	 *
	 *
	 * @param dim	size of ND array
	 * @param kradius Radius around center. Range will include [-R,R] 
	 * center that will will traverse.
	 */
	kernel_iterator(const std::vector<size_t>& dim, 
			const std::vector<size_t>& kradius);
	
	/**
	 * @brief Constructs a iterator with the given dimensions and bounding 
	 * box over the full area. Kernel will range from 
	 * [kRange[0].first, kRange[0].second] 
	 * [kRange[1].first, kRange[1].second] 
	 * ....
	 *
	 *
	 * @param dim	size of ND array
	 * @param kRange Range to iterate over. This determines the offset from 
	 * center that will will traverse.
	 * @param roi	min/max, roi is pair<size_t,size_t> = [min,max] 
	 */
	kernel_iterator(const std::vector<size_t>& dim, 
			const std::vector<std::pair<int64_t, int64_t>>& krange,
			const std::vector<std::pair<size_t,size_t>>& roi);
	
	/**
	 * @brief Constructs a iterator with the given dimensions and bounding 
	 * box over the full area. Kernel will range from 
	 * [kRange[0].first, kRange[0].second] 
	 * [kRange[1].first, kRange[1].second] 
	 * ....
	 *
	 *
	 * @param dim	size of ND array
	 * @param kradius Radius around center. Range will include [-R,R] 
	 * center that will will traverse.
	 * @param roi	min/max, roi is pair<size_t,size_t> = [min,max] 
	 */
	kernel_iterator(const std::vector<size_t>& dim, 
			const std::vector<size_t>& kradius,
			const std::vector<std::pair<size_t,size_t>>& roi);
	
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
	bool isEnd(size_t dim) const { return m_pos[m_center][dim] == m_roi[dim].second; };
	
	/**
	 * @brief Are we at the begin in a particular dimension
	 *
	 * @param dim	dimension to check
	 *
	 * @return whether we are at the start of the particular dimension
	 */
	bool isBegin(size_t dim) const { return m_pos[m_center][dim] == m_roi[dim].first; };
	
	/**
	 * @brief Are we at the begining of iteration?
	 *
	 * @return true if we are at the begining
	 */
	bool isBegin() const { return m_linpos[m_center] == m_begin; };

	/**
	 * @brief Are we at the end of iteration? Note that this will be 1 past the
	 * end, as typically is done in c++
	 *
	 * @return true if we are at the end
	 */
	bool isEnd() const { return m_end; };


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
	size_t operator++(int);
	

	/**
	 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
	 * order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	size_t operator++();
	
	/**
	 * @brief Postfix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	old value of linear position
	 */
	size_t operator--(int);
	
	/**
	 * @brief Prefix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	size_t operator--();

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
	void goIndex(const std::vector<size_t>& newpos, bool* outside = NULL);

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
	size_t operator*() const { return m_linpos[m_center]; };
	
	/**
	 * @brief Same as dereference operator. Returns the linear position in the 
	 * array given the n-dimensional position.
	 *
	 * @return 
	 */
	inline
	size_t get() const 
	{ 
		assert(!m_end);
		return m_linpos[m_center]; 
	};

	/**
	 * @brief Get both ND position and linear position
	 *
	 * @param ndpos	Output, ND position
	 *
	 * @return linear position
	 */
	size_t get(std::vector<size_t>& ndpos) const
	{
		assert(!m_end);
		ndpos.assign(m_pos[m_center].begin(), m_pos[m_center].end());
		return m_linpos[m_center];
	};
	
	/**
	 * @brief Same as dereference operator. Returns the linear position in the 
	 * array given the n-dimensional position.
	 *
	 * @return 
	 */
	inline
	size_t get(size_t kit) const { 
		assert(!m_end);
		assert(kit < m_numoffs);
		return m_linpos[kit]; 
	};

	/**
	 * @brief Get both ND position and linear position
	 *
	 * @param ndpos	Output, ND position
	 *
	 * @return linear position
	 */
	size_t get(size_t kit, std::vector<size_t>& ndpos) const
	{
		assert(!m_end);
		assert(kit < m_numoffs);
		ndpos.assign(m_pos[kit].begin(), m_pos[kit].end());
		return m_linpos[kit]; 
	};

	/**
	 * @brief Get both ND position and linear position
	 *
	 * @param ndpos	Output, ND position
	 *
	 * @return linear position
	 */
	size_t ksize() const
	{
		return m_numoffs;
	};

	/**
	 * @brief All around intializer. Sets all internal variables.
	 *
	 * @param dim 		Dimension (size) of memory block.
	 * @param krange 	Range of offset values. Each pair indicats a min and max
	 * 					in the i'th dimension. So {{-3,0}, {-3,3},{0,3}} would
	 * 					indicate a kernel from (X-3,Y-3,Z+0) to (X+0,Y+3,Z+3).
	 * 					Ranges must include zero.
	 * @param roi 		Range of region of interest. Pairs indicates the range 
	 * 					in i'th dimension, so krange = {{1,5},{0,9},{32,100}}
	 * 					would cause the iterator to range from (1,0,32) to
	 * 					(5,9,100)
	 */
	void initialize(const std::vector<size_t>& dim, 
			const std::vector<std::pair<int64_t, int64_t>>& krange,
			const std::vector<std::pair<size_t,size_t>>& roi);

protected:
	size_t m_direction;
	size_t m_dim;
	std::vector<size_t> m_size;
	std::vector<size_t> m_strides;

	// for begin/end calculation
	std::vector<std::pair<size_t,size_t>> m_roi;

	int64_t m_fradius; //forward radius, should be positive
	int64_t m_rradius; //reverse radius, should be positive

	// for each of the neighbors we need to know 
	size_t m_numoffs;
	std::vector<std::vector<int64_t>> m_offs;
	size_t m_center;
	bool m_end;
	size_t m_begin;;

	std::vector<std::vector<size_t>> m_pos;
	std::vector<size_t> m_linpos;

};

} // npl

#endif //SLICER_H
