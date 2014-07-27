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

#ifndef KERNEL_SLICER_H
#define KERNEL_SLICER_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cassert>

using namespace std;

namespace npl {

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
class KSlicer
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
	KSlicer();

	/**
	 * @brief Constructs a iterator with the given dimensions 
	 *
	 * @param ndim	Number of dimensions 
	 * @param dim	size of ND array
	 */
	KSlicer(size_t ndim, const size_t* dim);

	/***************************************************
	 *
	 * Basic Settings
	 *
	 ***************************************************/

	/**
	 * @brief Sets the region of interest. During iteration or any motion the
	 * position will not move outside the specified range
	 *
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(const std::vector<std::pair<int64_t, int64_t>> roi = {});
	
	/**
	 * @brief Set the order of iteration, in terms of which dimensions iterate
	 * the fastest and which the slowest.
	 *
	 * @param order order of iteration. {0,1,2} would mean that dimension 0 (x)
	 * would move the fastest and 2 the slowest. If the image is a 5D image then
	 * that unmentioned (3,4) would be the slowest. 
	 * @param revorder Reverse the speed of iteration. So the first dimension 
	 * in the order vector would in fact be the slowest and un-referenced 
	 * dimensions will be the fastest. (in the example for order this would be
	 * 4 and 3).
	 */
	void setOrder(const std::vector<size_t> order = {}, bool revorder = false);
	
	/**
	 * @brief Returns the array giving the order of dimension being traversed.
	 * So 3,2,1,0 would mean that the next point in dimension 3 will be next,
	 * when wrapping the next point in 2 is visited, when that wraps the next
	 * in one and so on. 
	 *
	 * @return Order of dimensions
	 */
	const std::vector<size_t>& getOrder() const { return m_order; } ;
	
	/**
	 * @brief Set the radius of the kernel window. All directions will 
	 * have equal distance, with the radius in each dimension set by the 
	 * magntitude of the kradius vector. So if kradius = {2,1,0} then 
	 * dimension 0 (x) will have a radius of 2, dimension 1 (y) will have 
	 * a readius of 1 and dimension 2 will have a radius of 0 (won't step
	 * out from the middle at all).
	 *
	 * @param kradius vector of radii in the given dimension. Unset values
	 * assumed to be 0. So a 10 dimensional image with 3 values will have
	 * non-zero values for x,y,z but 0 values in higher dimensions
	 */
	void setRadius(std::vector<size_t> kradius = {});
	
	/**
	 * @brief Set the radius of the kernel window. All directions will 
	 * have equal distance in all dimensions. So if kradius = 2 then 
	 * dimension 0 (x) will have a radius of 2, dimension 2 (y) will have 
	 * a readius of 2 and so on. Warning images may have more dimensions
	 * than you know, so if the image has a dimension that is only size 1
	 * it will have a radius of 0, but if you didn't know you had a 10D image
	 * all the dimensions about to support the radius will.
	 *
	 * @param radii in all directions. 
	 */
	void setRadius(size_t kradius);
	
	/**
	 * @brief Set the ROI from the center of the kernel. The first value 
	 * should be <= 0, the second should be >= 0. The ranges are inclusive.
	 * So if kradius = {{-1,1},{0,1},{-1,0}}, in the x dimension values will 
	 * range from center - 1 to center + 1, y indices will range from center 
	 * to center + 1, and z indices will range from center-1 to center. 
	 * Kernel will range from
	 * [kRange[0].first, kRange[0].second]
	 * [kRange[1].first, kRange[1].second]
	 * ...
	 *
	 * @param Vector of [inf, sup] in each dimension. Unaddressed (missing) 
	 * values are assumed to be [0,0]. 
	 */
	void setWindow(const std::vector<std::pair<int64_t, int64_t>>& krange);

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
	bool isBegin() const { return m_linpos[m_center] == m_begin; };

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
	KSlicer& operator++();
	
	/**
	 * @brief Prefix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	KSlicer& operator--();

	/**
	 * @brief Go to the beginning
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
	void goIndex(const std::vector<int64_t>& newpos);

	/****************************************
	 *
	 * Actually get the linear location
	 *
	 ***************************************/

	/**
	 * @brief Get image linear index of center.
	 *
	 * @return Linear index
	 */
	inline
	int64_t center() const
	{
		assert(!m_end);
		return m_linpos[m_center];
	};
	
	/**
	 * @brief Get image linear index of center. Identitcal to center() just
	 * more confusing
	 *
	 * @return Linear index
	 */
	inline
	int64_t operator*() const
	{
		assert(!m_end);
		return m_linpos[m_center];
	};


	/**
	 * @brief Places the first len dimensions of ND-position in the given
	 * array. If the number
	 * of dimensions exceed the len then the additional dimensions will be 
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ. 
	 *
	 * @param ndim size of index 
	 * @param index output index variable
	 */
	void center_index(size_t len, int64_t* index) const;
	
	/**
	 * @brief Get index of i'th kernel (center-offset) element.
	 *
	 * @return linear position
	 */
	inline
	int64_t offset(int64_t kit) const {
		assert(!m_end);
		assert(kit < m_numoffs);
		return m_linpos[kit];
	};
	
	/**
	 * @brief Same as offset(int64_t kit)
	 *
	 * @return linear position
	 */
	inline
	int64_t operator[](int64_t kit) const {
		assert(!m_end);
		assert(kit < m_numoffs);
		return m_linpos[kit];
	};

	/**
	 * @brief Get the ND position of the specified offset (kernel) element.
	 *
	 * @param kit Kernel index
	 * @param len size of index 
	 * @param index output index variable
	 * @param bound report the actual sampled point (ie point after clamping 
	 * position to be in the image. Interior points will be the same, but on
	 * the boundary if you set bound you will only get indices inside the image 
	 * ROI, otherwise you would get values like -1, -1 -1 for radius 1 pos 
	 * 0,0,0
	 *
	 * @return ND position
	 */
	void offset_index(size_t kit, size_t len, int64_t* index, bool bound = true) const;

	/**
	 * @brief Returns the distance from the center projected onto the specified
	 * dimension. So center is {0,0,0}, and {1,2,1} would return 1,2,1 for inputs
	 * dim=0, dim=1, dim=2
	 *
	 * @param kit Which pixel to return distance from
	 * @param dim dimension to get distance in
	 *
	 * @return Offset from center of given pixel (kit)
	 */
	int64_t from_center(size_t kit, size_t dim);
	
	/**
	 * @brief Returns offset from center of specified pixel (kit).
	 *
	 * @param kit Pixel we are referring to
	 * @param len lenght of dindex array
	 * @param dindex output paramter indicating distance of pixel from the 
	 * center of the kernel in each dimension. If this array is shorter than 
	 * the iteration dimensions, only the first len will be filled. If it is 
	 * longer the additional values won't be touched
	 */
	void from_center(size_t kit, size_t len, int64_t* dindex) const;

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

protected:

	/**
	 * @brief All around intializer. Sets all internal variables.
	 *
	 * @param dim 		Dimension (size) of memory block.
	 * @param krange 	Range of offset values. Each pair indicats a min and max
	 * 					in the i'th dimension. So {{-3,0}, {-3,3},{0,3}} would
	 * 					indicate a kernel from (X-3,Y-3,Z+0) to (X+0,Y+3,Z+3).
	 * 					Ranges must include zero.
	 */
	void initialize(size_t ndim, const size_t* dim);

	// order of traversal, constructor initializes
	size_t m_dim; // constructor
	std::vector<size_t> m_size; // constructor
	std::vector<size_t> m_strides; //constructor
	
	// setOrder
	std::vector<size_t> m_order; 

	// setRadius/setWindow
	// for each of the neighbors we need to know
	size_t m_numoffs; // setRadius/setWindow/
	std::vector<std::vector<int64_t>> m_offs; // setRadius/setWindow
	size_t m_center;  // setRadius/setWindow
	int64_t m_fradius; //forward radius, should be positive
	int64_t m_rradius; //reverse radius, should be positive

	// setROI
	std::vector<std::pair<int64_t,int64_t>> m_roi;
	size_t m_begin;

	// goBegin/goEnd/goIndex
	bool m_end;
	std::vector<std::vector<int64_t>> m_pos;
	std::vector<int64_t> m_linpos;

};

} // npl

#endif //SLICER_H
