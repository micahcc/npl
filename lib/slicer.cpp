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
 * @file slicer.cpp Contains definition for Slicer and Chunk slicer, two tools
 * for sequentially walking through ND-spaces. 
 *
 *****************************************************************************/

#include "slicer.h"
#include "basic_functions.h"

#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace npl {

/**
 * @brief Default Constructor, max a length 1, dimension 1 slicer
 */
Slicer::Slicer()
{
	size_t tmp = 1;
	setDim(1, &tmp);
	goBegin();
};

/**
 * @brief Simple (no ROI, no order) Constructor
 *
 * @param dim	size of ND array
 */
Slicer::Slicer(size_t ndim, const size_t* dim)
{
	setDim(ndim, dim);
	goBegin();
};

/**
 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
 * order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
Slicer& Slicer::operator++()
{
	if(isEnd())
		return *this;

	m_end = true; // m_end set to false indicates we have found a non-end incr
	for(size_t ii=0; m_end && ii < m_ndim; ii++){
		size_t dd = m_order[ii];
		if(m_pos[dd] < m_roi[dd].second) {
			m_pos[dd]++;
			m_linpos += m_strides[dd];
			m_end = false; 
		} else {
			// roll over 
			m_linpos -= (m_pos[dd]-m_roi[dd].first)*m_strides[dd];
			m_pos[dd] = m_roi[dd].first;
		}
	}

	return *this;
}

/**
 * @brief Prefix negative  iterator. Iterates in the order dictatored by
 * the dimension order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
Slicer& Slicer::operator--()
{
	if(isBegin())
		return *this;
	m_end = false;

	for(size_t ii=0; ii<m_ndim; ii++){
		size_t dd = m_order[ii];
		if(m_pos[dd] > m_roi[dd].first) {
			m_pos[dd]--;
			m_linpos -= m_strides[dd];
			break;
		} else {
			// roll over
			m_linpos += (m_roi[dd].second-m_pos[dd])*m_strides[dd];
			m_pos[dd] = m_roi[dd].second;
		}
	}

	return *this;
};

/**
 * @brief Updates dimensions of target nd array
 *
 * Invalidates position, Order, ROI gets reset, 
 *
 * @param ndim	size of dim array (the number of image dimensions)
 * @param dim	size of array, number of dimesions given by dim.size()
 */
void Slicer::setDim(size_t ndim, const size_t* dim)
{
	m_linpos = 0;
	m_linfirst = 0;
	m_pos.resize(ndim);
	std::fill(m_pos.begin(), m_pos.end(), 0);
	m_dim.assign(dim, dim+ndim);
	
	m_end = false;
	m_ndim = ndim;

	// reset order
	m_order.resize(ndim);
	for(int64_t ii=0; ii<ndim; ii++)
		m_order[ii] = ((int64_t)ndim)-1-ii;

	// set ROI to max
	m_roi.resize(ndim);
	for(size_t ii = 0; ii<ndim; ii++) {
		m_roi[ii].first = 0; 
		m_roi[ii].second = dim[ii]-1;
	}
	
	// set up strides
	m_strides.resize(ndim);
	m_strides[ndim-1] = 1;
	for(int64_t ii=((int64_t)ndim)-2; ii>=0; ii--) {
		m_strides[ii] = m_strides[ii+1]*dim[ii+1];
	}
};

/**
 * @brief Places the first ndim dimension in the given array. If the number
 * of dimensions exceed the ndim then the additional dimensions will be
 * ignored, if ndim exceeds the dimensionality then index[dim...ndim-1] = 0.
 * In other words index will be completely overwritten in the most sane way
 * possible if the internal dimensions and size index differ.
 *
 * @param len size of index
 * @param index output index variable
 */
void Slicer::index(size_t len, int64_t* index) const
{
	for(size_t ii=0; ii<len; ii++) {
		if(ii<m_ndim) {
			index[ii] = m_pos[ii];
		} else {
			index[ii] = 0;
		}
	}
}

/**
 * @brief Places the first ndim dimension in the given array. If the number
 * of dimensions exceed the ndim then the additional dimensions will be
 * ignored, if ndim exceeds the dimensionality then index[dim...ndim-1] = 0.
 * In other words index will be completely overwritten in the most sane way
 * possible if the internal dimensions and size index differ.
 *
 * @param len size of index
 * @param index output index variable
 */
void Slicer::index(size_t len, int* index) const
{
	for(size_t ii=0; ii<len; ii++) {
		if(ii<m_ndim) {
			index[ii] = m_pos[ii];
		} else {
			index[ii] = 0;
		}
	}
}

/**
 * @brief Places the first ndim dimension in the given array. If the number
 * of dimensions exceed the ndim then the additional dimensions will be
 * ignored, if ndim exceeds the dimensionality then index[dim...ndim-1] = 0.
 * In other words index will be completely overwritten in the most sane way
 * possible if the internal dimensions and size index differ.
 *
 * @param len size of index
 * @param index output index variable
 */
void Slicer::index(size_t len, double* index) const
{
	for(size_t ii=0; ii<len; ii++) {
		if(ii<m_ndim) {
			index[ii] = m_pos[ii];
		} else {
			index[ii] = 0;
		}
	}
}

/**
 * @brief Sets the region of interest, with lower bound of 0.
 * During iteration or any motion the
 * position will not move outside the specified range. Invalidates position.
 *
 * Invalidates position
 *
 * @param len
 * @param roi Size of ROI (which runs from [0,0...] to [roi[0], roi[1]...]
 */
void Slicer::setROI(size_t len, const size_t* roi)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim ; ii++) {
        m_roi[ii].first = 0;
		if(ii < len) {
			// clamp, to be <= 0...sizes[ii]-1
			m_roi[ii].second = clamp<int64_t>(0, m_dim[ii]-1, roi[ii]);
		} else {
			// no specification, just make it all
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_strides[ii]*m_roi[ii].first;
	}

	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
}

/**
 * @brief Sets the region of interest, with lower bound of 0.
 * During iteration or any motion the
 * position will not move outside the specified range. Invalidates position.
 *
 * Invalidates position
 *
 * @param len length of roi array
 * @param roi Size of ROI (which runs from [0,0...] to [roi[0], roi[1]...]
 */
void Slicer::setROI(size_t len, const int64_t* roi)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim ; ii++) {
        m_roi[ii].first = 0;
		if(ii < len) {
			// clamp, to be <= 0...sizes[ii]-1
			m_roi[ii].second = clamp<int64_t>(0, m_dim[ii]-1, roi[ii]);
		} else {
			// no specification, just make it all
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_strides[ii]*m_roi[ii].first;
	}

	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
}

/**
 * @brief Sets the region of interest. During iteration or any motion the
 * position will not move outside the specified range. Invalidates position.
 *
 * Invalidates position
 *
 * @param roi	pair of [min,max] values in the desired hypercube
 */
void Slicer::setROI(const std::vector<std::pair<int64_t, int64_t>>& roi)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim ; ii++) {
		if(ii < roi.size()) {
			// clamp, to be <= 0...sizes[ii]-1
			m_roi[ii].first = clamp<int64_t>(0, m_dim[ii]-1, roi[ii].first);
			m_roi[ii].second = clamp<int64_t>(0, m_dim[ii]-1, roi[ii].second);
		} else {
			// no specification, just make it all
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_strides[ii]*m_roi[ii].first;
	}

	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
}

/**
 * @brief Sets the region of interest. During iteration or any motion the
 * position will not move outside the specified range. Invalidates position.
 * Any missing dimensions will be set to the largest possible region. IE a
 * length=2 lower and upper for a 3D space will have [0, dim[2]] as the range
 *
 * Invalidates position
 *
 * @param len	Length of lower/upper arrays
 * @param lower	Lower bound of ROI (ND-index)
 * @param upper Upper bound of ROI (ND-index)
 */
void Slicer::setROI(size_t len, const int64_t* lower, const int64_t* upper)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim ; ii++) {
		if(ii < len) {
			// clamp, to be <= 0...sizes[ii]-1
			m_roi[ii].first = clamp<int64_t>(0, m_dim[ii]-1, lower[ii]);
			m_roi[ii].second = clamp<int64_t>(0, m_dim[ii]-1, upper[ii]);
		} else {
			// no specification, just make it all
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_strides[ii]*m_roi[ii].first;
	}

	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
}
	
/**
 * @brief Sets the order of iteration from ++/-- operators. Order will be
 * the default (highest to lowest)
 *
 * Invalidates position
 *
 */
void Slicer::setOrder()
{
	m_order.resize(m_ndim);
	for(int64_t ii=0; ii<(int64_t)m_ndim; ii++) 
		m_order[ii] = m_ndim-ii-1;
}

/**
 * @brief Sets the order of iteration from ++/-- operators. Invalidates
 * position
 *
 * @param order	vector of priorities, with first element being the fastest
 * iteration and last the slowest. All other dimensions not used will be
 * slower than the last
 * @param revorder	Reverse order, in which case the first element of order
 * 					will have the slowest iteration, and dimensions not
 * 					specified in order will be faster than those included.
 */
void Slicer::setOrder(const std::vector<size_t>& order, bool revorder)
{
	m_order.clear();

	// need to ensure that all dimensions get covered
	std::list<size_t> avail;
	for(size_t ii=0 ; ii<m_ndim ; ii++) {
		if(revorder)
			avail.push_front(ii);
		else
			avail.push_back(ii);
	}

	// add dimensions to internal order, but make sure there are
	// no repeats
	for(auto ito=order.begin(); ito != order.end(); ito++) {

		// determine whether the given is available still
		auto it = std::find(avail.begin(), avail.end(), *ito);

		if(it != avail.end()) {
			m_order.push_back(*it);
			avail.erase(it);
		}
	}

	// we would like the dimensions to be added so that steps are small,
	// so in revorder case, add dimensions in increasing order (since they will
	// be flipped), in normal case add in increasing order.
	// so dimensions 0 3, 5 might be remaining, with order currently:
	// m_order = {1,4,2},
	// in the case of revorder we will add the remaining dimensions as
	// m_order = {1,4,2,0,3,5}, because once we flip it will be {5,3,0,2,4,1}
	if(revorder) {
		for(auto it=avail.begin(); it != avail.end(); ++it)
			m_order.push_back(*it);
		// reverse 6D, {0,5},{1,4},{2,3}
		// reverse 5D, {0,4},{1,3}
		for(size_t ii=0; ii<m_ndim/2; ii++)
			std::swap(m_order[ii],m_order[m_ndim-1-ii]);
	} else {
		for(auto it=avail.rbegin(); it != avail.rend(); ++it)
			m_order.push_back(*it);
	}
};

/**
 * @brief Sets the order of iteration from ++/-- operators. Invalidates
 * position
 *
 * @param order	vector of priorities, with first element being the fastest
 * iteration and last the slowest. All other dimensions not used will be
 * slower than the last
 * @param revorder	Reverse order, in which case the first element of order
 * 					will have the slowest iteration, and dimensions not
 * 					specified in order will be faster than those included.
 */
void Slicer::setOrder(std::initializer_list<size_t> order, bool revorder)
{
	m_order.clear();

	// need to ensure that all dimensions get covered
	std::list<size_t> avail;
	for(size_t ii=0 ; ii<m_ndim ; ii++) {
		if(revorder)
			avail.push_front(ii);
		else
			avail.push_back(ii);
	}

	// add dimensions to internal order, but make sure there are
	// no repeats
	for(auto ito=order.begin(); ito != order.end(); ito++) {

		// determine whether the given is available still
		auto it = std::find(avail.begin(), avail.end(), *ito);

		if(it != avail.end()) {
			m_order.push_back(*it);
			avail.erase(it);
		}
	}

	// we would like the dimensions to be added so that steps are small,
	// so in revorder case, add dimensions in increasing order (since they will
	// be flipped), in normal case add in increasing order.
	// so dimensions 0 3, 5 might be remaining, with order currently:
	// m_order = {1,4,2},
	// in the case of revorder we will add the remaining dimensions as
	// m_order = {1,4,2,0,3,5}, because once we flip it will be {5,3,0,2,4,1}
	if(revorder) {
		for(auto it=avail.begin(); it != avail.end(); ++it)
			m_order.push_back(*it);
		// reverse 6D, {0,5},{1,4},{2,3}
		// reverse 5D, {0,4},{1,3}
		for(size_t ii=0; ii<m_ndim/2; ii++)
			std::swap(m_order[ii],m_order[m_ndim-1-ii]);
	} else {
		for(auto it=avail.rbegin(); it != avail.rend(); ++it)
			m_order.push_back(*it);
	}
};

/**
 * @brief Jump to the given position, additional values in newpos beyond dim
 * will be ignored. Any values missing due to ndim > len will be treated as
 * zeros.
 *
 * @param newpos	location to move to
 */
void Slicer::goIndex(size_t len, int64_t* newpos)
{
	m_linpos = 0;
	size_t ii=0;

	// copy the dimensions
	for(ii = 0;  ii<m_pos.size(); ii++) {
		assert(newpos[ii] >= m_roi[ii].first && newpos[ii] <= m_roi[ii].second);

		// set position
		if(ii < len) 
			m_pos[ii] = newpos[ii];
		else 
			m_pos[ii] = 0;
		m_linpos += m_strides[ii]*m_pos[ii];
	}

	m_end = false;
};

/**
 * @brief Jump to the given position
 *
 * @param newpos	location to move to
 */
void Slicer::goIndex(std::vector<int64_t> newpos)
{
	m_linpos = 0;

	// copy the dimensions
	for(size_t ii=0; ii < m_ndim; ii++) {
		assert(newpos[ii] >= m_roi[ii].first && newpos[ii] <= m_roi[ii].second);

		// set position
		if(ii < newpos.size()) 
			m_pos[ii] = newpos[ii];
		else
			m_pos[ii] = 0;
		m_linpos += m_strides[ii]*m_pos[ii];
	}

	m_end = false;
};
	
	
/**
 * @brief Are we at the begining of iteration?
 *
 * @return true if we are at the begining
 */
void Slicer::goBegin()
{
	for(size_t ii=0; ii<m_ndim ; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
	m_end = false;
}
	
void Slicer::goEnd()
{
	goBegin();
	m_end = true;
}

/**
 * @brief Default Constructor, max a length 1, dimension 1 slicer
 */
ChunkSlicer::ChunkSlicer() 
{
	size_t i=1;
	setDim(1, &i);
}

/**
 * @brief Constructor, takses the number of dimensions and the size of the
 * image.
 *
 * @param ndim	size of ND array
 * @param dim	array providing the size in each dimension
 */
ChunkSlicer::ChunkSlicer(size_t ndim, const size_t* dim) 
{
	setDim(ndim, dim);
}

/**
 * @brief Sets the dimensionality of iteration, and the dimensions of the image
 *
 * Position, ROI, Order, ChunkSize will all be set to their defaults.
 *
 * @param ndim Number of dimension
 * @param dim Dimensions (size)
 */
void ChunkSlicer::setDim(size_t ndim, const size_t* dim)
{
	m_linpos = 0;
	m_linfirst = 0;
	m_pos.resize(ndim);
	std::fill(m_pos.begin(), m_pos.end(), 0);
	
	m_end = false;
	m_ndim = ndim;
	
	m_dim.assign(dim, dim+ndim);

	// reset order
	m_order.resize(ndim);
	for(int64_t ii=0; ii<ndim; ii++)
		m_order[ii] = ((int64_t)ndim)-1-ii;

	// set ROI to max
	m_roi.resize(ndim);
	for(size_t ii = 0; ii<ndim; ii++) {
		m_roi[ii].first = 0; 
		m_roi[ii].second = dim[ii]-1;
	}

	// set up strides
	m_strides.resize(ndim);
	m_strides[ndim-1] = 1;
	for(int64_t ii=((int64_t)ndim)-2; ii>=0; ii--) {
		m_strides[ii] = m_strides[ii+1]*dim[ii+1];
	}
	
	m_chunkfirst = m_linfirst;
	m_chunksizes.resize(m_ndim);
	std::fill(m_chunksizes.begin(), m_chunksizes.end(), 0);
	
	m_chunk.resize(m_ndim);
	for(size_t ii=0; ii<ndim; ii++) {
		m_chunk[ii].first = m_roi[ii].first;
		m_chunk[ii].second = m_chunksizes[ii]-1 + m_chunk[ii].first;
		if(m_chunk[ii].second > m_roi[ii].second)
			m_chunk[ii].second = m_roi[ii].second;
	}
	
}

/*************************************
 * Movement
 ***********************************/

/**
 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
 * order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
ChunkSlicer& ChunkSlicer::operator++()
{
	if(isChunkEnd()) 
		return *this;
	
	m_chunkend = true;
	for(size_t ii=0; ii < m_ndim; ii++){
		size_t dd = m_order[ii];
		if(m_pos[dd] < m_chunk[dd].second) {
			m_pos[dd]++;
			m_linpos += m_strides[dd];
			m_chunkend = false;
			break;
		} else {
			// rool over 
			m_linpos -= (m_pos[dd]-m_chunk[dd].first)*m_strides[dd];
			m_pos[dd] = m_chunk[dd].first;
		}
	}

	return *this;
};

/**
 * @brief Prefix negative  iterator. Iterates in the order dictatored by
 * the dimension order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
ChunkSlicer& ChunkSlicer::operator--()
{
	if(isChunkBegin()) 
		return *this;

	m_chunkend = false;
	for(size_t ii=0; ii < m_ndim; ii++){
		size_t dd = m_order[ii];
		if(m_pos[dd] > m_chunk[dd].first) {
			m_pos[dd]--;
			m_linpos -= m_strides[dd];
			break;
		} else {
			// rool over 
			m_linpos += (m_chunk[dd].second-m_pos[dd])*m_strides[dd];
			m_pos[dd] = m_chunk[dd].second;
		}
	}
	
	return *this;
};

/**
 * @brief Proceed to the next chunk (if there is one).
 *
 * @return 	
 */
ChunkSlicer& ChunkSlicer::nextChunk()
{
	if(isEnd()) 
		return *this;
	
	// try to move in whatever dimension can be stepped
	m_end = true;
	m_chunkend = false;
	for(size_t ii=0; ii < m_ndim; ii++){
		size_t dd = m_order[ii];
		if(m_chunk[dd].second < m_roi[dd].second) {
			// increment beginning chunk 
			m_chunk[dd].first = m_chunk[dd].second+1;

			// update second
			if(m_chunksizes[dd]-1 + m_chunk[dd].first >= m_roi[dd].second || 
						m_chunksizes[dd] == 0) 
				m_chunk[dd].second = m_roi[dd].second;
			else 
				m_chunk[dd].second = m_chunksizes[dd]-1 + m_chunk[dd].first;
				
			m_end = false;
			break;
		} else {
			// rool over beginning of chunk
			m_chunk[dd].first = m_roi[dd].first;
			
			// update second
			if(m_chunksizes[dd]-1 + m_chunk[dd].first >= m_roi[dd].second || 
						m_chunksizes[dd] == 0) 
				m_chunk[dd].second = m_roi[dd].second;
			else 
				m_chunk[dd].second = m_chunksizes[dd]-1 + m_chunk[dd].first;
		}
	}

	// reset position
	m_linpos = 0;
	m_chunkfirst = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		m_pos[ii] = m_chunk[ii].first;
		m_chunkfirst += m_pos[ii]*m_strides[ii];
		m_linpos += m_pos[ii]*m_strides[ii];
	}

	return *this;
}

/**
 * @brief Return to the previous chunk (if there is one).
 *
 * @return 	new value of linear position
 */
ChunkSlicer& ChunkSlicer::prevChunk()
{
	if(isBegin()) 
		return *this;
	m_end = false;
	m_chunkend = false;

	// try to move in whatever dimension can be stepped
	for(size_t ii=0; ii < m_ndim; ii++){
		size_t dd = m_order[ii];
		if(m_chunk[dd].first > m_roi[dd].first) {
			// there is a gap between beginning of ROI and beginning of the
			// current chunk, so we will move into that gap

			// decrement beginning of chunk 
			m_chunk[dd].second = m_chunk[dd].first-1;
			
			// if this is not the case, then we have broken up a chunk at the
			// front, while we really want to break up the ones at the back
			assert(m_chunk[dd].second - (m_chunksizes[dd]-1) >= m_roi[dd].first);

			// update first 
			if(m_chunk[dd].second - (m_chunksizes[dd]-1) > m_roi[dd].first) {
				m_chunk[dd].first = m_chunk[dd].second - (m_chunksizes[dd]-1);
			} else if(m_chunk[dd].second - (m_chunksizes[dd]-1) == m_roi[dd].first) {
				m_chunk[dd].first = m_roi[dd].first;
			} else {
				throw std::logic_error("Failed chunk stepping call the programmer");
			}

			m_chunkend = false;
			break;
		} else {
			// roll over beginning of chunk, need to be careful that we don't
			// have slack at the front of iteration, so we decrease the size 
			// at the end when the image is not divisable by the chunk size
			m_chunk[dd].second = m_roi[dd].second;
			
			// update second
			if(m_chunksizes[dd] == 0 || m_chunksizes[dd] > (m_roi[dd].second - m_roi[dd].first+1)) {
				// just use the entire region
				m_chunk[dd].first = m_roi[dd].first;
			} else if(m_chunk[dd].second - (m_chunksizes[dd]-1)> m_roi[dd].first) {
				// calculate reduced size, if things are going to be broken
				// up due to modulus != 0, go ahead and reduce the size during
				// the roll over
				size_t rsize = (m_roi[dd].second - m_roi[dd].first + 1)%m_chunksizes[dd];
				if(rsize == 0)
					m_chunk[dd].first = m_chunk[dd].second - (m_chunksizes[dd]-1);
				else
					m_chunk[dd].first = m_chunk[dd].second - rsize + 1;

			} else if(m_chunk[dd].second - (m_chunksizes[dd]-1) == m_roi[dd].first) {
				m_chunk[dd].first = m_roi[dd].first;
			} else {
				throw std::logic_error("Failed chunk stepping call the programmer");
			}
		}
	}

	// reset position
	m_linpos = 0;
	m_chunkfirst = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		m_pos[ii] = m_chunk[ii].first;
		m_chunkfirst += m_pos[ii]*m_strides[ii];
		m_linpos += m_pos[ii]*m_strides[ii];
	}
	
	return *this;
};

/**
 * @brief Go to the end of the current chunk, if you are at the 
 * end of iteration, then it does nothing. You need to prevChunk() first
 *
 */
/**
 * @brief Go to the beginning for the current chunk, if you are at the 
 * end of iteration, then it does nothing. You need to prevChunk() first
 *
 */
void ChunkSlicer::goChunkBegin()
{
	if(isEnd())
		return;

	m_chunkend = false;
	m_end = false;
	m_linpos = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		m_pos[ii] = m_chunk[ii].first;
		m_linpos += m_strides[ii]*m_pos[ii];
	}
}

/**
 * @brief Jump to the end of current chunk.
 *
 * End of each chunk is represented as a wrapped around beginning, but with 
 * m_chunkend set. 
 *
 */
void ChunkSlicer::goChunkEnd()
{
	if(isEnd())
		return;

	goChunkBegin();
	m_chunkend = true;
};

/**
 * @brief Go to the very beginning for the first chunk.
 *
 * Depends on m_roi
 *
 * Set m_end, m_chunkend, m_linpos, m_linfirst, m_chunkfirst, m_pos, m_chunk
 *
 */
void ChunkSlicer::goBegin()
{
	m_end = false;
	m_chunkend = false;
	
	m_linpos = m_linfirst;
	for(size_t ii=0; ii<m_ndim; ii++) {
		m_pos[ii] = m_roi[ii].first;
		m_chunk[ii].first = m_roi[ii].first;
		if(m_chunksizes[ii] == 0 || m_chunksizes[ii] > 
					(m_roi[ii].second-m_roi[ii].first+1)) 
			m_chunk[ii].second = m_roi[ii].second; 
		else 
			m_chunk[ii].second = m_chunk[ii].first + m_chunksizes[ii]-1;
	}
}

/**
 * @brief Jump to the end of the last chunk.
 *
 */
void ChunkSlicer::goEnd()
{
	goBegin();
	m_end = true;
	m_chunkend = true;
};

/**
 * @brief Jump to the given position, additional values in newpos beyond dim
 * will be ignored. Any values missing due to ndim > len will be treated as
 * zeros.
 *
 * @param newpos	location to move to
 */
void ChunkSlicer::goIndex(size_t len, int64_t* newpos)
{
	m_linpos = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		if(ii < len) 
			m_pos[ii] = newpos[ii];
		else 
			m_pos[ii] = 0;
		m_linpos += m_strides[ii]*m_pos[ii];

		// if chunksizes is 0, or exceeds the ROI size, just set to the max
		if(m_chunksizes[ii] == 0 || m_chunksizes[ii] >= 
						(m_roi[ii].second - m_roi[ii].first + 1)) {
			m_chunk[ii].first = m_roi[ii].first;
			m_chunk[ii].second = m_roi[ii].second;
		} else {
			// otherwise determine with chunk we are in (int division) multiply
			int64_t tmp1 = m_pos[ii]/m_chunksizes[ii];
			m_chunk[ii].first = tmp1*m_chunksizes[ii]+m_roi[ii].first;

			// then set the second part m_chunksizes away
			if(m_chunk[ii].first + m_chunksizes[ii]-1 >= m_roi[ii].second)
				m_chunk[ii].second = m_roi[ii].second;
			else
				m_chunk[ii].second = m_chunk[ii].first + m_chunksizes[ii]-1;
		}
	}
};

/**
 * @brief Jump to the given position
 *
 * @param newpos	location to move to
 */
void ChunkSlicer::goIndex(std::vector<int64_t> newpos)
{
	m_linpos = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		if(ii < newpos.size()) 
			m_pos[ii] = newpos[ii];
		else 
			m_pos[ii] = 0;
		m_linpos += m_strides[ii]*m_pos[ii];

		// if chunksizes is 0, or exceeds the ROI size, just set to the max
		if(m_chunksizes[ii] == 0 || m_chunksizes[ii] >= 
						(m_roi[ii].second - m_roi[ii].first + 1)) {
			m_chunk[ii].first = m_roi[ii].first;
			m_chunk[ii].second = m_roi[ii].second;
		} else {
			// otherwise determine with chunk we are in (int division) multiply
			int64_t tmp1 = m_pos[ii]/m_chunksizes[ii];
			m_chunk[ii].first = tmp1*m_chunksizes[ii]+m_roi[ii].first;

			// then set the second part m_chunksizes away
			if(m_chunk[ii].first + m_chunksizes[ii]-1 >= m_roi[ii].second)
				m_chunk[ii].second = m_roi[ii].second;
			else
				m_chunk[ii].second = m_chunk[ii].first + m_chunksizes[ii]-1;
		}
	}
}

/****************************************
 *
 * Actually get the linear location
 *
 ***************************************/

/**
 * @brief Places the first len dimension in the given array. If the number
 * of dimensions exceed the len then the additional dimensions will be
 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
 * In other words index will be completely overwritten in the most sane way
 * possible if the internal dimensions and size index differ.
 *
 * @param len size of index
 * @param index output index variable
 */
void ChunkSlicer::index(size_t len, int64_t* index) const
{
	for(size_t ii=0; ii<len && ii<m_ndim; ii++) {
		index[ii] = m_pos[ii];
	}
}
/**
 * @brief Places the first len dimension in the given array. If the number
 * of dimensions exceed the len then the additional dimensions will be
 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
 * In other words index will be completely overwritten in the most sane way
 * possible if the internal dimensions and size index differ.
 *
 * @param len size of index
 * @param index output index variable
 */
void ChunkSlicer::index(size_t len, int* index) const
{
	for(size_t ii=0; ii<len && ii<m_ndim; ii++) {
		index[ii] = m_pos[ii];
	}
}

/**
 * @brief Places the first len dimension in the given array. If the number
 * of dimensions exceed the len then the additional dimensions will be
 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
 * In other words index will be completely overwritten in the most sane way
 * possible if the internal dimensions and size index differ.
 *
 * @param len size of index
 * @param index output index variable
 */
void ChunkSlicer::index(size_t len, double* index) const
{
	for(size_t ii=0; ii<len && ii<m_ndim; ii++) {
		index[ii] = m_pos[ii];
	}
}



/***********************************************
 *
 * Modification
 *
 **********************************************/

/**
 * @brief Sets the region of interest, with lower bound of 0.
 * During iteration or any motion the
 * position will not move outside the specified range. Invalidates position.
 *
 * Invalidates position
 *
 * @param len Length of roi
 * @param roisize Size of ROI (which runs from [0,0...] to [roi[0], roi[1]...]
 */
void ChunkSlicer::setROI(size_t len, const size_t* roi)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim ; ii++) {
        m_roi[ii].first = 0;
		if(ii < len) {
			// clamp, to be <= 0...sizes[ii]-1
			m_roi[ii].second = clamp<int64_t>(0, m_dim[ii]-1, roi[ii]);
		} else {
			// no specification, just make it all
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_strides[ii]*m_roi[ii].first;
	}

	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
	m_chunkfirst = m_linfirst;
}

/**
 * @brief Sets the region of interest, with lower bound of 0.
 * During iteration or any motion the
 * position will not move outside the specified range. Invalidates position.
 *
 * Invalidates position
 *
 * @param len Length of roi
 * @param roisize Size of ROI (which runs from [0,0...] to [roi[0], roi[1]...]
 */
void ChunkSlicer::setROI(size_t len, const int64_t* roi)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim ; ii++) {
        m_roi[ii].first = 0;
		if(ii < len) {
			// clamp, to be <= 0...sizes[ii]-1
			m_roi[ii].second = clamp<int64_t>(0, m_dim[ii]-1, roi[ii]);
		} else {
			// no specification, just make it all
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_strides[ii]*m_roi[ii].first;
	}

	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
	m_chunkfirst = m_linfirst;
}


/**
 * @brief Sets the region of interest. During iteration or any motion the
 * position will not move outside the specified range. Extra elements in
 * roi beyond the number of dimensions, are ignored.
 *
 * Invalidates position
 *
 * @param roi	pair of [min,max] values in the desired hypercube
 */
void ChunkSlicer::setROI(const std::vector<std::pair<int64_t, int64_t>>& roi)
{
	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		if(ii < roi.size()) {
			m_roi[ii].first = roi[ii].first;
			m_roi[ii].second = roi[ii].second;
		} else {
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_roi[ii].first*m_strides[ii];
	}
	
	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
	m_chunkfirst = m_linfirst;
}

/**
 * @brief Sets the region of interest. During iteration or motion the
 * position will not move outside the specified range
 *
 * Invalidates position
 *
 * @param len	Length of both lower and upper arrays.
 * @param lower	Coordinate at lower bound of bounding box.
 * @param upper	Coordinate at upper bound of bounding box.
 */
void ChunkSlicer::setROI(size_t len, const int64_t* lower, const int64_t* upper)
{
	if(lower == NULL || upper == NULL)
		len = 0;

	m_linfirst = 0;
	for(size_t ii=0; ii<m_ndim; ii++) {
		if(ii < len) {
			m_roi[ii].first = lower[ii];
			m_roi[ii].second = upper[ii];
		} else {
			m_roi[ii].first = 0;
			m_roi[ii].second = m_dim[ii]-1;
		}
		m_linfirst += m_roi[ii].first*m_strides[ii];
	}
	
	for(size_t ii=0; ii<m_ndim; ii++)
		m_pos[ii] = m_roi[ii].first;
	m_linpos = m_linfirst;
	m_chunkfirst = m_linfirst;
}

/**
 * @brief Set the sizes of chunks for each dimension. Chunks will end every
 * N steps in each of the provided dimension, with the caveout that 0
 * indicates no breaks in the given dimension. So size = {0, 2, 2} will
 * cause chunks to after {XLEN-1, 1, 1}. {0,0,0} (the default) indicate
 * that the entire image will be iterated and only one chunk will be used. 
 *
 * @param len Length of sizes array
 * @param sizes Size of chunk in each dimension. If you multiply together
 * the elements of sizes that is the MAXIMUM number of iterations between 
 * chunks. Note however that there could be less if we are at the edge.
 * @param defunity Sets the default to unity rather than 0. Thus 
 * unreferenced dimensions will be broken up at each step; so {0,1} for a
 * 4D image will be effectively {0,1,1,1} instead of {0,1,0,0}. This is
 * convenient, for instance, if you want to split up based on volumes,
 * {0,0,0} would stop at the end of each volume, whereas the default would
 * be to treat the entire ND-image as a chunk.
 */
void ChunkSlicer::setChunkSize(size_t len, const int64_t* sizes, bool defunity)
{
	assert(m_roi.size() == m_ndim);

	// missing values are assumed to be full (0)
	for(size_t ii=0; ii<m_ndim; ii++) {
		if(ii<len && sizes[ii] > 0)
			m_chunksizes[ii] = sizes[ii];
		else 
			m_chunksizes[ii] = (int)defunity;
	}
}

/**
 * @brief Sets the chunk sizes so that each chunk is a line in the given 
 * dimension. This would be analogous to itk's linear iterator. 
 * Note, you should call goBegin() or nextChunk() after this otherwise the
 * first chunk may be unitialized.
 * Usage:
 *
 * it.setLineChunk(0);
 * it.goBegin();
 * while(!it.isEnd()) {
 *	while(!it.isChunkEnd()) {
 *	
 *		++it;
 *	}
 *	it.nextChunk();
 * }
 *
 * @param dir Dimension to travel linearly along
 */
void ChunkSlicer::setLineChunk(size_t dir)
{
	assert(m_roi.size() == m_ndim);

	for(size_t ii=0; ii<m_ndim; ii++) {
		m_chunksizes[ii] = 1;
	}
	m_chunksizes[dir] = 0;
}

/**
 * @brief Sets the order of iteration from ++/-- operators
 *
 * Invalidates position
 *
 * @param order	vector of priorities, with first element being the fastest
 * iteration and last the slowest. All other dimensions not used will be
 * slower than the last
 * @param revorder	Reverse order, in which case the first element of order
 * 					will have the slowest iteration, and dimensions not
 * 					specified in order will be faster than those included.
 */
void ChunkSlicer::setOrder(std::initializer_list<size_t> order, bool revorder)
{
	m_order.clear();

	// need to ensure that all dimensions get covered
	std::list<size_t> avail;
	for(size_t ii=0 ; ii<m_ndim ; ii++) {
		if(revorder)
			avail.push_front(ii);
		else
			avail.push_back(ii);
	}

	// add dimensions to internal order, but make sure there are
	// no repeats
	for(auto ito=order.begin(); ito != order.end(); ito++) {

		// determine whether the given is available still
		auto it = std::find(avail.begin(), avail.end(), *ito);

		if(it != avail.end()) {
			m_order.push_back(*it);
			avail.erase(it);
		}
	}

	// we would like the dimensions to be added so that steps are small,
	// so in revorder case, add dimensions in increasing order (since they will
	// be flipped), in normal case add in increasing order.
	// so dimensions 0 3, 5 might be remaining, with order currently:
	// m_order = {1,4,2},
	// in the case of revorder we will add the remaining dimensions as
	// m_order = {1,4,2,0,3,5}, because once we flip it will be {5,3,0,2,4,1}
	if(revorder) {
		for(auto it=avail.begin(); it != avail.end(); ++it)
			m_order.push_back(*it);
		// reverse 6D, {0,5},{1,4},{2,3}
		// reverse 5D, {0,4},{1,3}
		for(size_t ii=0; ii<m_ndim/2; ii++)
			std::swap(m_order[ii],m_order[m_ndim-1-ii]);
	} else {
		for(auto it=avail.rbegin(); it != avail.rend(); ++it)
			m_order.push_back(*it);
	}
}

/**
 * @brief Sets the order of iteration from ++/-- operators
 *
 * Invalidates position
 *
 * @param order	vector of priorities, with first element being the fastest
 * iteration and last the slowest. All other dimensions not used will be
 * slower than the last
 * @param revorder	Reverse order, in which case the first element of order
 * 					will have the slowest iteration, and dimensions not
 * 					specified in order will be faster than those included.
 */
void ChunkSlicer::setOrder(const std::vector<size_t>& order, bool revorder)
{
	m_order.clear();

	// need to ensure that all dimensions get covered
	std::list<size_t> avail;
	for(size_t ii=0 ; ii<m_ndim ; ii++) {
		if(revorder)
			avail.push_front(ii);
		else
			avail.push_back(ii);
	}

	// add dimensions to internal order, but make sure there are
	// no repeats
	for(auto ito=order.begin(); ito != order.end(); ito++) {

		// determine whether the given is available still
		auto it = std::find(avail.begin(), avail.end(), *ito);

		if(it != avail.end()) {
			m_order.push_back(*it);
			avail.erase(it);
		}
	}

	// we would like the dimensions to be added so that steps are small,
	// so in revorder case, add dimensions in increasing order (since they will
	// be flipped), in normal case add in increasing order.
	// so dimensions 0 3, 5 might be remaining, with order currently:
	// m_order = {1,4,2},
	// in the case of revorder we will add the remaining dimensions as
	// m_order = {1,4,2,0,3,5}, because once we flip it will be {5,3,0,2,4,1}
	if(revorder) {
		for(auto it=avail.begin(); it != avail.end(); ++it)
			m_order.push_back(*it);
		// reverse 6D, {0,5},{1,4},{2,3}
		// reverse 5D, {0,4},{1,3}
		for(size_t ii=0; ii<m_ndim/2; ii++)
			std::swap(m_order[ii],m_order[m_ndim-1-ii]);
	} else {
		for(auto it=avail.rbegin(); it != avail.rend(); ++it)
			m_order.push_back(*it);
	}
}

} //npl
