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
 * @file kernel_slicer.cpp
 *
 *****************************************************************************/

#include "kernel_slicer.h"

#include <vector>
#include <list>
#include <algorithm>
#include <cassert>

namespace npl {

int64_t clamp(int64_t inf, int64_t sup, int64_t val)
{
	return std::min(sup, std::max(inf, val));
}

/**
 * @brief Default Constructor, max a length 1, dimension 1 slicer
 */
KSlicer::KSlicer()
{
	size_t tmp = 1;
	initialize(1, &tmp);
};

/**
 * @brief Constructs a iterator with the given dimensions
 *
 * @param ndim	Number of dimensions
 * @param dim	size of ND array
 */
KSlicer::KSlicer(size_t ndim, const size_t* dim)
{
	initialize(ndim, dim);
}
	
/**
 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
 * order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
KSlicer& KSlicer::operator++()
{
	if(isEnd())
		return *this;
	
	int64_t forbound = (int64_t)m_pos[m_center][m_order[0]]+m_fradius;
	int64_t revbound = (int64_t)m_pos[m_center][m_order[0]]-m_rradius;
	
	// if the entire kernel is within the line, then just add add 1/stride
	if(forbound < (int64_t)m_roi[m_order[0]].second &&
				revbound >= (int64_t)m_roi[m_order[0]].first) {
		for(size_t oo=0; oo<m_numoffs; oo++) {
			m_pos[oo][m_order[0]]++;
			m_linpos[oo] += m_strides[m_order[0]];
		}
	} else { // brute force
	
		// iterate center
		for(size_t ii=0; ii<m_dim; ii++){
			size_t dd = m_order[ii];
			if(m_pos[m_center][dd] < m_roi[dd].second) {
				m_pos[m_center][dd]++;
				break;
			} else if(ii != m_dim-1){
				// reset dimension
				m_pos[m_center][dd] = m_roi[dd].first;
			} else {
				// we are willing to go 1 past the last
				m_pos[m_center][dd]++;
				m_linpos[m_center] += m_strides[dd];
				m_end = true;

				// want to skip clamping, and not really a need to update
				// neighborhood when we are outside the image
				return *this;
			}
		}

		// calculate ND positions for each other offset
		for(size_t oo=0; oo<m_numoffs; oo++) {
			for(size_t dd=0; dd<m_dim; dd++) {
				m_pos[oo][dd] = clamp(m_roi[dd].first, m_roi[dd].second,
						m_pos[m_center][dd]+m_offs[oo][dd]);
			}
		}

		// calculate linear positions from each
		for(size_t oo = 0; oo < m_numoffs; oo++) {
			m_linpos[oo] = 0;
			for(size_t dd=0; dd<m_dim; dd++)
				m_linpos[oo] += m_pos[oo][dd]*m_strides[dd];
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
KSlicer& KSlicer::operator--()
{
	if(isBegin())
		return *this;
	
	m_end = false;

	int64_t forbound = (int64_t)m_pos[m_center][m_order[0]]+m_fradius;
	int64_t revbound = (int64_t)m_pos[m_center][m_order[0]]-m_rradius;
	
	// if the entire kernel is within the line, then just add add 1/stride
	if(forbound <= (int64_t)m_roi[m_order[0]].second &&
				revbound > (int64_t)m_roi[m_order[0]].first) {
		for(size_t oo=0; oo<m_numoffs; oo++) {
			m_pos[oo][m_order[0]]--;
			m_linpos[oo] -= m_strides[m_order[0]];
		}
	} else { // brute force
	
		// iterate center
		for(size_t ii=0; ii<m_order.size(); ii++){
			size_t dd = m_order[ii];
			if(m_pos[m_center][dd] != m_roi[dd].first) {
				m_pos[m_center][dd]--;
				break;
			} else if(ii != m_order.size()-1) {
				// jump forward in dd, (will pull back in next)
				m_pos[m_center][dd] = m_roi[dd].second;
			}
		}

		// calculate ND positions for each other offset
		for(size_t oo=0; oo<m_numoffs; oo++) {
			for(size_t dd=0; dd<m_dim; dd++) {
				m_pos[oo][dd] = clamp(m_roi[dd].first, m_roi[dd].second,
						m_pos[m_center][dd]+m_offs[oo][dd]);
			}
		}

		// calculate linear positions from each
		for(size_t oo = 0; oo < m_numoffs; oo++) {
			m_linpos[oo] = 0;
			for(size_t dd=0; dd<m_dim; dd++)
				m_linpos[oo] += m_pos[oo][dd]*m_strides[dd];
		}
	}

	return *this;
}
	
/**
 * @brief Set the radius of the kernel window. All directions will
 * have equal distance, with the radius in each dimension set by the
 * magntitude of the kradius vector. So if kradius = {2,1,0} then
 * dimension 0 (x) will have a radius of 2, dimension 1 (y) will have
 * a readius of 1 and dimension 2 will have a radius of 0 (won't step
 * out from the middle at all).
 *
 * You should call goBegin() after this
 *
 * @param kradius vector of radii in the given dimension. Unset values
 * assumed to be 0. So a 10 dimensional image with 3 values will have
 * non-zero values for x,y,z but 0 values in higher dimensions
 */
void KSlicer::setRadius(std::vector<size_t> kradius)
{
	std::vector<std::pair<int64_t, int64_t>> tmp(m_dim);
	for(size_t ii=0; ii<m_dim; ii++) {
		if(ii < kradius.size()) {
			tmp[ii].first = -clamp(0, m_size[ii]-1, kradius[ii]);
			tmp[ii].second = clamp(0, m_size[ii]-1, kradius[ii]);
		} else {
			tmp[ii].first = 0;
			tmp[ii].second = 0;
		}
	}

	setWindow(tmp);
}

/**
 * @brief Set the radius of the kernel window. All directions will
 * have equal distance in all dimensions. So if kradius = 2 then
 * dimension 0 (x) will have a radius of 2, dimension 2 (y) will have
 * a readius of 2 and so on. Warning images may have more dimensions
 * than you know, so if the image has a dimension that is only size 1
 * it will have a radius of 0, but if you didn't know you had a 10D image
 * all the dimensions about to support the radius will.
 *
 * You should call goBegin() after this
 *
 * @param radii in all directions.
 */
void KSlicer::setRadius(size_t kradius)
{
	std::vector<std::pair<int64_t, int64_t>> tmp(m_dim);
	for(size_t ii=0; ii<m_dim; ii++) {
		tmp[ii].first = -clamp(0, m_size[ii]-1, kradius);
		tmp[ii].second = clamp(0, m_size[ii]-1, kradius);
	}

	setWindow(tmp);
}

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
 * You should call goBegin() after this
 *
 * @param Vector of [inf, sup] in each dimension. Unaddressed (missing)
 * values are assumed to be [0,0].
 */
void KSlicer::setWindow(const std::vector<std::pair<int64_t, int64_t>>& krange)
{
	std::vector<int64_t> kmin(m_dim, 0);
	std::vector<int64_t> kmax(m_dim, 0);
	for(size_t dd=0; dd<krange.size(); dd++) {
		kmin[dd] = krange[dd].first;
		kmax[dd] = krange[dd].second;
		if(kmin[dd] > 0 || kmax[dd] < 0) {
			throw std::logic_error("Kernel window in KSlicer does "
					"not include the center!");
		}
	}

	// we need to know how far forward and back the kernel stretches because
	// when the kernel gets near an edge, we have to recompute so the kernel
	// doesn't go utside the image
	m_fradius = kmax[m_order[0]];
	m_rradius = -kmin[m_order[0]];

	// for each point, we need this
	m_numoffs = 1;
	for(size_t ii=0; ii<m_dim; ii++) {
		m_numoffs *= kmax[ii]-kmin[ii]+1;
	}

	m_offs.resize(m_numoffs);
	fill(m_offs.begin(), m_offs.end(), std::vector<int64_t>(m_dim));

	// initialize first offset then set remaining based on that
	for(size_t ii=0; ii<m_dim; ii++)
		m_offs[0][ii] = kmin[ii];

	m_center = 0;
	for(size_t oo=1; oo<m_offs.size(); oo++) {
		int64_t dd=m_dim-1;
		// copy from previous
		for(dd=0; dd<m_dim; dd++)
			m_offs[oo][dd] = m_offs[oo-1][dd];

		// advance 1
		for(dd=m_dim-1; dd>=0; dd--) {

			// if we can increase in bounds, just do that
			if(m_offs[oo-1][dd] < kmax[dd]) {
				m_offs[oo][dd]++;
				break;
			} else {
				// roll over if we hit the edge
				m_offs[oo][dd] = kmin[dd];
			}
		}

		// figure out if this is the center
		bool center = true;
		for(dd=0; dd<m_dim; dd++) {
			if(m_offs[oo][dd] != 0) {
				center = false;
				break;
			}
		}

		if(center)
			m_center = oo;
	}

	m_pos.resize(m_offs.size());
	for(size_t ii=0; ii<m_pos.size(); ii++)
		m_pos[ii].resize(m_dim,0);
	m_linpos.resize(m_offs.size());
}
	

/**
 * @brief Sets the region of interest. During iteration or any motion the
 * position will not move outside the specified range. Note that behavior
 * is not defined after you do this, until you call goBegin()
 *
 * You should call goBegin() after this
 *
 * @param roi Range of region of interest. Pairs indicates the range
 * 	in i'th dimension, so krange = {{1,5},{0,9},{32,100}}
 * 	would cause the iterator to range from (1,0,32) to (5,9,100)
 */
void KSlicer::setROI(std::vector<std::pair<int64_t, int64_t>> roi)
{
	// set up ROI, and calculate the m_begin location
	m_roi.resize(m_dim);
	m_begin = 0;
	for(size_t ii=0; ii<m_dim; ii++) {
		if(ii < roi.size()) {
			m_roi[ii].first = clamp(0, m_size[ii]-1, roi[ii].first);
			m_roi[ii].second = clamp(0, m_size[ii]-1, roi[ii].second);
		} else {
			// default to full range
			m_roi[ii].first = 0;
			m_roi[ii].second = m_size[ii]-1;
		}
		m_begin += m_roi[ii].first*m_strides[ii];
	}
}

/**
 * @brief All around intializer. Sets all internal variables.
 *
 * @param dim 		Dimension (size) of memory block.
 * @param krange 	Range of offset values. Each pair indicats a min and max
 * 					in the i'th dimension. So {{-3,0}, {-3,3},{0,3}} would
 * 					indicate a kernel from (X-3,Y-3,Z+0) to (X+0,Y+3,Z+3).
 * 					Ranges must include zero.
 */
void KSlicer::initialize(size_t ndim, const size_t* dim)
{
	m_dim = ndim;
	m_size.assign(dim, dim+ndim);

	// set up strides
	m_strides.resize(m_dim);
	m_strides[m_dim-1] = 1;
	for(int64_t ii=(int64_t)m_dim-2; ii>=0; ii--) {
		m_strides[ii] = m_strides[ii+1]*dim[ii+1];
	}

	setOrder();
	setRadius(0);
	setROI();
	goBegin();

};

/**
 * @brief Set the order of iteration, in terms of which dimensions iterate
 * the fastest and which the slowest.
 *
 * Changes m_order
 *
 * @param order order of iteration. {0,1,2} would mean that dimension 0 (x)
 * would move the fastest and 2 the slowest. If the image is a 5D image then
 * that unmentioned (3,4) would be the slowest.
 * @param revorder Reverse the speed of iteration. So the first dimension
 * in the order vector would in fact be the slowest and un-referenced
 * dimensions will be the fastest. (in the example for order this would be
 * 4 and 3).
 */
void KSlicer::setOrder(std::vector<size_t> order, bool revorder)
{
	m_order.resize(m_dim);
	size_t jj = 0;

	// need to ensure that all dimensions get covered
	std::list<size_t> avail;
	for(size_t ii=0 ; ii<m_dim ; ii++)
		avail.push_front(ii);

	// add dimensions to internal order, but make sure there are
	// no repeats
	for(size_t ii=0; ii<order.size(); ii++) {

		// determine whether the given is available still
		auto it = std::find(avail.begin(), avail.end(), order[ii]);
		if(it != avail.end()) {
			m_order[jj++] = order[ii];
			avail.erase(it);
		}
	}
	
	// to ensure we aren't constantly bouncing wrapping, we will iterate
	// in the direction of the longest dimension, rather than the fastest
	int64_t longest = 0;
	std::list<size_t>::iterator lit;
	for(auto it=avail.begin(); it != avail.end(); ++it) {
		if((int64_t)m_size[*it] >= longest) {
			longest = (int64_t)m_size[*it];
			lit = it;
		}
	}
	assert(jj < m_order.size());

	if(lit != avail.end()) {
		m_order[jj++] = *lit;
		avail.erase(lit);
	}

	// just add the remaining dimensions to order in reverse
	for(auto it=avail.begin(); it != avail.end(); ++it) {
		assert(jj < m_order.size());
		m_order[jj++] = *it;
	}
	
	if(revorder) {
		// reverse 6D, {0,5},{1,4},{2,3}
		// reverse 5D, {0,4},{1,3}
		for(size_t ii=0; ii<m_dim/2; ii++)
			std::swap(m_order[ii],m_order[m_dim-1-ii]);
	}

	// /these were invalidated, to refigure them
	m_fradius = 0;
	m_rradius = 0;
	for(size_t oo=0; oo<m_offs.size(); oo++) {
		if(m_offs[oo][m_order[0]] > m_fradius)
			m_fradius = m_offs[oo][m_order[0]];
		if(m_offs[oo][m_order[0]] < m_rradius)
			m_rradius = m_offs[oo][m_order[0]];
	}
	// turn offset into radius
	m_rradius = -m_rradius;
//	goBegin();
};

/**
 * @brief Jump to the given position
 *
 * @param newpos	location to move to
 */
void KSlicer::goIndex(const std::vector<int64_t>& newpos)
{
	if(newpos.size() != m_dim) {
		throw std::logic_error("Invalid index size in goIndex");
	}

	// don't do anything if we are already where we need to be...
	bool same = true;
	for(size_t ii=0; ii<m_dim; ii++) {
		if(newpos[ii] != m_pos[m_center][ii])
			same = false;
	}
	if(same)
		return ;

	// copy/clamp the center
	int64_t clamped;
	// copy the center
	for(size_t dd = 0; dd<m_dim; dd++) {
		// clamp to roi
		clamped = clamp(m_roi[dd].first, m_roi[dd].second, newpos[dd]);
		
		// set position
		m_pos[m_center][dd] = clamped;

	}
	
	// calculate ND positions for each other offset
	for(size_t oo=0; oo<m_numoffs; oo++) {
		for(size_t dd=0; dd<m_dim; dd++) {
			m_pos[oo][dd] = clamp(m_roi[dd].first, m_roi[dd].second,
					m_pos[m_center][dd]+m_offs[oo][dd]);
		}
	}

	// calculate linear positions from each
	for(size_t oo = 0; oo < m_numoffs; oo++) {
		m_linpos[oo] = 0;
		for(size_t dd=0; dd<m_dim; dd++)
			m_linpos[oo] += m_pos[oo][dd]*m_strides[dd];
	}
	
	m_end = false;
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
void KSlicer::center_index(size_t len, int64_t* index) const
{
	offset_index(m_center, len, index, true);
}
	
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
void KSlicer::offset_index(size_t kit, size_t len, int64_t* index, bool bound) const
{
	assert(!m_end);
	assert(kit < m_numoffs);
	assert(kit < m_numoffs);
	
	if(bound) {
		// m_pos already has been bound
		for(size_t ii=0; ii<len && ii<m_dim; ii++)
			index[ii] = m_pos[kit][ii];
	} else {
		// unbound
		for(size_t ii=0; ii < len && ii < m_dim; ii++)
			index[ii] = m_pos[m_center][ii]+m_offs[kit][ii];
	}
	
	// extra values to 0
	for(size_t ii=m_dim; ii<len; ii++) {
		index[ii] = 0;
	}
}

/**
 * @brief Returns the distance from the center projected onto the specified
 * dimension. So center is {0,0,0}, and {1,2,1} would return 1,2,1 for inputs
 * dim=0, dim=1, dim=2
 *
 * @param dim dimension to get distance in
 * @param kit Which pixel to return distance from
 *
 * @return Offset from center of given pixel (kit)
 */
int64_t KSlicer::from_center(size_t kit, size_t dim)
{
	return m_offs[kit][dim];
}

/**
 * @brief Returns offset from center of specified pixel (kit).
 *
 * @param len lenght of dindex array
 * @param dindex output paramter indicating distance of pixel from the
 * center of the kernel in each dimension. If this array is shorter than
 * the iteration dimensions, only the first len will be filled. If it is
 * longer the additional values won't be touched
 * @param kit Pixel we are referring to
 */
void KSlicer::from_center(size_t kit, size_t len, int64_t* dindex) const
{
	for(size_t ii=0; ii<len && ii<m_dim; ii++)
		dindex[ii] = m_offs[kit][ii];
};

/**
 * @brief Go to the beginning
 *
 */
void KSlicer::goBegin()
{
	// copy the center
	for(size_t dd = 0; dd<m_dim; dd++) {
		// clamp to roi
		m_pos[m_center][dd] = m_roi[dd].first;
	}
	
	// calculate ND positions for each other offset
	for(size_t oo=0; oo<m_numoffs; oo++) {
		for(size_t dd=0; dd<m_dim; dd++) {
			m_pos[oo][dd] = clamp(m_roi[dd].first, m_roi[dd].second,
					m_pos[m_center][dd]+m_offs[oo][dd]);
		}
	}

	// calculate linear positions from each
	for(size_t oo = 0; oo < m_numoffs; oo++) {
		m_linpos[oo] = 0;
		for(size_t dd=0; dd<m_dim; dd++)
			m_linpos[oo] += m_pos[oo][dd]*m_strides[dd];
	}
	
	m_end = false;
};
	
/**
 * @brief Jump to the end of iteration.
 *
 */
void KSlicer::goEnd()
{
	// copy the center
	for(size_t dd = 0; dd<m_dim; dd++) {
		// clamp to roi
		m_pos[m_center][dd] = m_roi[dd].second;
	}
	
	// calculate ND positions for each other offset
	for(size_t oo=0; oo<m_numoffs; oo++) {
		for(size_t dd=0; dd<m_dim; dd++) {
			m_pos[oo][dd] = clamp(m_roi[dd].first, m_roi[dd].second,
					m_pos[m_center][dd]+m_offs[oo][dd]);
		}
	}

	// calculate linear positions from each
	for(size_t oo = 0; oo < m_numoffs; oo++) {
		m_linpos[oo] = 0;
		for(size_t dd=0; dd<m_dim; dd++)
			m_linpos[oo] += m_pos[oo][dd]*m_strides[dd];
	}
	
	m_end = true;
}

} //npl


