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
	std::vector<std::pair<int64_t, int64_t>> tmpk;
	initialize(1, &tmp, tmpk);
};

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
KSlicer::KSlicer(size_t ndim, const size_t* dim, 
		const std::vector<std::pair<int64_t, int64_t>>& krange)
{
	initialize(ndim, dim, krange);
}
	
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
KSlicer::KSlicer(size_t ndim, const size_t* dim, 
		const std::vector<size_t>& kradius)
{
	std::vector<std::pair<int64_t,int64_t>> range(kradius.size());
	for(size_t dd=0; dd<kradius.size(); dd++) {
		int64_t tmp = kradius[dd];
		range[dd].first = -tmp;
		range[dd].second = tmp;
	}

	initialize(ndim, dim, range);
}
	
/**
 * @brief Postfix iterator. Iterates in the order dictatored by the dimension
 * order passsed during construction or by setOrder
 *
 * @param int	unused
 *
 * @return 	old value of linear position
 */
size_t KSlicer::operator++(int)
{
	size_t ret = m_linpos[m_center];
	operator++();
	return ret;
}

/**
 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
 * order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
size_t KSlicer::operator++() 
{
	if(isEnd())
		return m_linpos[m_center];
	
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
				m_linpos[m_center] += m_strides[dd];
				break;
			} else if(ii != m_dim-1){
				// reset dimension
				m_pos[m_center][dd] = m_roi[dd].first;
			} else {
				// we are willing to go 1 past the last
				m_pos[m_center][dd]++;
				m_end = true;

				// want to skip clamping, and not really a need to update 
				// neighborhood when we are outside the image
				return m_linpos[m_center]; 
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

	return m_linpos[m_center];
}

/**
 * @brief Postfix negative  iterator. Iterates in the order dictatored by
 * the dimension order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
size_t KSlicer::operator--(int)
{
	size_t ret = m_linpos[m_center];
	operator--();
	return ret;
};

/**
 * @brief Prefix negative  iterator. Iterates in the order dictatored by
 * the dimension order passsed during construction or by setOrder
 *
 * @return 	new value of linear position
 */
size_t KSlicer::operator--() 
{
	if(isBegin())
		return m_linpos[m_center];
	
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

	return m_linpos[m_center];
}

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
void KSlicer::initialize(size_t ndim, const size_t* dim, 
			const std::vector<std::pair<int64_t, int64_t>>& krange)
{
	m_dim = ndim;
	m_size.assign(dim, dim+ndim);

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

	// to ensure we aren't constantly bouncing wrapping, we will iterate
	// in the direction of the longest dimension, rather than the fastest
	int64_t longest = 0;
	m_order.resize(m_dim);
	for(size_t dd=0; dd<m_dim; dd++) {
		if((int64_t)m_size[dd]+kmin[dd]-kmax[dd] >= longest) {
			longest = (int64_t)m_size[dd]+kmin[dd]-kmax[dd]; 
			m_order[0] = dd;
		}
	}

	// just add the remaining dimensions to order in reverse
	for(int64_t ii=m_dim-1, jj=1; ii>=0; ii--) {
		if(ii != m_order[0])
			m_order[jj++] = ii;
	}

	// set up strides
	m_strides.resize(m_dim);
	m_strides[m_dim-1] = 1;
	for(int64_t ii=(int64_t)m_dim-2; ii>=0; ii--) {
		m_strides[ii] = m_strides[ii+1]*dim[ii+1];
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
	
	// set up ROI, and calculate the m_begin location
	m_begin = 0;
	m_roi.resize(m_dim);
	for(size_t ii=0; ii<m_dim; ii++) {
		// default to full range 
		m_roi[ii].first = 0;
		m_roi[ii].second = dim[ii]-1;
	}
	
	m_pos.resize(m_numoffs);
	fill(m_pos.begin(), m_pos.end(), std::vector<size_t>(m_dim));
	m_linpos.resize(m_numoffs);
	
	goBegin();
};

void KSlicer::setROI(const std::vector<std::pair<size_t,size_t>>& roi)
{
	// set up ROI, and calculate the m_begin location
	m_begin = 0;
	for(size_t ii=0; ii<m_dim; ii++) {
		if(ii<roi.size()) {
			m_roi[ii] = roi[ii];
		} else {
			// default to full range 
			m_roi[ii].first = 0;
			m_roi[ii].second = m_size[ii]-1;
		}

		m_begin += m_roi[ii].first*m_strides[ii];
	}

	goBegin();
}

/**
 * @brief Jump to the given position
 *
 * @param newpos	location to move to
 * @param outside	variable to return whether we are outside the ROI or not
 */
void KSlicer::goIndex(const std::vector<size_t>& newpos, bool* outside)
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

	// set up outside if we have been asked for it
	if(outside)
		*outside = false;
	
	// copy/clamp the center
	size_t clamped;
	// copy the center
	for(size_t dd = 0; dd<m_dim; dd++) {
		// clamp to roi
		clamped = clamp(m_roi[dd].first, m_roi[dd].second, newpos[dd]);
		
		// if clamping had an effect, then set outside
		if(outside && clamped != newpos[dd])
			*outside = true;

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


