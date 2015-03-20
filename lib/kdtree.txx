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
 * @file kdtree.txx
 *
 *****************************************************************************/

#ifndef KDTREE_TXX
#define KDTREE_TXX

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include <iostream>
using namespace std;

namespace npl {

template <size_t K, size_t E, typename T, typename D>
void KDTree<K,E,T,D>::insert(size_t plen, const T* pt, size_t dlen, const D* data)
{
	m_allnodes.push_back(new KDTreeNode<K,E,T,D>(plen, pt, dlen, data));
}

template <size_t K, size_t E, typename T, typename D>
void KDTree<K,E,T,D>::insert(const std::vector<T>& pt, const std::vector<D>& data)
{
	m_allnodes.push_back(new KDTreeNode<K,E,T,D>(pt.size(), pt.data(),
				data.size(), data.data()));
}


/**
 * @brief Recursive helper function to construct the tree.
 *
 * @tparam K	Number of dimensions
 * @tparam E	Number of data elements
 * @tparam T	Type of coordinates
 * @tparam D	Type data stored
 * @param begin	First in range to be inserted
 * @param end	End of range to be inserted / sorted
 * @param depth	Current Depth
 *
 * @return
 */
template <size_t K, size_t E, typename T, typename D>
KDTreeNode<K,E,T,D>* KDTree<K,E,T,D>::build_helper(
		typename std::vector<KDTreeNode<K,E,T,D>*>::iterator begin,
		typename std::vector<KDTreeNode<K,E,T,D>*>::iterator end,
		size_t depth)
{
	size_t length = (size_t)(end-begin);


	////////////////////////////////////////////////
	// sort indexes based on dimension k
	////////////////////////////////////////////////

	size_t axis = depth % K;

	// sort based on the given axis
	std::sort(begin, end,
		[&](KDTreeNode<K,E,T,D>* left, KDTreeNode<K,E,T,D>* right)
		{
			return left->m_point[axis] < right->m_point[axis];
		});

	////////////////////////////////////////////////
	// recurse on the left and right sides
	////////////////////////////////////////////////

	// get median node, and recurse on left/right
	auto median_it = begin+length/2;

	if(length == 1) {
		// no children to be had
		(*median_it)->left = NULL;
		(*median_it)->right = NULL;
	} else if(length == 2) {
		// the median is the lower of the 2
		(*median_it)->left = *(median_it-1);
		(*median_it)->right = NULL;
	} else if(length == 3) {
		// the median is the middle of the 2
		(*median_it)->left = *(median_it-1);
		(*median_it)->right = *(median_it+1);
	} else {
		// too big, need to recurse
		(*median_it)->left = build_helper(begin, median_it, depth+1);
		(*median_it)->right = build_helper(median_it+1, end, depth+1);
	}

	return *median_it;
}

/**
 * @brief Constructs the tree for searching.
 *
 * @tparam K	Number of dimensions
 * @tparam E	Number of data elements
 * @tparam T	Type of coordinates
 * @tparam D	Type data stored
 */
template <size_t K, size_t E, typename T, typename D>
void KDTree<K,E,T,D>::build()
{
	m_treehead = build_helper(m_allnodes.begin(), m_allnodes.end(), 0);
	if(!m_treehead) {
		std::cerr << "Something went wrong with building!" << std::endl;
	}

	m_built = true;
}


/**
 * @brief Takes a depth and subtree and returns the best node
 * within the subtree that beats distsq, if none exists then it returns null
 *
 * @tparam K
 * @tparam E
 * @tparam T
 * @tparam D
 * @param depth
 * @param pos
 * @param pt
 * @param distsq
 *
 * @return
 */
template <size_t K, size_t E, typename T, typename D>
const KDTreeNode<K,E,T,D>* KDTree<K,E,T,D>::nearest_help(size_t depth,
		KDTreeNode<K,E,T,D>* pos, const T* pt, double& distsq) const
{
	size_t axis = depth%K;

	const KDTreeNode<K,E,T,D>* out = NULL;

	// compute our distance to the point
	double curdist = 0;
	for(size_t ii=0; ii<K; ii++) {
		curdist += (pos->m_point[ii]-pt[ii])*(pos->m_point[ii]-pt[ii]);
	}

	// update current best
	if(curdist < distsq) {
		distsq = curdist;
		out = pos;
	}

	if(pos->left && pt[axis] <= pos->m_point[axis]) {
		// if it falls below the current coordinate, and there are nodes to
		// be inspected, recurse
		auto tmp = nearest_help(depth+1, pos->left, pt, distsq);

		// if the search found something, then update the output
		if(tmp)
			out = tmp;

	} else if(pos->right && pt[axis] > pos->m_point[axis]) {
		// if it falls above the current coordinate, and there are nodes to
		// be inspected, recurse
		auto tmp = nearest_help(depth+1, pos->right, pt, distsq);

		// if the search found something, then update the output
		if(tmp)
			out = tmp;
	}

	// out now hold the nearest node in the left or right subtree,
	// or is null, and distsq is the updated best distsqance.
	// If the hyperplane through the current node intersects
	// with the sphere of the best (closest) node, then we need to search
	// the side opposite the original search

	// distance^2 to hyperplane
	double hypdistsq = pow(pt[axis]-pos->m_point[axis],2);

	if(distsq > hypdistsq) {

		/////////////////////////////////////
		// go to the opposite side as before
		/////////////////////////////////////
		if(pos->right && pt[axis] <= pos->m_point[axis]) {
			// if it falls below the current coordinate, and there are nodes to
			// be inspected, recurse

			auto tmp = nearest_help(depth+1, pos->right, pt, distsq);
			if(tmp)
				out = tmp;

		} else if(pos->left && pt[axis] > pos->m_point[axis]) {
			// if it falls above the current coordinate, and there are nodes to
			// be inspected, recurse
			auto tmp = nearest_help(depth+1, pos->left, pt, distsq);

			if(tmp)
				out = tmp;
		}
	}

	return out;
}

/**
 * @brief Reeturns a pointer to a single KDTreeNode that is within distance and
 * is nearest to the center point.
 *
 * @tparam K	Number of dimensions
 * @tparam E	Number of data elements
 * @tparam T	Type of coordinates
 * @tparam D	Type data stored
 * @param pt	Point to search for (must be length K)
 * @param dist	Distance away from pt to search
 *
 * @return 		Pointer to KDTree node that is closest, may be null if none is
 * 				found
 */
template <size_t K, size_t E, typename T, typename D>
const KDTreeNode<K,E,T,D>* KDTree<K,E,T,D>::nearest(const std::vector<T>& pt,
		double& dist) const
{
	return nearest(pt.size(), pt.data(), dist);
}

/**
 * @brief Reeturns a pointer to a single KDTreeNode that is within distance and
 * is nearest to the center point.
 *
 * @tparam K	Number of dimensions
 * @tparam E	Number of data elements
 * @tparam T	Type of coordinates
 * @tparam D	Type data stored
 * @param len	Length (rank/dimension) of point (pt)
 * @param pt	Point to search for (must be length K)
 * @param dist	Distance away from pt to search
 *
 * @return 		Pointer to KDTree node that is closest, may be null if none is
 * 				found
 */
template <size_t K, size_t E, typename T, typename D>
const KDTreeNode<K,E,T,D>* KDTree<K,E,T,D>::nearest(size_t len, const T* pt,
		double& dist) const
{
	if(!m_built) {
		std::cerr << "Error! Must build tree before performing search!"
			<< std::endl;
		return NULL;
	}

	T adjpt[K];
	std::fill(adjpt, adjpt+K, 0);

	for(size_t ii=0; ii<K && ii<len; ii++)
		adjpt[ii] = pt[ii];

	if(std::isnormal(dist)) {
		dist = dist*dist;
	} else {
		dist = INFINITY;
	}

	const KDTreeNode<K,E,T,D>* out = nearest_help(0, m_treehead, adjpt, dist);
	dist = sqrt(dist);

	return out;
}


template <size_t K, size_t E, typename T, typename D>
std::list<const KDTreeNode<K,E,T,D>*> KDTree<K,E,T,D>::withindist_help(size_t depth,
		KDTreeNode<K,E,T,D>* pos, const T* pt, double distsq) const
{
	size_t axis = depth%K;
	std::list<const KDTreeNode<K,E,T,D>*> out;

	// compute our distance to the point
	double curdist = 0;
	for(size_t ii=0; ii<K; ii++)
		curdist += pow(pos->m_point[ii]-pt[ii],2);

	if(curdist < distsq)
		out.push_back(pos);

	// distance^2 to hyperplane
	double hypdistsq = pow(pt[axis]-pos->m_point[axis],2);

	// recurse to the side that pt is on, or to the opposite side if
	// a point could be on the other side and still be under sqrt(distsq)
	// away
	if(pos->left && (pt[axis] <= pos->m_point[axis] || hypdistsq < distsq)) {
		auto tmplist = withindist_help(depth+1, pos->left, pt, distsq);
		out.insert(out.begin(), tmplist.begin(), tmplist.end());
	}

	if(pos->right && (pt[axis] >= pos->m_point[axis] || hypdistsq < distsq)) {
		auto tmplist = withindist_help(depth+1, pos->right, pt, distsq);
		out.insert(out.begin(), tmplist.begin(), tmplist.end());
	}

	return out;
}

/**
 * @brief Returns a list of KDTreeNodes within the given distance
 *
 * @tparam K	Number of dimensions
 * @tparam E	Number of data elements
 * @tparam T	Type of coordinates
 * @tparam D	Type data stored
 * @param pt	Point to search for (must be length K)
 * @param dist	Distance away from pt to search
 *
 * @return 		List of KDTreeNodes which match the criterea
 */
template <size_t K, size_t E, typename T, typename D>
std::list<const KDTreeNode<K,E,T,D>*> KDTree<K,E,T,D>::withindist(
			const std::vector<T>& pt, double dist) const
{
	return withindist(pt.size(), pt.data(), dist);
}

/**
 * @brief Returns a list of KDTreeNodes within the given distance
 *
 * @tparam K	Number of dimensions
 * @tparam E	Number of data elements
 * @tparam T	Type of coordinates
 * @tparam D	Type data stored
 * @param len	Length/rank/dim of the input point.
 * @param pt	Point to search for (must be length K)
 * @param dist	Distance away from pt to search
 *
 * @return 		List of KDTreeNodes which match the criterea
 */
template <size_t K, size_t E, typename T, typename D>
std::list<const KDTreeNode<K,E,T,D>*> KDTree<K,E,T,D>::withindist(
			size_t len, const T* pt, double dist) const
{
	// create adjusted point of known length
	T adjpt[K];
	std::fill(adjpt, adjpt+K, 0);
	for(size_t ii=0; ii<K && ii<len; ii++)
		adjpt[ii] = pt[ii];

	return withindist_help(0, m_treehead, adjpt, dist*dist);
}

template <size_t K, size_t E, typename T, typename D>
void KDTree<K,E,T,D>::clear()
{
	m_built = false;
	m_treehead = NULL;
	m_allnodes.clear();
}

} //npl

#endif //KDTREE_XX
