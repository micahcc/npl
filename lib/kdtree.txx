#ifndef KDTREE_TXX
#define KDTREE_TXX

#include <algorithm>

namespace npl {

template <size_t K, size_t E, typename T, typename D>
void KDTree<K,E,T,D>::insert(const std::vector<T>& pt, const std::vector<T>& data)
{
	m_allnodes.push_back(KDTreeNode<K,E,T,D>(pt, data));
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
		typename std::vector<KDTreeNode<K,E,T,D>>::iterator begin,
		typename std::vector<KDTreeNode<K,E,T,D>>::iterator end,
		size_t depth)
{
	size_t length = (size_t)(end-begin);
	

	////////////////////////////////////////////////
	// sort indexes based on dimension k
	////////////////////////////////////////////////
	
	size_t axis = depth % K;
	
	// sort based on the given axis
	std::sort(begin, end, 
		[&](const KDTreeNode<K,E,T,D>& left, const KDTreeNode<K,E,T,D>& right) 
		{
			return left.m_point[axis] < right.m_point[axis];
		});

#ifdef DEBUG
	std::cerr << "-----------------\nPoints Ordered by dimension " 
			<< axis << std::endl;
	for(auto it = begin; it != end; it++) {
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << it->m_point[ii] << ",";
		}
		std::cerr << std::endl;

	}
#endif //DEBUG

	////////////////////////////////////////////////
	// recurse on the left and right sides
	////////////////////////////////////////////////
	
	// get median node, and recurse on left/right
	auto median_it = begin+length/2;

	if(length == 1) {
		// no children to be had
#ifdef DEBUG
		std::cerr << "NC Leaf: " << std::endl;
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->m_point[ii] << ",";
		}
		std::cerr << std::endl;
#endif //DEBUG
		median_it->left = NULL;
		median_it->right = NULL;
	} else if(length == 2) {
		// the median is the lower of the 2
#ifdef DEBUG
		std::cerr << "1C Leaf: " << std::endl;
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->m_point[ii] << ",";
		}
		std::cerr << std::endl;
#endif //DEBUG
		median_it->left = &(*(median_it-1));
		median_it->right = NULL;
#ifdef DEBUG
		std::cerr << "\tLChild: " << std::endl << "\t";
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->left->m_point[ii] << ",";
		}
		std::cerr << std::endl;
#endif //DEBUG
	} else if(length == 3) {
		// the median is the middle of the 2
#ifdef DEBUG
		std::cerr << "2C Leaf: " << std::endl;
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->m_point[ii] << ",";
		}
		std::cerr << std::endl;
#endif //DEBUG
		median_it->left = &(*(median_it-1));;
		median_it->right = &(*(median_it+1));
#ifdef DEBUG
		std::cerr << "\tLChild: " << std::endl << "\t";
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->left->m_point[ii] << ",";
		}
		std::cerr << std::endl;
		std::cerr << "\tRChild: " << std::endl << "\t";
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->right->m_point[ii] << ",";
		}
		std::cerr << std::endl;
#endif //DEBUG
	} else {
		// too big, need to recurse
		median_it->left = build_helper(begin, median_it, depth+1);
		median_it->right = build_helper(median_it+1, end, depth+1);
		
#ifdef DEBUG
		std::cerr << "2C Intr: " << std::endl;
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->m_point[ii] << ",";
		}
		std::cerr << std::endl;
		std::cerr << "\tLChild: " << std::endl << "\t";
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->left->m_point[ii] << ",";
		}
		std::cerr << std::endl;
		std::cerr << "\tRChild: " << std::endl << "\t";
		for(size_t ii=0; ii<K; ii++) {
			std::cerr << median_it->right->m_point[ii] << ",";
		}
		std::cerr << std::endl;
#endif //DEBUG
	}

	return &(*median_it);
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
KDTreeNode<K,E,T,D>* KDTree<K,E,T,D>::nearest_help(size_t depth, 
		KDTreeNode<K,E,T,D>* pos, const std::vector<T>& pt, double& distsq)
{
	size_t axis = depth%K;

	KDTreeNode<K,E,T,D>* out = NULL;

	// compute our distance to the point
	double curdist = 0;
	for(size_t ii=0; ii<K; ii++) {
		curdist += (pos->m_point[ii]-pt[ii])*(pos->m_point[ii]-pt[ii]);
	}
		
#ifdef DEBUG
	std::cerr << "[" << pos << "] " << curdist << ", (";
	for(size_t ii=0; ii<K; ii++)
		std::cerr << pos->m_point[ii] << ",";
	std::cerr << ")" << std::endl;
#endif //DEBUG

	// update current best
	if(curdist < distsq) {
#ifdef DEBUG
		std::cerr << "New Best!" << std::endl;
#endif //DEBUG
		distsq = curdist;
		out = pos;
	}

	if(pos->left && pt[axis] < pos->m_point[axis]) {
		// if it falls below the current coordinate, and there are nodes to 
		// be inspected, recurse
#ifdef DEBUG
		std::cerr << "[" << pos << "] pt[" << axis << "]=" << pt[axis] << " < " 
			<< "cur[" << axis << "]=" << pos->m_point[axis] 
			<< "\nGoing Left\n";
#endif //DEBUG
		auto tmp = nearest_help(depth+1, pos->left, pt, distsq);

		// if the search found something, then update the output
		if(tmp)
			out = tmp;
	
	} else if(pos->right && pt[axis] > pos->m_point[axis]) {
		// if it falls above the current coordinate, and there are nodes to 
		// be inspected, recurse
#ifdef DEBUG
		std::cerr << "[" << pos << "] pt[" << axis << "]=" << pt[axis] << " > " 
			<< "cur[" << axis << "]=" << pos->m_point[axis] << "\nGoing Right\n";
#endif //DEBUG
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
#ifdef DEBUG
		std::cerr << "[" << pos << "] " << "dsqr (" << distsq << ")" 
			<< " > hyperplane dist (" << pow(pt[axis] - pos->m_point[axis], 2)
			<< ")" << std::endl;
#endif //DEBUG

		/////////////////////////////////////
		// go to the opposite side as before
		/////////////////////////////////////
		if(pos->right && pt[axis] < pos->m_point[axis]) {
			// if it falls below the current coordinate, and there are nodes to 
			// be inspected, recurse
#ifdef DEBUG
			std::cerr << "[" << pos << "] Previous left, now right" << std::endl;
#endif //DEBUG
			
			auto tmp = nearest_help(depth+1, pos->right, pt, distsq);
			if(tmp)
				out = tmp;

		} else if(pos->left && pt[axis] > pos->m_point[axis]) {
			// if it falls above the current coordinate, and there are nodes to 
			// be inspected, recurse
#ifdef DEBUG
			std::cerr << "[" << pos << "] Previous right, now left" << std::endl;
#endif //DEBUG
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
KDTreeNode<K,E,T,D>* KDTree<K,E,T,D>::nearest(const std::vector<T>& pt, double& dist)
{

	if(!m_built) {
		std::cerr << "Error! Must build tree before performing search!" 
			<< std::endl;
		return NULL;
	}

	if(std::isnormal(dist)) {
		dist = dist*dist;
	} else {
		dist = INFINITY;
	}
	
	KDTreeNode<K,E,T,D>* out = nearest_help(0, m_treehead, pt, dist);
	dist = sqrt(dist);

	return out;
}


template <size_t K, size_t E, typename T, typename D>
std::list<KDTreeNode<K,E,T,D>*> KDTree<K,E,T,D>::withindist_help(size_t depth, 
		KDTreeNode<K,E,T,D>* pos, const std::vector<T>& pt, double distsq)
{
	size_t axis = depth%K;
	std::list<KDTreeNode<K,E,T,D>*> out;
	
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
	if(pos->left && (pt[axis] < pos->m_point[axis] || hypdistsq < distsq)) {
#ifdef DEBUG
		std::cerr << "[" << pos << "] " << "pt[" << axis << "]=" << pt[axis] << " > " 
			<< "cur[" << axis << "]" << pos->m_point[axis] << "(";
		for(size_t ii=0; ii<K; ii++)
			std::cerr << pos->m_point[ii] << ",";
		std::cerr << ") Going Left" << std::endl;
#endif //DEBUG
		auto tmplist = withindist_help(depth+1, pos->left, pt, distsq, out);
		out.insert(out.begin(), tmplist.begin(), tmplist.end());
	}

	if(pos->right && (pt[axis] > pos->m_point[axis] || hypdistsq < distsq)) {
#ifdef DEBUG
		std::cerr << "[" << pos << "] " << "pt[" << axis << "]=" << pt[axis] << " < " 
			<< "cur[" << axis << "]" << pos->m_point[axis] << "(";
		for(size_t ii=0; ii<K; ii++)
			std::cerr << pos->m_point[ii] << ",";
		std::cerr << ") Going Left" << std::endl;
#endif //DEBUG
		auto tmplist = withindist_help(depth+1, pos->right, pt, distsq, out);
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
std::list<KDTreeNode<K,E,T,D>*> KDTree<K,E,T,D>::withindist(const std::vector<T>& pt, double dist)
{
	return withindist_help(0, m_treehead, pt, dist*dist);
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
