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

#ifndef KDTREE_H
#define KDTREE_H

#include <list>
#include <vector>

namespace npl {

template <size_t K, size_t E, typename T = double, typename D = double>
class KDTree;

template <size_t K, size_t E, typename T = double, typename D = double>
class KDTreeNode
{
	public:
		KDTreeNode(const std::vector<T>& pt, const std::vector<T>& data) : 
			left(NULL), right(NULL)
		{
			if(pt.size() != K) {
				std::cerr << "Error, incorrect point dimensionality, "
					"point will not be created!" << std::endl;
				return;
			}
			if(data.size() != E) {
				std::cerr << "Error, incorrect data dimensionality, "
					"point will not be inserted!" << std::endl;
				return;
			}

			for(size_t ii=0; ii<K; ii++)
				m_point[ii] = pt[ii];
			
			for(size_t ii=0; ii<E; ii++)
				m_data[ii] = data[ii];
		};

		T m_point[K];
		D m_data[E];
	private:
		KDTreeNode* left;
		KDTreeNode* right;

		friend KDTree<K,E,T,D>;

		KDTreeNode(const KDTreeNode& other)
		{
			(void)(other);
			throw -1;
		}
		
		KDTreeNode(KDTreeNode&& other)
		{
			(void)(other);
			throw -1;
		}
};

template <size_t K, size_t E, typename T, typename D>
class KDTree
{
public:

	/**
	 * @brief Constructor
	 */
	KDTree() : m_built(false),m_treehead(NULL)  { };

	~KDTree()  {
		while(!m_allnodes.empty()) {
			delete m_allnodes.back() ;
			m_allnodes.pop_back();
		}
	};

	/**
	 * @brief Insert a node (not that this is not dynamic, new nodes won't be 
	 * found by search until build() is called)
	 *
	 * @param pt	Point
	 * @param data	Node Data
	 */
	void insert(const std::vector<T>& pt, const std::vector<T>& data);

	/**
	 * @brief Create Tree, until this is called search won't find anything
	 */
	void build();
	bool built() { return built; };

	/**
	 * @brief Remove all elements
	 */
	void clear();

	KDTreeNode<K,E,T,D>* nearest(const std::vector<T>& pt, double& dist);

	std::list<const KDTreeNode<K,E,T,D>*> withindist(const std::vector<T>& pt, double dist);
	
private:
	
	// whether we have constructed the tree yet
	bool m_built;

	KDTreeNode<K,E,T,D>* m_treehead; 
	std::vector<KDTreeNode<K,E,T,D>*> m_allnodes; 

	// helper functions
	KDTreeNode<K,E,T,D>* nearest_help(size_t depth, KDTreeNode<K,E,T,D>* pos,
		const std::vector<T>& pt, double& distsq);

	std::list<const KDTreeNode<K,E,T,D>*> withindist_help(size_t depth, 
		KDTreeNode<K,E,T,D>* pos, const std::vector<T>& pt, double distsq);

	KDTreeNode<K,E,T,D>* build_helper(
		typename std::vector<KDTreeNode<K,E,T,D>*>::iterator begin,
		typename std::vector<KDTreeNode<K,E,T,D>*>::iterator end,
		size_t depth);
};

}

#include "kdtree.txx"

#endif //KDTREE_H
