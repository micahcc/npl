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
 * @file kdtree.h
 *
 *****************************************************************************/

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
		KDTreeNode(const std::vector<T>& pt, const std::vector<D>& data) :
			left(NULL), right(NULL)
		{
			size_t ii;
			for(ii=0; ii<K && ii<pt.size(); ii++)
				m_point[ii] = pt[ii];
			for(; ii<K; ii++)
				m_point[ii] = 0;
			
			for(ii=0; ii<E && ii<data.size(); ii++)
				m_data[ii] = data[ii];
			for(; ii<E; ii++)
				m_data[ii] = 0;
		};
		
		KDTreeNode(size_t plen, const T* pt, size_t dlen, const D* data) :
			left(NULL), right(NULL)
		{
			size_t ii;
			for(ii=0; ii<K && ii<plen; ii++)
				m_point[ii] = pt[ii];
			for(; ii<K; ii++)
				m_point[ii] = 0;
			
			for(ii=0; ii<E && ii<dlen; ii++)
				m_data[ii] = data[ii];
			for(; ii<E; ii++)
				m_data[ii] = 0;
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
	void insert(const std::vector<T>& pt, const std::vector<D>& data);

	/**
	 * @brief Insert a node (not that this is not dynamic, new nodes won't be
	 * found by search until build() is called)
	 *
	 * @param plen	Length of point array
	 * @param pt	Point
	 * @param dlen	Length of data array
	 * @param data	Node Data
	 */
	void insert(size_t plen, const T* pt, size_t dlen, const D* data);

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
	KDTreeNode<K,E,T,D>* nearest(size_t len, const T* pt, double& dist);

	std::list<const KDTreeNode<K,E,T,D>*> withindist(const std::vector<T>& pt, double dist);
	std::list<const KDTreeNode<K,E,T,D>*> withindist(size_t len, const T* pt, double dist);
	
private:
	
	// whether we have constructed the tree yet
	bool m_built;

	KDTreeNode<K,E,T,D>* m_treehead;
	std::vector<KDTreeNode<K,E,T,D>*> m_allnodes;

	// helper functions
	KDTreeNode<K,E,T,D>* nearest_help(size_t depth, KDTreeNode<K,E,T,D>* pos,
		const T* pt, double& distsq);

	std::list<const KDTreeNode<K,E,T,D>*> withindist_help(size_t depth,
		KDTreeNode<K,E,T,D>* pos, const T* pt, double distsq);

	KDTreeNode<K,E,T,D>* build_helper(
		typename std::vector<KDTreeNode<K,E,T,D>*>::iterator begin,
		typename std::vector<KDTreeNode<K,E,T,D>*>::iterator end,
		size_t depth);
};

}

#include "kdtree.txx"

#endif //KDTREE_H
