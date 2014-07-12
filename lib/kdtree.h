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
};

template <size_t K, size_t E, typename T, typename D>
class KDTree
{
public:

	/**
	 * @brief Constructor
	 */
	KDTree() : m_built(false),m_treehead(NULL)  { };

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
	std::list<KDTreeNode<K,E,T,D>*> withindist(const std::vector<T>& pt, double dist);
	
private:
	
	// whether we have constructed the tree yet
	bool m_built;

	KDTreeNode<K,E,T,D>* m_treehead; 
	std::vector<KDTreeNode<K,E,T,D>> m_allnodes; 

	// helper functions
	KDTreeNode<K,E,T,D>* nearest_help(size_t depth, KDTreeNode<K,E,T,D>* pos,
		const std::vector<T>& pt, double& distsq);

	std::list<KDTreeNode<K,E,T,D>*> withindist_help(size_t depth, 
		KDTreeNode<K,E,T,D>* pos, const std::vector<T>& pt, double distsq);

	KDTreeNode<K,E,T,D>* build_helper(
		typename std::vector<KDTreeNode<K,E,T,D>>::iterator begin,
		typename std::vector<KDTreeNode<K,E,T,D>>::iterator end,
		size_t depth);
};

}

#include "kdtree.txx"

#endif //KDTREE_H
