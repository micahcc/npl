#ifndef SLICER_H
#define SLICER_H

#include <vector>
#include <list>

using namespace std;

/**
 * @brief This class is used to slice an image in along a dimension, and to 
 * step an arbitrary direction in an image. Order may be any size from 0 to the
 * number of dimensions. The first member of order will be the fastest moving,
 * and the last will be the slowest. Any not dimensions not included in the order
 * vector will be slower than the last member of order
 */
class Slicer 
{
public:

	/**
	 * @brief Default Constructor, max a length 1, dimension 1 slicer
	 */
	Slicer();
	

	/**
	 * @brief Full Featured Constructor
	 *
	 * @param dim	size of ND array
	 * @param order	order of iteration during ++, this doesn't affect step()
	 * @param roi	min/max, roi is pair<size_t,size_t> = [min,max] 
	 */
	Slicer(std::vector<size_t>& dim, std::list<size_t>& order,
			std::vector<std::pair<size_t,size_t>>& roi);

	/**
	 * @brief Simple (no ROI, no order) Constructor
	 *
	 * @param dim	size of ND array
	 */
	Slicer(std::vector<size_t>& dim);
	
	/**
	 * @brief Constructor that takes a dimension and order of ++/-- iteration
	 *
	 * @param dim	size of ND array
	 * @param order	iteration direction, steps will be fastest in the direction
	 * 				of order[0] and slowest in order.back()
	 */
	Slicer(std::vector<size_t>& dim, std::list<size_t>& order);

	/**
	 * @brief Constructor that takes a dimension and region of interest, which
	 * is defined as min,max (inclusive)
	 *
	 * @param dim	size of ND array
	 * @param roi	min/max, roi is pair<size_t,size_t> = [min,max] 
	 */
	Slicer(std::vector<size_t>& dim, std::vector<std::pair<size_t,size_t>>& roi);

	/**
	 * @brief Directional step, this will not step outside the region of 
	 * interest. Useful for kernels (maybe)
	 *
	 * @param dd	dimension to step in
	 * @param dist	distance to step (may be negative)
	 *
	 * @return new linear index
	 */
	size_t step(size_t dim, int64_t dist = 1);


	/**
	 * @brief Are we at the end in a particular dimension
	 *
	 * @param dim	dimension to check
	 *
	 * @return whether we are at the tail end of the particular dimension
	 */
	bool end(size_t dim);
	
	/**
	 * @brief Are we at the begin in a particular dimension
	 *
	 * @param dim	dimension to check
	 *
	 * @return whether we are at the start of the particular dimension
	 */
	bool begin(size_t dim);

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
	 * @brief dereference operator. Returns the linear position in the array 
	 * given the n-dimensional position.
	 *
	 * @return 
	 */
	size_t operator*()
	{
		return m_linpos;
	};

	/**
	 * @brief Get both ND position and linear position
	 *
	 * @param ndpos	Output, ND position
	 *
	 * @return linear position
	 */
	size_t getPos(std::vector<size_t>& ndpos)
	{
		ndpos.assign(m_pos.begin(), m_pos.end());
		return m_linpos;
	};

	/**
	 * @brief Are we at the begining of iteration?
	 *
	 * @return true if we are at the begining
	 */
	bool isBegin()
	{
		return m_linpos==m_linfirst;
	}

	/**
	 * @brief Are we at the end of iteration? Note that this will be 1 past the
	 * end, as typically is done in c++
	 *
	 * @return true if we are at the end
	 */
	bool isEnd()
	{
		return m_end;
	}

	/**
	 * @brief Are we at the begining of iteration?
	 *
	 * @return true if we are at the begining
	 */
	void setBegin()
	{
		for(size_t ii=0; ii<m_sizes.size(); ii++)
			m_pos[ii] = m_roi[ii].first;
		m_linpos = m_linfirst;
		m_end = false;
	}

	/**
	 * @brief Jump to the end of iteration.
	 *
	 * @return 
	 */
	void setEnd()
	{
		for(size_t ii=0; ii<m_sizes.size(); ii++)
			m_pos[ii] = m_roi[ii].second;
		m_linpos = m_linlast;
		m_end = true;
	}

	/**
	 * @brief Updates dimensions of target nd array
	 *
	 * @param dim	size of nd array, number of dimesions given by dim.size()
	 */
	void updateDim(std::vector<size_t>& dim);

	/**
	 * @brief Sets the region of interest. During iteration or any motion the
	 * position will not move outside the specified range
	 *
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(std::vector<std::pair<size_t, size_t>>& roi);

	/**
	 * @brief Sets the order of iteration from ++/-- operators
	 *
	 * @param order	vector of priorities, with first element being the fastest
	 * iteration and last the slowest. All other dimensions not used will be 
	 * slower than the last
	 */
	void setOrder(std::list<size_t>& order);


	/**
	 * @brief Jump to the given position
	 *
	 * @param newpos	location to move to
	 */
	void setPos(std::vector<size_t>& newpos);


private:

	size_t m_linpos;
	size_t m_linfirst;
	size_t m_linlast;
	bool m_end;
	std::vector<size_t> m_order;
	std::vector<size_t> m_pos;
	std::vector<std::pair<size_t,size_t>> m_roi;

	// these might benefit from being constant
	std::vector<size_t> m_sizes;
	std::vector<size_t> m_strides;
	size_t m_total;

	void updateLinRange();
};

#endif //SLICER_H

