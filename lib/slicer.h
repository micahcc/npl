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
 * @file slicer.h
 *
 *****************************************************************************/

#ifndef SLICER_H
#define SLICER_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>

using namespace std;

namespace npl {

/**
 * @brief This class is used to step through an ND array in order of dimensions.
 * Order may be any size from 0 to the number of dimensions. The first member
 * of order will be the fastest moving, and the last will be the slowest. Any
 * not dimensions not included in the order vector will be slower than the last
 * member of order
 */
class Slicer
{
public:
	
	/****************************************
	 *
	 * Constructors
	 *
	 ****************************************/

	/**
	 * @brief Default Constructor, max a length 1, dimension 1 slicer
	 */
	Slicer();

	/**
	 * @brief Constructor, takses the number of dimensions and the size of the
	 * image.
	 *
	 * @param ndim	size of ND array
	 * @param dim	array providing the size in each dimension
	 */
	Slicer(size_t ndim, const size_t* dim);

	/**
	 * @brief Updates dimensions of target nd array
	 *
	 * @param ndim Rank (dimensionality) of data block, length of dim
     * @param dim Size of data block, in each dimension, so dim = {32, 2,54 }
     * would have 32*2*54 members 
	 */
	void setDim(size_t ndim, const size_t* dim);
	
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
	bool isBegin() const { return !isEnd() && m_linpos==m_linfirst; };

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
	Slicer& operator++();
	
	/**
	 * @brief Prefix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	Slicer& operator--();

	/**
	 * @brief Are we at the begining of iteration?
	 *
	 */
	void goBegin();

	/**
	 * @brief Jump to the end of iteration.
	 *
	 */
	void goEnd();

	/**
	 * @brief Jump to the given position, additional values in newpos beyond dim
	 * will be ignored. Any values missing due to ndim > len will be treated as
	 * zeros.
	 *
	 * @param len Length of newpos array
	 * @param newpos location to move to
	 */
	void goIndex(size_t len, int64_t* newpos);
	
	/**
	 * @brief Jump to the given position
	 *
	 * @param newpos location to move to
	 */
	void goIndex(std::vector<int64_t> newpos);

	/****************************************
	 *
	 * Actually get the linear location
	 *
	 ***************************************/

	/**
	 * @brief dereference operator. Returns the linear position in the array
	 * given the n-dimensional position.
	 *
	 * @return The current linear index.
	 */
	inline
	int64_t operator*() const { return m_linpos; };
	
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
	void index(size_t len, int64_t* index) const;
	
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
	void index(size_t len, int* index) const;

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
	void index(size_t len, double* index) const;
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param ind output index variable
	 */
	void index(std::vector<int64_t>& ind) const
	{
		index(ind.size(), ind.data()); 
	}
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param ind output index variable
	 */
	void index(std::vector<int>& ind) const
	{
		index(ind.size(), ind.data()); 
	};

	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param ind output index variable
	 */
	void index(std::vector<double>& ind) const 
	{
		index(ind.size(), ind.data()); 
	};

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
     * @param len length of roi array
     * @param roisize Size of ROI (which runs in the block from:
     * [0 to roisize[0]-1,0. to roisize[1]-1, etc]
     */
    void setROI(size_t len, const size_t* roisize);

    /**
     * @brief Sets the region of interest, with lower bound of 0.
     * During iteration or any motion the
     * position will not move outside the specified range. Invalidates position.
     *
     * Invalidates position
     *
     * @param len length of roi array
     * @param roisize Size of ROI (which runs in the block from:
     * [0 to roisize[0]-1,0. to roisize[1]-1, etc]
     */
    void setROI(size_t len, const int64_t* roisize);

	/**
	 * @brief Sets the region of interest. During iteration or any motion the
	 * position will not move outside the specified range. Extra elements in
	 * roi beyond the number of dimensions, are ignored
	 *
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(const std::vector<std::pair<int64_t, int64_t>>& roi);

	/**
	 * @brief Sets the region of interest. During iteration or motion the
	 * position will not move outside the specified range
	 *
	 * @param len	Length of both lower and upper arrays.
	 * @param lower	Coordinate at lower bound of bounding box.
	 * @param upper	Coordinate at upper bound of bounding box.
	 */
	void setROI(size_t len = 0, const int64_t* lower = NULL, 
			const int64_t* upper = NULL);

	/**
	 * @brief Sets the order of iteration from ++/-- operators
	 *
	 * @param order	vector of priorities, with first element being the fastest
	 * iteration and last the slowest. All other dimensions not used will be
	 * slower than the last
	 * @param revorder	Reverse order, in which case the first element of order
	 * 					will have the slowest iteration, and dimensions not
	 * 					specified in order will be faster than those included.
	 */
	void setOrder(const std::vector<size_t>& order, bool revorder = false);
	
	/**
	 * @brief Sets the order of iteration from ++/-- operators
	 *
	 * @param order	vector of priorities, with first element being the fastest
	 * iteration and last the slowest. All other dimensions not used will be
	 * slower than the last
	 * @param revorder	Reverse order, in which case the first element of order
	 * 					will have the slowest iteration, and dimensions not
	 * 					specified in order will be faster than those included.
	 */
	void setOrder(std::initializer_list<size_t> order, bool revorder = false);
	
	/**
	 * @brief Sets the order of iteration from ++/-- operators. Order will be
	 * the default (highest to lowest)
	 *
	 */
	void setOrder();

	/**
	 * @brief Returns the array giving the order of dimension being traversed.
	 * So 3,2,1,0 would mean that the next point in dimension 3 will be next,
	 * when wrapping the next point in 2 is visited, when that wraps the next
	 * in one and so on.
	 *
	 * @return Order of dimensions
	 */
	const std::vector<size_t>& getOrder() const { return m_order; } ;

protected:
	
	/******************************************
	 *
	 * Offset, useful to kernel processing
	 *
	 ******************************************/
	
	size_t m_linpos;
	size_t m_linfirst;
	std::vector<int64_t> m_pos;
	bool m_end;

	size_t m_ndim;
	std::vector<size_t> m_order;
	std::vector<std::pair<int64_t,int64_t>> m_roi;
	std::vector<size_t> m_dim;
	std::vector<size_t> m_strides;
};

/**
 * @brief This class is used to step through an ND array in order of
 * dimensions, but unlike Slicer it breaks the NDArray into chunks. Iteration
 * stops at the end of each chunk until nextChunk is called. By default only
 * one chunk is used, equal to the entire image. Use setBreaks() to set the
 * frequency of chunks. The input to setBreaks() is an array of integers, where
 * the next chunk occurs when dist%break == 0, with the special property that
 * break=0 indicates no breaks for the dimension. So for a 3D image,
 * setBreaks({1,0,0}) will stop every time a new x-values is reached. Note that
 * the affects the order of iteration, so x cannot be the fastest iterator.
 * setChunkSize() is an alias for setBreaks. Note that in cases where
 * break%size != 0, for example image size = 5 and break = 3, chunk sizes will
 * differ during the course of iteration!
 *
 * Order may still be set to decide the order of iteration. The first member
 * of setOrder will be the fastest moving, and the last will be the slowest.
 * Any dimensions not included in the order vector will be slower than the last
 * member of order. Note that order will not necessarily be strictly obeyed
 * when more than one chunk size is > 0. For isntance if setBreaks({1,1,0}), 
 * setOrder({0,1,2}) are used, then all of 2 will visited before we iterate in
 * 0 or 1, because otherwise we would be stepping across chunks.
 */
class ChunkSlicer 
{
public:
	
	/****************************************
	 *
	 * Constructors
	 *
	 ****************************************/

	/**
	 * @brief Default Constructor, max a length 1, dimension 1 slicer
	 */
	ChunkSlicer();

	/**
	 * @brief Constructor, takses the number of dimensions and the size of the
	 * image.
	 *
	 * @param ndim	size of ND array
	 * @param dim	array providing the size in each dimension
	 */
	ChunkSlicer(size_t ndim, const size_t* dim);

	/**
	 * @brief Sets the dimensionality of iteration, and the dimensions of the image
	 *
	 * Invalidates position and ROI.
	 *
	 * @param ndim Number of dimension
	 * @param dim Dimensions (size)
	 */
	void setDim(size_t ndim, const size_t* dim);
	
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
	bool isBegin() const { return !isEnd() && m_linpos==m_linfirst; };

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

	/**
	 * @brief Returns true when we have reached the end of the chunk.
	 *
	 * @return Are we ready for the next chunk?
	 */
	bool isChunkBegin() const { return m_linpos==m_chunkfirst; };

	/**
	 * @brief Returns true when we have reached the end of the chunk.
	 *
	 * @return Are we ready for the next chunk?
	 */
	bool isChunkEnd() const { return m_chunkend; };

	/**
	 * @brief Returns true when we have reached the end of the chunk.
	 *
	 * @return Are we ready for the next chunk?
	 */
	bool eoc() const { return m_chunkend; };

	/*************************************
	 * Movement
	 ***********************************/

	/**
	 * @brief Prefix iterator. Iterates in the order dictatored by the dimension
	 * order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	ChunkSlicer& operator++();
	
	/**
	 * @brief Prefix negative  iterator. Iterates in the order dictatored by
	 * the dimension order passsed during construction or by setOrder
	 *
	 * @return 	new value of linear position
	 */
	ChunkSlicer& operator--();
	
	/**
	 * @brief Proceed to the next chunk (if there is one).
	 *
	 * @return 	
	 */
	ChunkSlicer& nextChunk();
	
	/**
	 * @brief Return to the previous chunk (if there is one).
	 *
	 * @return 	new value of linear position
	 */
	ChunkSlicer& prevChunk();

	/**
	 * @brief Go to the beginning for the current chunk
	 *
	 */
	void goChunkBegin();

	/**
	 * @brief Jump to the end of current chunk.
	 *
	 */
	void goChunkEnd();

	/**
	 * @brief Go to the very beginning for the first chunk
	 *
	 */
	void goBegin();

	/**
	 * @brief Jump to the end of the last chunk.
	 *
	 */
	void goEnd();

	/**
	 * @brief Jump to the given position, additional values in newpos beyond dim
	 * will be ignored. Any values missing due to ndim > len will be treated as
	 * zeros.
	 *
	 * @param len length of newpos array 
	 * @param newpos Position to move to 
	 */
	void goIndex(size_t len, int64_t* newpos);
	
	/**
	 * @brief Jump to the given position
	 *
	 * @param newpos	location to move to
	 */
	void goIndex(std::vector<int64_t> newpos);

	/****************************************
	 *
	 * Actually get the linear location
	 *
	 ***************************************/

	/**
	 * @brief dereference operator. Returns the linear position in the array
	 * given the n-dimensional position.
	 *
	 * @return
	 */
	inline
	int64_t operator*() const { return m_linpos; };
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param len size of index array
	 * @param index output index variable
	 */
	void index(size_t len, int64_t* index) const;
	
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
	void index(size_t len, int* index) const;
	
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
	void index(size_t len, double* index) const;
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param ind output index variable
	 */
	void index(std::vector<int64_t>& ind) const
	{
		index(ind.size(), ind.data());
	};
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param ind output index variable
	 */
	void index(std::vector<int>& ind) const
	{
		index(ind.size(), ind.data());
	};
	
	/**
	 * @brief Places the first len dimension in the given array. If the number
	 * of dimensions exceed the len then the additional dimensions will be
	 * ignored, if len exceeds the dimensionality then index[dim...len-1] = 0.
	 * In other words index will be completely overwritten in the most sane way
	 * possible if the internal dimensions and size index differ.
	 *
	 * @param ind output index variable
	 */
	void index(std::vector<double>& ind) const
	{
		index(ind.size(), ind.data());
	};

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
     * @param len length of roi array
     * @param roisize Size of ROI (which runs in the block from:
     * [0 to roisize[0]-1,0. to roisize[1]-1, etc]
     */
    void setROI(size_t len, const size_t* roisize);

    /**
     * @brief Sets the region of interest, with lower bound of 0.
     * During iteration or any motion the
     * position will not move outside the specified range. Invalidates position.
     *
     * Invalidates position
     *
     * @param len length of roi array
     * @param roisize Size of ROI (which runs in the block from:
     * [0 to roisize[0]-1,0. to roisize[1]-1, etc]
     */
    void setROI(size_t len, const int64_t* roisize);

	/**
	 * @brief Sets the region of interest. During iteration or any motion the
	 * position will not move outside the specified range. Extra elements in
	 * roi beyond the number of dimensions, are ignored
	 *
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(const std::vector<std::pair<int64_t, int64_t>>& roi);

	/**
	 * @brief Sets the region of interest. During iteration or motion the
	 * position will not move outside the specified range
	 *
	 * @param len	Length of both lower and upper arrays.
	 * @param lower	Coordinate at lower bound of bounding box.
	 * @param upper	Coordinate at upper bound of bounding box.
	 */
	void setROI(size_t len = 0, const int64_t* lower = NULL, 
			const int64_t* upper = NULL);

	/**
	 * @brief Set the sizes of chunks for each dimension. Chunks will end every
	 * N steps in each of the provided dimension, with the caveout that 0
	 * indicates no breaks in the given dimension. So size = {0, 2, 2} will
	 * cause chunks to after \f$\{N_x-1, 1, 1\}\f$. \f$\{0,0,0\}\f$ (the default)
	 * indicate * that the entire image will be iterated and only one chunk
	 * will be used. Defunity will cause the default to be 1 for non-specified
	 * dimensions.
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
	void setChunkSize(size_t len, const int64_t* sizes, bool defunity = false);

	/**
	 * @brief Set the sizes of chunks for each dimension. Chunks will end every
	 * N steps in each of the provided dimension, with the caveout that 0
	 * indicates no breaks in the given dimension. So size = {0, 2, 2} will
	 * cause chunks to after \f$\{N_x-1, 1, 1\}\f$. \f$\{0,0,0\}\f$ (the default)
	 * indicate * that the entire image will be iterated and only one chunk
	 * will be used. Defunity will cause the default to be 1 for non-specified
	 * dimensions.
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
	void setChunkSize(size_t len, const size_t* sizes, bool defunity = false);

	/**
	 * @brief Sets the chunk sizes so that each chunk is a line in the given 
	 * dimension. This would be analogous to itk's linear iterator. 
	 * Usage:
	 *
	 * it.setLineChunk(0);
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
	void setLineChunk(size_t dir);

	/**
	 * @brief Sets the order of iteration from ++/-- operators
	 *
	 * @param order	vector of priorities, with first element being the fastest
	 * iteration and last the slowest. All other dimensions not used will be
	 * slower than the last
	 * @param revorder	Reverse order, in which case the first element of order
	 * 					will have the slowest iteration, and dimensions not
	 * 					specified in order will be faster than those included.
	 */
	void setOrder(const std::vector<size_t>& order, bool revorder = false);
	
	/**
	 * @brief Sets the order of iteration from ++/-- operators
	 *
	 * @param order	vector of priorities, with first element being the fastest
	 * iteration and last the slowest. All other dimensions not used will be
	 * slower than the last
	 * @param revorder	Reverse order, in which case the first element of order
	 * 					will have the slowest iteration, and dimensions not
	 * 					specified in order will be faster than those included.
	 */
	void setOrder(std::initializer_list<size_t> order, bool revorder = false);

	/**
	 * @brief Returns the array giving the order of dimension being traversed.
	 * So 3,2,1,0 would mean that the next point in dimension 3 will be next,
	 * when wrapping the next point in 2 is visited, when that wraps the next
	 * in one and so on.
	 *
	 * @return Order of dimensions
	 */
	const std::vector<size_t>& getOrder() const { return m_order; } ;

protected:
	
	/******************************************
	 *
	 * Offset, useful to kernel processing
	 *
	 ******************************************/
	
	// used to determine begin/end status
	size_t m_linfirst;
	size_t m_chunkfirst;
	bool m_end;
	bool m_chunkend;
	
	// position
	size_t m_linpos;
	std::vector<int64_t> m_pos;
	std::vector<std::pair<int64_t,int64_t>> m_chunk;
	
	// relatively static
	std::vector<size_t> m_order;
	std::vector<std::pair<int64_t,int64_t>> m_roi;
	std::vector<int64_t> m_chunksizes;

	size_t m_ndim;
	std::vector<size_t> m_dim;
	std::vector<size_t> m_strides;
};

/**
 * @brief This class is used to slice an image in along a dimension, and to
 * step an arbitrary direction in an image. Order may be any size from 0 to the
 * number of dimensions. The first member of order will be the fastest moving,
 * and the last will be the slowest. Any not dimensions not included in the order
 * vector will be slower than the last member of order.
 *
 * TODO: Change API to be more intuitive.
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
	 * @param kradius Radius in all directions.
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
	 * @param krange Vector of [inf, sup] in each dimension. Unaddressed (missing)
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
	int64_t getC() const
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
	 * @param len size of index
	 * @param index output index variable
	 */
	void indexC(size_t len, int64_t* index) const;
	
	/**
	 * @brief Get index of i'th kernel (center-offset) element.
	 *
	 * @return linear position
	 */
	inline
	int64_t getK(int64_t kit) const {
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
	void indexK(size_t kit, size_t len, int64_t* index, bool bound = true) const;

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
	int64_t offsetK(size_t kit, size_t dim);
	
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
	void offsetK(size_t kit, size_t len, int64_t* dindex) const;

	/**
	 * @brief Get linear position
	 *
	 * @return linear position
	 */
	size_t ksize() const
	{
		return m_numoffs;
	};

	/**
	 * @brief All around intializer. Sets all internal variables.
	 *
	 * @param ndim 	Rank (number of dimensions), also length of dim array
     * @param dim Dimension (size) of memory block.
	 */
	void setDim(size_t ndim, const size_t* dim);

protected:

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

