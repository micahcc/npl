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
	bool isBegin() const { return m_linpos==m_linfirst; };

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
	 * @param newpos	location to move to
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
	 * @param ndim size of index
	 * @param index output index variable
	 */
	void index(size_t len, int64_t* index) const;

	/***********************************************
	 *
	 * Modification
	 *
	 **********************************************/

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
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(size_t len, const int64_t* lower, const int64_t* upper);

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
	size_t m_linlast;
	bool m_end;
	std::vector<size_t> m_order;
	std::vector<int64_t> m_pos;
	std::vector<std::pair<int64_t,int64_t>> m_roi;

	// these might benefit from being constant
	std::vector<size_t> m_sizes;
	std::vector<size_t> m_strides;
	size_t m_total;

	void updateLinRange();

	/**
	 * @brief Updates dimensions of target nd array
	 *
	 * @param dim	size of nd array, number of dimesions given by dim.size()
	 */
	void updateDim(size_t ndim, const size_t* dim);
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
	bool isBegin() const { return m_linpos==m_linfirst; };

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
	 * @param newpos	location to move to
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
	 * @param ndim size of index
	 * @param index output index variable
	 */
	void index(size_t len, int64_t* index) const;

	/***********************************************
	 *
	 * Modification
	 *
	 **********************************************/

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
	 * @param roi	pair of [min,max] values in the desired hypercube
	 */
	void setROI(size_t len = 0, const int64_t* lower = NULL, 
			const int64_t* upper = NULL);

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
	 */
	void setChunkSize(size_t len, const int64_t* sizes);

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
	 */
	void setBreaks(size_t len, const int64_t* sizes);

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
	std::vector<size_t> m_pos;
	std::vector<std::pair<int64_t,int64_t>> m_chunk;
	
	// relatively static
	std::vector<size_t>  m_order;
	std::vector<std::pair<int64_t,int64_t>> m_roi;
	std::vector<int64_t> m_chunksizes;

	size_t m_ndim;
	std::vector<size_t> m_dim;
	std::vector<size_t> m_strides;
};

} // npl

#endif //SLICER_H

