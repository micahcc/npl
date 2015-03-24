/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file iterators.cpp
 *
 *****************************************************************************/
#include "iterators.h"

namespace npl {

template <> class FlatIter<double>;
template <> class FlatIter<uint8_t>;
template <> class FlatIter<int8_t>;
template <> class FlatIter<uint16_t>;
template <> class FlatIter<int16_t>;
template <> class FlatIter<uint32_t>;
template <> class FlatIter<int32_t>;
template <> class FlatIter<uint64_t>;
template <> class FlatIter<int64_t>;
template <> class FlatIter<float>;
template <> class FlatIter<double>;
template <> class FlatIter<long double>;
template <> class FlatIter<cfloat_t>;
template <> class FlatIter<cdouble_t>;
template <> class FlatIter<cquad_t>;
template <> class FlatIter<rgb_t>;
template <> class FlatIter<rgba_t>;

template <> class FlatConstIter<double>;
template <> class FlatConstIter<uint8_t>;
template <> class FlatConstIter<int8_t>;
template <> class FlatConstIter<uint16_t>;
template <> class FlatConstIter<int16_t>;
template <> class FlatConstIter<uint32_t>;
template <> class FlatConstIter<int32_t>;
template <> class FlatConstIter<uint64_t>;
template <> class FlatConstIter<int64_t>;
template <> class FlatConstIter<float>;
template <> class FlatConstIter<double>;
template <> class FlatConstIter<long double>;
template <> class FlatConstIter<cfloat_t>;
template <> class FlatConstIter<cdouble_t>;
template <> class FlatConstIter<cquad_t>;
template <> class FlatConstIter<rgb_t>;
template <> class FlatConstIter<rgba_t>;

template <> class NDConstIter<double>;
template <> class NDConstIter<uint8_t>;
template <> class NDConstIter<int8_t>;
template <> class NDConstIter<uint16_t>;
template <> class NDConstIter<int16_t>;
template <> class NDConstIter<uint32_t>;
template <> class NDConstIter<int32_t>;
template <> class NDConstIter<uint64_t>;
template <> class NDConstIter<int64_t>;
template <> class NDConstIter<float>;
template <> class NDConstIter<double>;
template <> class NDConstIter<long double>;
template <> class NDConstIter<cfloat_t>;
template <> class NDConstIter<cdouble_t>;
template <> class NDConstIter<cquad_t>;
template <> class NDConstIter<rgb_t>;
template <> class NDConstIter<rgba_t>;

template <> class NDIter<double>;
template <> class NDIter<uint8_t>;
template <> class NDIter<int8_t>;
template <> class NDIter<uint16_t>;
template <> class NDIter<int16_t>;
template <> class NDIter<uint32_t>;
template <> class NDIter<int32_t>;
template <> class NDIter<uint64_t>;
template <> class NDIter<int64_t>;
template <> class NDIter<float>;
template <> class NDIter<double>;
template <> class NDIter<long double>;
template <> class NDIter<cfloat_t>;
template <> class NDIter<cdouble_t>;
template <> class NDIter<cquad_t>;
template <> class NDIter<rgb_t>;
template <> class NDIter<rgba_t>;

template <> class ChunkConstIter<double>;
template <> class ChunkConstIter<uint8_t>;
template <> class ChunkConstIter<int8_t>;
template <> class ChunkConstIter<uint16_t>;
template <> class ChunkConstIter<int16_t>;
template <> class ChunkConstIter<uint32_t>;
template <> class ChunkConstIter<int32_t>;
template <> class ChunkConstIter<uint64_t>;
template <> class ChunkConstIter<int64_t>;
template <> class ChunkConstIter<float>;
template <> class ChunkConstIter<double>;
template <> class ChunkConstIter<long double>;
template <> class ChunkConstIter<cfloat_t>;
template <> class ChunkConstIter<cdouble_t>;
template <> class ChunkConstIter<cquad_t>;
template <> class ChunkConstIter<rgb_t>;
template <> class ChunkConstIter<rgba_t>;

template <> class ChunkIter<double>;
template <> class ChunkIter<uint8_t>;
template <> class ChunkIter<int8_t>;
template <> class ChunkIter<uint16_t>;
template <> class ChunkIter<int16_t>;
template <> class ChunkIter<uint32_t>;
template <> class ChunkIter<int32_t>;
template <> class ChunkIter<uint64_t>;
template <> class ChunkIter<int64_t>;
template <> class ChunkIter<float>;
template <> class ChunkIter<double>;
template <> class ChunkIter<long double>;
template <> class ChunkIter<cfloat_t>;
template <> class ChunkIter<cdouble_t>;
template <> class ChunkIter<cquad_t>;
template <> class ChunkIter<rgb_t>;
template <> class ChunkIter<rgba_t>;

template <> class KernelIter<double>;
template <> class KernelIter<uint8_t>;
template <> class KernelIter<int8_t>;
template <> class KernelIter<uint16_t>;
template <> class KernelIter<int16_t>;
template <> class KernelIter<uint32_t>;
template <> class KernelIter<int32_t>;
template <> class KernelIter<uint64_t>;
template <> class KernelIter<int64_t>;
template <> class KernelIter<float>;
template <> class KernelIter<double>;
template <> class KernelIter<long double>;
template <> class KernelIter<cfloat_t>;
template <> class KernelIter<cdouble_t>;
template <> class KernelIter<cquad_t>;
template <> class KernelIter<rgb_t>;
template <> class KernelIter<rgba_t>;

template <> class KernelIter<double>;
template <> class KernelIter<uint8_t>;
template <> class KernelIter<int8_t>;
template <> class KernelIter<uint16_t>;
template <> class KernelIter<int16_t>;
template <> class KernelIter<uint32_t>;
template <> class KernelIter<int32_t>;
template <> class KernelIter<uint64_t>;
template <> class KernelIter<int64_t>;
template <> class KernelIter<float>;
template <> class KernelIter<double>;
template <> class KernelIter<long double>;
template <> class KernelIter<cfloat_t>;
template <> class KernelIter<cdouble_t>;
template <> class KernelIter<cquad_t>;
template <> class KernelIter<rgb_t>;
template <> class KernelIter<rgba_t>;

}
