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

template <> class OrderConstIter<double>;
template <> class OrderConstIter<uint8_t>;
template <> class OrderConstIter<int8_t>;
template <> class OrderConstIter<uint16_t>;
template <> class OrderConstIter<int16_t>;
template <> class OrderConstIter<uint32_t>;
template <> class OrderConstIter<int32_t>;
template <> class OrderConstIter<uint64_t>;
template <> class OrderConstIter<int64_t>;
template <> class OrderConstIter<float>;
template <> class OrderConstIter<double>;
template <> class OrderConstIter<long double>;
template <> class OrderConstIter<cfloat_t>;
template <> class OrderConstIter<cdouble_t>;
template <> class OrderConstIter<cquad_t>;
template <> class OrderConstIter<rgb_t>;
template <> class OrderConstIter<rgba_t>;

template <> class OrderIter<double>;
template <> class OrderIter<uint8_t>;
template <> class OrderIter<int8_t>;
template <> class OrderIter<uint16_t>;
template <> class OrderIter<int16_t>;
template <> class OrderIter<uint32_t>;
template <> class OrderIter<int32_t>;
template <> class OrderIter<uint64_t>;
template <> class OrderIter<int64_t>;
template <> class OrderIter<float>;
template <> class OrderIter<double>;
template <> class OrderIter<long double>;
template <> class OrderIter<cfloat_t>;
template <> class OrderIter<cdouble_t>;
template <> class OrderIter<cquad_t>;
template <> class OrderIter<rgb_t>;
template <> class OrderIter<rgba_t>;

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
