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

#include "mrimage.h"
#include "mrimage.txx"

#include "ndarray.h"
#include "nifti.h"
#include "byteswap.h"
#include "slicer.h"

#include "zlib.h"

#include <cstring>

namespace npl {

/* Pre-Compile Certain Image Types */
template class MRImageStore<1, double>;
template class MRImageStore<1, long double>;
template class MRImageStore<1, cdouble_t>;
template class MRImageStore<1, cquad_t>;
template class MRImageStore<1, float>;
template class MRImageStore<1, cfloat_t>;
template class MRImageStore<1, int64_t>;
template class MRImageStore<1, uint64_t>;
template class MRImageStore<1, int32_t>;
template class MRImageStore<1, uint32_t>;
template class MRImageStore<1, int16_t>;
template class MRImageStore<1, uint16_t>;
template class MRImageStore<1, int8_t>;
template class MRImageStore<1, uint8_t>;
template class MRImageStore<1, rgba_t>;
template class MRImageStore<1, rgb_t>;

template class MRImageStore<2, double>;
template class MRImageStore<2, long double>;
template class MRImageStore<2, cdouble_t>;
template class MRImageStore<2, cquad_t>;
template class MRImageStore<2, float>;
template class MRImageStore<2, cfloat_t>;
template class MRImageStore<2, int64_t>;
template class MRImageStore<2, uint64_t>;
template class MRImageStore<2, int32_t>;
template class MRImageStore<2, uint32_t>;
template class MRImageStore<2, int16_t>;
template class MRImageStore<2, uint16_t>;
template class MRImageStore<2, int8_t>;
template class MRImageStore<2, uint8_t>;
template class MRImageStore<2, rgba_t>;
template class MRImageStore<2, rgb_t>;

template class MRImageStore<3, double>;
template class MRImageStore<3, long double>;
template class MRImageStore<3, cdouble_t>;
template class MRImageStore<3, cquad_t>;
template class MRImageStore<3, float>;
template class MRImageStore<3, cfloat_t>;
template class MRImageStore<3, int64_t>;
template class MRImageStore<3, uint64_t>;
template class MRImageStore<3, int32_t>;
template class MRImageStore<3, uint32_t>;
template class MRImageStore<3, int16_t>;
template class MRImageStore<3, uint16_t>;
template class MRImageStore<3, int8_t>;
template class MRImageStore<3, uint8_t>;
template class MRImageStore<3, rgba_t>;
template class MRImageStore<3, rgb_t>;

template class MRImageStore<4, double>;
template class MRImageStore<4, long double>;
template class MRImageStore<4, cdouble_t>;
template class MRImageStore<4, cquad_t>;
template class MRImageStore<4, float>;
template class MRImageStore<4, cfloat_t>;
template class MRImageStore<4, int64_t>;
template class MRImageStore<4, uint64_t>;
template class MRImageStore<4, int32_t>;
template class MRImageStore<4, uint32_t>;
template class MRImageStore<4, int16_t>;
template class MRImageStore<4, uint16_t>;
template class MRImageStore<4, int8_t>;
template class MRImageStore<4, uint8_t>;
template class MRImageStore<4, rgba_t>;
template class MRImageStore<4, rgb_t>;

template class MRImageStore<5, double>;
template class MRImageStore<5, long double>;
template class MRImageStore<5, cdouble_t>;
template class MRImageStore<5, cquad_t>;
template class MRImageStore<5, float>;
template class MRImageStore<5, cfloat_t>;
template class MRImageStore<5, int64_t>;
template class MRImageStore<5, uint64_t>;
template class MRImageStore<5, int32_t>;
template class MRImageStore<5, uint32_t>;
template class MRImageStore<5, int16_t>;
template class MRImageStore<5, uint16_t>;
template class MRImageStore<5, int8_t>;
template class MRImageStore<5, uint8_t>;
template class MRImageStore<5, rgba_t>;
template class MRImageStore<5, rgb_t>;

template class MRImageStore<6, double>;
template class MRImageStore<6, long double>;
template class MRImageStore<6, cdouble_t>;
template class MRImageStore<6, cquad_t>;
template class MRImageStore<6, float>;
template class MRImageStore<6, cfloat_t>;
template class MRImageStore<6, int64_t>;
template class MRImageStore<6, uint64_t>;
template class MRImageStore<6, int32_t>;
template class MRImageStore<6, uint32_t>;
template class MRImageStore<6, int16_t>;
template class MRImageStore<6, uint16_t>;
template class MRImageStore<6, int8_t>;
template class MRImageStore<6, uint8_t>;
template class MRImageStore<6, rgba_t>;
template class MRImageStore<6, rgb_t>;

template class MRImageStore<7, double>;
template class MRImageStore<7, long double>;
template class MRImageStore<7, cdouble_t>;
template class MRImageStore<7, cquad_t>;
template class MRImageStore<7, float>;
template class MRImageStore<7, cfloat_t>;
template class MRImageStore<7, int64_t>;
template class MRImageStore<7, uint64_t>;
template class MRImageStore<7, int32_t>;
template class MRImageStore<7, uint32_t>;
template class MRImageStore<7, int16_t>;
template class MRImageStore<7, uint16_t>;
template class MRImageStore<7, int8_t>;
template class MRImageStore<7, uint8_t>;
template class MRImageStore<7, rgba_t>;
template class MRImageStore<7, rgb_t>;

template class MRImageStore<8, double>;
template class MRImageStore<8, long double>;
template class MRImageStore<8, cdouble_t>;
template class MRImageStore<8, cquad_t>;
template class MRImageStore<8, float>;
template class MRImageStore<8, cfloat_t>;
template class MRImageStore<8, int64_t>;
template class MRImageStore<8, uint64_t>;
template class MRImageStore<8, int32_t>;
template class MRImageStore<8, uint32_t>;
template class MRImageStore<8, int16_t>;
template class MRImageStore<8, uint16_t>;
template class MRImageStore<8, int8_t>;
template class MRImageStore<8, uint8_t>;
template class MRImageStore<8, rgba_t>;
template class MRImageStore<8, rgb_t>;

} // npl
