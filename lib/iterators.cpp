#include "iterators.h"

namespace npl {

template <> class BasicIter<double>;
template <> class BasicIter<uint8_t>;
template <> class BasicIter<int8_t>;
template <> class BasicIter<uint16_t>;
template <> class BasicIter<int16_t>;
template <> class BasicIter<uint32_t>;
template <> class BasicIter<int32_t>;
template <> class BasicIter<uint64_t>;
template <> class BasicIter<int64_t>;
template <> class BasicIter<float>;
template <> class BasicIter<double>;
template <> class BasicIter<long double>;
template <> class BasicIter<cfloat_t>;
template <> class BasicIter<cdouble_t>;
template <> class BasicIter<cquad_t>;
template <> class BasicIter<rgb_t>;
template <> class BasicIter<rgba_t>;
}
