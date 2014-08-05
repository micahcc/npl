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
 * @file basic_functions.h
 *
 *****************************************************************************/
#ifndef BASIC_FUNCTIONS_H
#define BASIC_FUNCTIONS_H

namespace npl {

/**********************************************************
 * Functions for dealing with boundaries (wrap and clamp)
 **********************************************************/

/**
 * @brief Clamps value to range of [inf, sup]. Values outside will be pulled
 * to either sup or inf, whichever is closer.
 *
 * @tparam T Value of inf/sup/v/output
 * @param inf Lower bound (infinum)
 * @param sup Upper bound (supremum)
 * @param v Value to clamp
 *
 * @return
 */
template <typename T>
inline T clamp(T inf, T sup, T v)
{
	return std::min(sup, std::max(inf, v));
}

/**
 * @brief Wraps and index based on the range [inf, sup] (when v is outside that)
 * range. Thus inf = 1, sup = 5, v = 0 will wrap around to 4, and v = 6 would wrap
 * around to 1.
 *
 * @tparam T Value of inf/sup/v
 * @param inf lower bound (inclusive)
 * @param sup upper bound (inclusive)
 * @param v valueto wrap. Output will be the position wrapped.
 *
 * @return Wrapped value
 */
template <typename T>
T wrap(T inf, T sup, T v)
{
	T len = sup-inf+1;
	T vtmp = v-inf;
	T out = vtmp < 0 ? sup-((-vtmp-1)%len) : inf+(vtmp%len);
	return out;
}


} // npl

#endif //BASIC_FUNCTIONS_H
