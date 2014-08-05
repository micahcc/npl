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
 * @file hasher.h
 *
 *****************************************************************************/

#include <functional>
#include <vector>

template <typename T>
void hash_combine(size_t& seed, T const& v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename It>
size_t hash_range(It first, It last)
{
	size_t seed = 0;
	for(; first != last; ++first) {
		hash_combine(seed, *first);
	}
	return seed;
}

template <typename T>
class hash_vector {
public:
	size_t operator()(const std::vector<T>& v) const
	{
		return hash_range(v.begin(), v.end());
	};
};
