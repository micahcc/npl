/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
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
