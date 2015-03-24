/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
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
