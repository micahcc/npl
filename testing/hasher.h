#include <functional>

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
