#ifndef BYTESWAP_H
#define BYTESWAP_H

template <typename T>
union Bytes 
{
	T iv;
	unsigned char bytes[sizeof(T)];
} __attribute__((packed));

template <typename T>
void swap(T* val)
{
	Bytes<T> tmp;
	tmp.iv = *val;
	for(size_t ii=0; ii<sizeof(T)/2; ii++)
		std::swap(tmp.bytes[sizeof(T)-ii-1], tmp.bytes[ii]);
	*val = tmp.iv;
}

#endif //BYTESWAP_H
