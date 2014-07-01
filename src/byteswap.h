#ifndef BYTESWAP_H
#define BYTESWAP_H

template <typename T>
union Bytes
{
	T iv;
	unsigned char bytes[sizeof(T)];
};

template <typename T>
T swap(T val)
{
	Bytes<T> tmp1;
	Bytes<T> tmp2;
	tmp1.iv = val;
	for(size_t ii=0; ii<sizeof(T); ii++)
		tmp2.bytes[ii] = tmp1.bytes[sizeof(T)-ii-1];
	return tmp2.iv;
}

#endif //BYTESWAP_H
