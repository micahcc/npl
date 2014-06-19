#include <cstddef>
#include <cmath>
#include <initializer_list>

class NDArray
{
public:
	virtual double getD(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0) = 0;
	
	double operator()(int x = 0, int y = 0, int z = 0, int t = 0, int u = 0,
			int v = 0, int w = 0);

	virtual int getI(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0) = 0;
	
	virtual void setD(double newval, int x = 0, int y = 0, int z = 0, 
			int t = 0, int u = 0, int v = 0, int w = 0) = 0;

	virtual void setI(int newval, int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0) = 0;

	virtual size_t getBytes() = 0;
	virtual size_t getNDim() = 0;
	virtual size_t dim(size_t dir) = 0;
	virtual size_t* dim() = 0;
};

/**
 * @brief Basic storage unity for ND array. Creates a big chunk of memory.
 *
 * @tparam D dimension of array
 * @tparam T type of sample
 */
template <int D, typename T>
class NDArrayStore : public virtual NDArray
{
public:
	NDArrayStore(std::initializer_list<size_t> a_args);
	NDArrayStore(size_t dim[D]);
	NDArrayStore();
	
	~NDArrayStore() { delete[] m_data; };

	virtual size_t getAddr(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);
	virtual size_t getAddr(size_t index[D]);

	virtual double getD(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);

	virtual int getI(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);
	
	virtual void setD(double newval, int x = 0, int y = 0, int z = 0, 
			int t = 0, int u = 0, int v = 0, int w = 0);

	virtual void setI(int newval, int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);

	virtual double operator()(int x = 0, int y = 0, int z = 0, int t = 0, int u = 0,
			int v = 0, int w = 0);

	virtual size_t getBytes();
	virtual size_t getNDim();
	virtual size_t dim(size_t dir);
	virtual size_t* dim();

	virtual void resize(size_t dim[D]);
	
	T* m_data;
	size_t m_dim[D];	// overall image dimension
};

typedef NDArrayStore<2, double> Matrix;

/**
 * @brief Initializes an array with a size and a chache size. The layout will
 * be cubes in each dimension to make the clusters.
 *
 * @tparam D
 * @tparam T
 * @param dim[D]
 * @param csize
 */
template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(size_t dim[D])
{
	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		m_dim[ii] = dim[ii];
		dsize *= m_dim[ii];
	}

	m_data = new T[dsize];
}

template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(std::initializer_list<size_t> a_args)
{
	size_t dsize = 1;
	size_t ii;
	
	// set dimensions with matching size to the minimum length, ignoring
	// any extra parts of a_args
	auto it = a_args.begin();
	for(ii=0; ii < D && it != a_args.end(); ii++, ++it) {
		m_dim[ii] = *it;
		dsize *= m_dim[ii];
	}

	// make any remaining dimensions size 1
	for(ii=0; ii < D; ii++) {
		m_dim[ii] = 1;
		dsize *= m_dim[ii];
	}

	m_data = new T[dsize];
}

template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore()
{
	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		m_dim[ii] = 1;
		dsize *= m_dim[ii];
	}

	m_data = new T[dsize];
}

template <int D, typename T>
void NDArrayStore<D,T>::resize(size_t dim[D])
{
	delete[] m_data;

	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		m_dim[ii] = dim[ii];
		dsize *= m_dim[ii];
	}

	m_data = new T[dsize];
}

template <int D, typename T>
inline
size_t NDArrayStore<D,T>::getAddr(int x, int y, int z, int t, int u, int v, int w)
{
	size_t tmp[D];
	switch(D) {
		case 7:
			tmp[6] = w;
		case 6:
			tmp[5] = v;
		case 5:
			tmp[4] = u;
		case 4:
			tmp[3] = t;
		case 3:
			tmp[2] = z;
		case 2:
			tmp[1] = y;
		case 1:
			tmp[0] = x;
	}
	return getAddr(tmp);
}

template <int D, typename T>
inline
size_t NDArrayStore<D,T>::getAddr(size_t index[D])
{
	size_t loc = index[0];
	size_t jump = m_dim[0];           // jump to global position
	for(size_t ii=1; ii<D; ii++) {
		loc += index[ii]*jump;
		jump *= m_dim[ii];
	}
	return loc;
}

template <int D, typename T>
double NDArrayStore<D,T>::getD(int x, int y, int z, int t, int u, int v, 
		int w)
{
	size_t ii = getAddr(x,y,z,t,u,v,w);
	return (double)m_data[ii];
}

template <int D, typename T>
int NDArrayStore<D,T>::getI(int x, int y, int z, int t, int u, int v, int w)
{
	return (int)m_data[getAddr(x,y,z,t,u,v,w)];
}

template <int D, typename T>
void NDArrayStore<D,T>::setD(double newval, int x, int y, int z, int t, 
		int u, int v, int w)
{
	m_data[getAddr(x,y,z,t,u,v,w)] = (T)newval;
}

template <int D, typename T>
void NDArrayStore<D,T>::setI(int newval, int x, int y, int z, int t, int u, 
		int v, int w)
{
	m_data[getAddr(x,y,z,t,u,v,w)] = (T)newval;
}

template <int D, typename T>
double NDArrayStore<D,T>::operator()(int x, int y, int z, int t, int u, int v, 
		int w)
{
	return (double)m_data[getAddr(x,y,z,t,u,v,w)];
}

template <int D, typename T>
size_t NDArrayStore<D,T>::getBytes()
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= m_dim[ii];
	return out*sizeof(T);
}

template <int D, typename T>
size_t NDArrayStore<D,T>::getNDim() 
{
	return D;
}

template <int D, typename T>
size_t NDArrayStore<D,T>::dim(size_t dir)
{
	return m_dim[dir];
}

template <int D, typename T>
size_t* NDArrayStore<D,T>::dim()
{
	return m_dim;
}



