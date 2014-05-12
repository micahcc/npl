#include <cstddef>
#include <cmath>

class NDarray
{
	virtual double getDouble(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0) = 0;

	virtual int getInt(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0) = 0;
	
	virtual void setDouble(double newval, int x = 0, int y = 0, int z = 0, 
			int t = 0, int u = 0, int v = 0, int w = 0) = 0;

	virtual void setInt(int newval, int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0) = 0;

	virtual size_t getBytes() = 0;
	virtual size_t getNDim() = 0;
	virtual void getDim(int* x = NULL, int* y = NULL, int* z = NULL, 
			int* t = NULL, int* u = NULL, int* v = NULL, int* w = NULL) = 0;
};

template <int D, typename T>
class NDArrayStore : public NDarray
{
public:
	// make hyper-cubes to match cache
	NDArrayStore(size_t dim[D], size_t csize = 1024);
	
	// make strides according to the input (0 indicates to use the entire 
	// of a dimension as a stride). Thus stride[3] = {0,100, 0} would make
	// x and z dimensions match the image dimensions
	NDArrayStore(size_t dim[D], size_t strides[D], size_t csize = 1024);
	
	~NDArrayStore() { delete[] m_data; };

	virtual size_t getAddr(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);
	virtual size_t getAddr(size_t index[D]);

	virtual double getDouble(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);

	virtual int getInt(int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);
	
	virtual void setDouble(double newval, int x = 0, int y = 0, int z = 0, 
			int t = 0, int u = 0, int v = 0, int w = 0);

	virtual void setInt(int newval, int x = 0, int y = 0, int z = 0, int t = 0, 
			int u = 0, int v = 0, int w = 0);

	virtual size_t getBytes();
	virtual size_t getNDim();
	virtual void getDim(int* x = NULL, int* y = NULL, int* z = NULL, 
			int* t = NULL, int* u = NULL, int* v = NULL, int* w = NULL);
	
	T* m_data;
	size_t m_cstride;   // stride for each cluster
	size_t m_csize[D];  // cluster sizes in each dimension
	size_t m_cdim[D]; 	// number of clusters in each dimension
	size_t m_pdim[D]; 	// dimension including extra padding from clustering
	size_t m_dim[D];	// overall image dimension
};


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
NDArrayStore<D,T>::NDArrayStore(size_t dim[D], size_t csize)
{
	size_t strides = (size_t)std::pow<double>(csize, 1./D);
	m_cstride = 1;
	for(size_t ii=0; ii<D; ii++) {
		m_csize[ii] = strides;
		m_cstride *= m_csize[ii];
	}
	
	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		m_dim[ii] = dim[ii];
		m_cdim[ii] = ceil(dim[ii]/(double)m_csize[ii]);
		m_pdim[ii] = m_csize[ii]*m_cdim[ii];
		dsize *= m_pdim[ii];
	}

	m_data = new T[dsize];
}

/**
 * @brief initializes an array with a desired stride shape (hyper rectangle). 
 * Zero values will be replaced by the dimension
 *
 * @tparam D
 * @tparam T
 * @param dim[D]
 * @param strides[D]
 * @param csize
 */
template <int D, typename T>
NDArrayStore<D,T>::NDArrayStore(size_t dim[D], size_t strides[D], size_t csize)
{
	m_cstride = 1;
	for(size_t ii=0; ii<D; ii++) {
		if(strides[ii] == 0) 
			m_csize[ii] = dim[ii];
		else
			m_csize[ii] = strides[ii];
		m_cstride *= m_csize[ii];
	}
	
	size_t dsize = 1;
	for(size_t ii=0; ii<D; ii++) {
		m_dim[ii] = dim[ii];
		m_cdim[ii] = ceil(dim[ii]/(double)m_csize[ii]);
		m_pdim[ii] = m_csize[ii]*m_cdim[ii];
		dsize *= m_pdim[ii];
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
	size_t cluster = index[0]/m_csize[0];  // which cluster we are in
	size_t pixel = index[0]%m_csize[0];
	size_t cjump = m_cdim[0]; // cluster jump
	size_t ijump = m_csize[0]; // pixel jump, from previous dimensions
	for(size_t ii=1; ii<D; ii++) {
		size_t cluster_ii = index[ii]/m_csize[ii];
		size_t pixel_ii = index[ii]%m_csize[ii];
		cluster += cjump*cluster_ii;
		pixel += ijump*pixel_ii;

		cjump *= m_cdim[ii];
		ijump *= m_pdim[ii];
	}
	return m_cstride*cluster+pixel;
}

template <int D, typename T>
double NDArrayStore<D,T>::getDouble(int x, int y, int z, int t, int u, int v, 
		int w)
{
	size_t ii = getAddr(x,y,z,t,u,v,w);
	return (double)m_data[ii];
}

template <int D, typename T>
int NDArrayStore<D,T>::getInt(int x, int y, int z, int t, int u, int v, int w)
{
	return (int)m_data[getAddr(x,y,z,t,u,v,w)];
}

template <int D, typename T>
void NDArrayStore<D,T>::setDouble(double newval, int x, int y, int z, int t, 
		int u, int v, int w)
{
	m_data[getAddr(x,y,z,t,u,v,w)] = (T)newval;
}

template <int D, typename T>
void NDArrayStore<D,T>::setInt(int newval, int x, int y, int z, int t, int u, 
		int v, int w)
{
	m_data[getAddr(x,y,z,t,u,v,w)] = (T)newval;
}

template <int D, typename T>
size_t NDArrayStore<D,T>::getBytes()
{
	size_t out = 1;
	for(size_t ii=0; ii<D; ii++)
		out*= m_pdim[ii];
	return out*sizeof(T);
}

template <int D, typename T>
size_t NDArrayStore<D,T>::getNDim() 
{
	return D;
}

template <int D, typename T>
void NDArrayStore<D,T>::getDim(int* x, int* y, int* z, int* t, int* u, int* v,
		int* w)
{
	switch(D) {
		case 7:
			if(w) *w = m_dim[6];
		case 6:
			if(v) *w = m_dim[5];
		case 5:
			if(u) *u = m_dim[4];
		case 4:
			if(t) *t = m_dim[3];
		case 3:
			if(z) *z = m_dim[2];
		case 2:
			if(y) *y = m_dim[1];
		case 1:
			if(x) *x = m_dim[0];
	}
}
