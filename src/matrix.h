#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <memory>

namespace npl { 

class MatrixP
{
public:
	virtual double& operator[](size_t row) = 0;
	virtual double& operator()(size_t row, size_t col = 0) = 0;
	virtual const double& operator[](size_t row) const = 0;
	virtual const double& operator()(size_t row, size_t col = 0) const = 0;

	virtual void mvproduct(const MatrixP* rhs, MatrixP* out) const = 0;
	virtual void mvproduct(const MatrixP& rhs, MatrixP& out) const = 0;
	virtual void mvproduct(const std::vector<double>& rhs, 
			std::vector<double>& out) const = 0;
	virtual void mvproduct(const std::vector<size_t>& rhs, 
			std::vector<double>& out) const = 0;

	virtual size_t rows() const = 0;
	virtual size_t cols() const = 0;
};

template <int D1, int D2>
class Matrix : public virtual MatrixP
{
public:

	// identity
	Matrix() {
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = (ii==jj);
			}
		}
	};
	
	// constant
	Matrix(double v) {
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = v;
			}
		}
	};
	
	// data 
	Matrix(double* v) {
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = *v++;
			}
		}
	};
	
	Matrix(const std::vector<double>& v) {
		size_t kk=0;
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				if(kk < v.size())
					data[ii][jj] = v[kk++];
				else
					data[ii][jj] = 0;
			}
		}
	};
	
	Matrix(const std::vector<size_t>& v) {
		size_t kk=0;
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				if(kk < v.size())
					data[ii][jj] = v[kk++];
				else
					data[ii][jj] = 0;
			}
		}
	};

	Matrix(const Matrix& m)
	{
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = m.data[ii][jj];
			}
		}
	}

	double& operator[](size_t row) 
	{ 
		assert(row < D1);
		return data[row][0]; 
	};

	double& operator()(size_t row, size_t col = 0) 
	{
		assert(row < D1 && col < D2);
		return data[row][col]; 
	};

	const double& operator[](size_t row) const
	{ 
		assert(row < D1);
		return data[row][0]; 
	};

	const double& operator()(size_t row, size_t col = 0) const
	{
		assert(row < D1 && col < D2);
		return data[row][col]; 
	};

	// 
	void mvproduct(const MatrixP* rhs, MatrixP* out) const;
	void mvproduct(const MatrixP& rhs, MatrixP& out) const;
	void mvproduct(const std::vector<double>& rhs, 
			std::vector<double>& out) const;
	void mvproduct(const std::vector<size_t>& rhs, 
			std::vector<double>& out) const;

	size_t rows() const {return D1;};
	size_t cols() const {return D2;};
private:
	double data[D1][D2];
};

template <int D1, int D2>
void Matrix<D1,D2>::mvproduct(const MatrixP* rhs, MatrixP* out) const
{
	assert(rhs != out);
	try{
		const Matrix<D2,1>& iv = dynamic_cast<const Matrix<D2, 1>&>(*rhs);
		Matrix<D1,1>& ov = dynamic_cast<Matrix<D1, 1>&>(*out);
		const Matrix<D1,D2>& m = *this;
		for(size_t ii=0; ii<D1; ii++) {
			ov[ii] = 0;
			for(size_t jj=0; jj<D2; jj++) {
				ov[ii] += m(ii,jj)*iv[jj];
			}
		}
	} catch(...){
		std::cerr << "Error, failed dynamic_cast for matrix-vector product"
			" check the input to mvproduct matrix.h:" << __LINE__ << std::endl;
		throw;
	}
}

template <int D1, int D2>
void Matrix<D1,D2>::mvproduct(const MatrixP& rhs, MatrixP& out) const
{
	assert(&rhs != &out);
	try{
		const Matrix<D2,1>& iv = dynamic_cast<const Matrix<D2, 1>&>(rhs);
		Matrix<D1,1>& ov = dynamic_cast<Matrix<D1, 1>&>(out);
		const Matrix<D1,D2>& m = *this;
		for(size_t ii=0; ii<D1; ii++) {
			ov[ii] = 0;
			for(size_t jj=0; jj<D2; jj++) {
				ov[ii] += m(ii,jj)*iv[jj];
			}
		}
	} catch(...){
		std::cerr << "Error, failed dynamic_cast for matrix-vector product"
			" check the input to mvproduct matrix.h:" << __LINE__ << std::endl;
		throw;
	}
}

template <int D1, int D2>
void Matrix<D1,D2>::mvproduct(const std::vector<double>& iv, 
		std::vector<double>& ov) const
{
	assert(&iv != &ov);
	assert(iv.size() == D2);
	ov.resize(D1);
	const Matrix<D1,D2>& m = *this;

	for(size_t ii=0; ii<D1; ii++) {
		ov[ii] = 0;
		for(size_t jj=0; jj<D2; jj++) {
			ov[ii] += m(ii,jj)*iv[jj];
		}
	}
}

template <int D1, int D2>
void Matrix<D1,D2>::mvproduct(const std::vector<size_t>& iv, 
		std::vector<double>& ov) const
{
	assert(iv.size() == D2);
	ov.resize(D1);
	const Matrix<D1,D2>& m = *this;

	for(size_t ii=0; ii<D1; ii++) {
		ov[ii] = 0;
		for(size_t jj=0; jj<D2; jj++) {
			ov[ii] += m(ii,jj)*iv[jj];
		}
	}
}

template <int D1, int D2>
std::ostream& operator<<(std::ostream& os, const Matrix<D1,D2>& b)
{
	for(size_t rr=0; rr<b.rows(); rr++) {
		os << "[ ";
		for(size_t cc=0; cc<b.cols(); cc++) {
			os << std::setw(11) << std::setprecision(5) << b(rr,cc);
		}
		os << " ];" << std::endl;
	}
	os << std::endl;
	return os;
}

std::ostream& operator<<(std::ostream& os, const MatrixP& b)
{
	for(size_t rr=0; rr<b.rows(); rr++) {
		os << "[ ";
		for(size_t cc=0; cc<b.cols(); cc++) {
			os << std::setw(11) << std::setprecision(5) << b(rr,cc);
		}
		os << " ];" << std::endl;
	}
	os << std::endl;
	return os;
}


template <int D1, int D2>
Matrix<D1, D2> operator+=(Matrix<D1, D2>& lhs, const Matrix<D1, D2>& rhs)
{
	for(size_t ii=0; ii<D1; ii++) {
		for(size_t jj=0; jj<D2; jj++) {
			lhs(ii,jj) += rhs(ii,jj);
		}
	}
	return lhs;
}

template <int D1, int D2>
Matrix<D1, D2> operator+(const Matrix<D1, D2>& lhs, const Matrix<D1, D2>& rhs)
{
	Matrix<D1, D2> out(0.0);
	for(size_t ii=0; ii<D1; ii++) {
		for(size_t jj=0; jj<D2; jj++) {
			out(ii,jj) = lhs(ii,jj)+rhs(ii,jj);
		}
	}
	return out;
}

template <int D1, int D2>
Matrix<D1, D2> operator-(const Matrix<D1, D2>& lhs, const Matrix<D1, D2>& rhs)
{
	Matrix<D1, D2> out(0.0);
	for(size_t ii=0; ii<D1; ii++) {
		for(size_t jj=0; jj<D2; jj++) {
			out(ii,jj) = lhs(ii,jj)-rhs(ii,jj);
		}
	}
	return out;
}

template <int D1, int D2>
Matrix<D1, D2> operator-(const Matrix<D1, D2>& rhs)
{
	Matrix<D1, D2> out(0.0);
	for(size_t ii=0; ii<D1; ii++) {
		for(size_t jj=0; jj<D2; jj++) {
			out(ii,jj) = -rhs(ii,jj);
		}
	}
	return out;
}

template <int D1, int D2, int D3>
Matrix<D1, D3> operator*(const Matrix<D1, D2>& lhs, const Matrix<D2, D3>& rhs)
{
	Matrix<D1, D3> out(0.0);
	for(size_t ii=0; ii<D1; ii++) {
		for(size_t jj=0; jj<D3; jj++) {
			out(ii,jj) = 0;
			for(size_t kk=0; kk<D2; kk++) {
				out(ii,jj) += lhs(ii, kk)*rhs(kk, jj);
			}
		}
	}
	return out;
}

/**
 * @brief Join together 4 matrices from a blockwise decomposition
 *
 * @tparam D1
 * @tparam D2
 * @param tl
 * @param tr
 * @param bl
 * @param br
 *
 * @return 
 */
template <int D1, int D2>
Matrix<D1+D2,D1+D2> join(const Matrix<D1,D1>& tl, const Matrix<D1, D2>& tr, 
		const Matrix<D2, D1>& bl, const Matrix<D2,D2>& br)
{
	Matrix<D1+D2, D1+D2> out;

	// top left
	for(size_t rr=0; rr<D1; rr++) {
		for(size_t cc=0; cc<D1; cc++) {
			out(rr,cc) = tl(rr,cc);
		}
	}
	
	// top right
	for(size_t rr=0; rr<D1; rr++) {
		for(size_t cc=0; cc<D2; cc++) {
			out(rr,cc+D1) = tr(rr,cc);
		}
	}
	
	// bottom left 
	for(size_t rr=0; rr<D2; rr++) {
		for(size_t cc=0; cc<D1; cc++) {
			out(rr+D1,cc) = bl(rr,cc);
		}
	}
	
	// bottom right 
	for(size_t rr=0; rr<D2; rr++) {
		for(size_t cc=0; cc<D2; cc++) {
			out(rr+D1,cc+D1) = br(rr,cc);
		}
	}

	return out;
}

template <int D1, int D2>
void split(const Matrix<D1+D2, D1+D2>& input,
		Matrix<D1,D1>& tl, Matrix<D1, D2>& tr, 
		Matrix<D2, D1>& bl, Matrix<D2,D2>& br)
{
	// top left
	for(size_t rr=0; rr<D1; rr++) {
		for(size_t cc=0; cc<D1; cc++) {
			tl(rr,cc) = input(rr,cc);
		}
	}
	
	// top right
	for(size_t rr=0; rr<D1; rr++) {
		for(size_t cc=0; cc<D2; cc++) {
			tr(rr,cc)= input(rr, cc+D1);
		}
	}
	
	// bottom left 
	for(size_t rr=0; rr<D2; rr++) {
		for(size_t cc=0; cc<D1; cc++) {
			bl(rr,cc) = input(rr+D1,cc);
		}
	}
	
	// bottom right 
	for(size_t rr=0; rr<D2; rr++) {
		for(size_t cc=0; cc<D2; cc++) {
			br(rr,cc) = input(rr+D1,cc+D1);
		}
	}
}

// Determinant //

double determinant(const Matrix<1, 1>& trg)
{
	return trg(0,0);
}

double determinant(const Matrix<2, 2>& trg)
{
	return trg(0,0)*trg(1,1) - trg(1,0)*trg(0,1);
}

template <int DIM>
double determinant(const Matrix<DIM, DIM>& trg)
{

	// break up into smaller matrices
	Matrix<DIM/2,DIM/2> A;
	Matrix<DIM/2,(DIM+1)/2> B;
	Matrix<(DIM+1)/2,DIM/2> C;
	Matrix<(DIM+1)/2,(DIM+1)/2> D;
	split(trg, A, B, C, D);
	
	auto AI = inverse(A);
	double a = determinant(A)*determinant(D-C*AI*B);

	if(std::isnan(a)) {
		auto DI = inverse(D);
		double b = determinant(D)*determinant(A-B*DI*C);
		if(std::isnan(b)) {
			std::cerr << "Determinant failed " << std::endl; 
			return NAN;
		} else
			return b;
	} else 
		return a;
}

Matrix<1, 1> inverse(const Matrix<1, 1>& trg)
{
	return Matrix<1,1>(1./trg(0,0));
}

Matrix<2, 2> inverse(const Matrix<2, 2>& trg)
{
	Matrix<2,2> tmp;
	double det = trg(0,0)*trg(1,1)-trg(1,0)*trg(0,1);
	if(det == 0) {
		std::cerr << "Error non-invertible matrix!" << std::endl;
		return Matrix<2,2>();
	}

	tmp(0,0) = trg(1,1)/det;
	tmp(1,1) = trg(0,0)/det;
	tmp(1,0) = -trg(1,0)/det;
	tmp(0,1) = -trg(0,1)/det;
	return tmp;
}

template <int DIM>
Matrix<DIM, DIM> inverse(const Matrix<DIM, DIM>& trg)
{

	// break up into smaller matrices
	Matrix<DIM/2,DIM/2> A;
	Matrix<DIM/2,(DIM+1)/2> B;
	Matrix<(DIM+1)/2,DIM/2> C;
	Matrix<(DIM+1)/2,(DIM+1)/2> D;
	split(trg, A, B, C, D);
	
	auto AI = inverse(A);
	auto betaI = inverse(D-C*AI*B);

	auto tl = AI + AI*B*betaI*C*AI;
	auto tr = -AI*B*betaI;
	auto bl = -betaI*C*AI;
	auto br = betaI;
	
	auto ret = join(tl, tr, bl, br);

	return ret;
}

} // npl

#endif // MATRIX_H
