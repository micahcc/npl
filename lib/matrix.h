/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
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
	/**
	 * @brief Returns row in column 0.
	 *
	 * @param row Row to lookup
	 *
	 * @return Value at (row,0)
	 */
	virtual double& operator[](size_t row) = 0;
	
	/**
	 * @brief Returns value at given row/column. 
	 *
	 * @param row Row to lookup
	 * @param col Col to lookup
	 *
	 * @return Value at (row,col)
	 */
	virtual double& operator()(size_t row, size_t col = 0) = 0;
	
	/**
	 * @brief Returns row in column 0.
	 *
	 * @param row Row to lookup
	 *
	 * @return Value at (row,0)
	 */
	virtual const double& operator[](size_t row) const = 0;
	
	/**
	 * @brief Returns value at given row/column. 
	 *
	 * @param row Row to lookup
	 * @param col Col to lookup
	 *
	 * @return Value at (row,col)
	 */
	virtual const double& operator()(size_t row, size_t col = 0) const = 0;

	/**
	 * @brief Performs matrix-vector product of right hand side (rhs) and 
	 * the current matrix, writing the result in out. RHS and OUT are 
	 * cast to the appropriate types (dimensions). Will throw if the dimension
	 * requirements are not met.
	 *
	 * @param rhs Right hand side of matrix-vector product
	 * @param out Output of matrix-vector product
	 */
	virtual void mvproduct(const MatrixP* rhs, MatrixP* out) const = 0;
	
	/**
	 * @brief Performs matrix-vector product of right hand side (rhs) and 
	 * the current matrix, writing the result in out. RHS and OUT are 
	 * cast to the appropriate types (dimensions). Will throw if the dimension
	 * requirements are not met.
	 *
	 * @param rhs Right hand side of matrix-vector product
	 * @param out Output of matrix-vector product
	 */
	virtual void mvproduct(const MatrixP& rhs, MatrixP& out) const = 0;
//	virtual void mvproduct(const std::vector<double>& rhs, 
//			std::vector<double>& out) const = 0;
//	virtual void mvproduct(const std::vector<size_t>& rhs, 
//			std::vector<double>& out) const = 0;

	virtual double det() const = 0;
	virtual double norm() const = 0;
	virtual double sum() const = 0;
	virtual size_t rows() const = 0;
	virtual size_t cols() const = 0;
};

template <int D1, int D2>
class Matrix : public virtual MatrixP
{
public:

	/**
	 * @brief Default constructor, sets the matrix to the identity matrix.
	 */
	Matrix() {
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = (ii==jj);
			}
		}
	};
	

	/**
	 * @brief Constructor, sets the entire matrix to the given constant
	 *
	 * @param v constant to set all elements to 
	 */
	Matrix(double v) {
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = v;
			}
		}
	};
	
	/**
	 * @brief Initialize a matrix from array of length l, made up of an array
	 * at v*. Missing values are assumed to be 0, extra values ignored.
	 *
	 * @param l length of v
	 * @param v vector of data
	 */
	Matrix(size_t l, const double* v) {
		size_t kk=0;
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				if(kk < l)
					data[ii][jj] = v[kk++];
				else
					data[ii][jj] = 0;
			}
		}
	};
	
	/**
	 * @brief Initialize a matrix from array of length l, made up of an array
	 * at v*. Missing values are assumed to be 0, extra values ignored.
	 *
	 * @param l length of v
	 * @param v vector of data
	 */
	Matrix(size_t l, const int64_t* v) {
		size_t kk=0;
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				if(kk < l)
					data[ii][jj] = v[kk++];
				else
					data[ii][jj] = 0;
			}
		}
	};
	
	/**
	 * @brief Initialize matrix with vector, any missing values will be assumed
	 * zero, extra values are ignored.
	 *
	 * @param v Input data vector
	 */
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
	
	/**
	 * @brief Initialize matrix with vector, any missing values will be assumed
	 * zero, extra values are ignored.
	 *
	 * @param v Input data vector
	 */
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

	/**
	 * @brief Initialize matrix with vector, any missing values will be assumed
	 * zero, extra values are ignored.
	 *
	 * @param v Input data vector
	 */
	Matrix(const std::vector<int64_t>& v) {
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

	/**
	 * @brief Copy constructor, copies all the elements of the other vector.
	 *
	 * @param m
	 */
	Matrix(const Matrix& m)
	{
		for(size_t ii=0; ii<D1; ii++) {
			for(size_t jj=0; jj<D2; jj++) {
				data[ii][jj] = m.data[ii][jj];
			}
		}
	}

	/**
	 * @brief Returns row in column 0.
	 *
	 * @param row Row to lookup
	 *
	 * @return Value at (row,0)
	 */
	double& operator[](size_t row) 
	{ 
		assert(row < D1);
		return data[row][0]; 
	};

	/**
	 * @brief Returns value at given row/column. 
	 *
	 * @param row Row to lookup
	 * @param col Col to lookup
	 *
	 * @return Value at (row,col)
	 */
	double& operator()(size_t row, size_t col = 0) 
	{
		assert(row < D1 && col < D2);
		return data[row][col]; 
	};

	/**
	 * @brief Returns row in column 0.
	 *
	 * @param row Row to lookup
	 *
	 * @return Value at (row,0)
	 */
	const double& operator[](size_t row) const
	{ 
		assert(row < D1);
		return data[row][0]; 
	};

	/**
	 * @brief Returns value at given row/column. 
	 *
	 * @param row Row to lookup
	 * @param col Col to lookup
	 *
	 * @return Value at (row,col)
	 */
	const double& operator()(size_t row, size_t col = 0) const
	{
		assert(row < D1 && col < D2);
		return data[row][col]; 
	};


	/**
	 * @brief Performs matrix-vector product of right hand side (rhs) and 
	 * the current matrix, writing the result in out. RHS and OUT are 
	 * cast to the appropriate types (dimensions). Will throw if the dimension
	 * requirements are not met.
	 *
	 * @param rhs Right hand side of matrix-vector product
	 * @param out Output of matrix-vector product
	 */
	void mvproduct(const MatrixP* rhs, MatrixP* out) const;
	
	/**
	 * @brief Performs matrix-vector product of right hand side (rhs) and 
	 * the current matrix, writing the result in out. RHS and OUT are 
	 * cast to the appropriate types (dimensions). Will throw if the dimension
	 * requirements are not met.
	 *
	 * @param rhs Right hand side of matrix-vector product
	 * @param out Output of matrix-vector product
	 */
	void mvproduct(const MatrixP& rhs, MatrixP& out) const;
	
//	void mvproduct(const std::vector<double>& rhs, 
//			std::vector<double>& out) const;
//	void mvproduct(const std::vector<size_t>& rhs, 
//			std::vector<double>& out) const;

	virtual double det() const;
	virtual double norm() const;
	virtual double sum() const;

	size_t rows() const {return D1;};
	size_t cols() const {return D2;};
private:
	double data[D1][D2];
};

//double det(const Matrix<1, 1>& trg);
//double det(const Matrix<2, 2>& trg);
//template <int D1, int D2>
//double det(const Matrix<D1, D2>& trg);
//Matrix<1, 1> inverse(const Matrix<1, 1>& trg);
//Matrix<2, 2> inverse(const Matrix<2, 2>& trg);
//template <int DIM>
//Matrix<DIM, DIM> inverse(const Matrix<DIM, DIM>& trg);
//

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
//
//template <int D1, int D2>
//void Matrix<D1,D2>::mvproduct(const std::vector<double>& iv, 
//		std::vector<double>& ov) const
//{
//	assert(&iv != &ov);
//	assert(iv.size() == D2);
//	ov.resize(D1);
//	const Matrix<D1,D2>& m = *this;
//
//	for(size_t ii=0; ii<D1; ii++) {
//		ov[ii] = 0;
//		for(size_t jj=0; jj<D2; jj++) {
//			ov[ii] += m(ii,jj)*iv[jj];
//		}
//	}
//}
//
//template <int D1, int D2>
//void Matrix<D1,D2>::mvproduct(const std::vector<size_t>& iv, 
//		std::vector<double>& ov) const
//{
//	assert(iv.size() == D2);
//	ov.resize(D1);
//	const Matrix<D1,D2>& m = *this;
//
//	for(size_t ii=0; ii<D1; ii++) {
//		ov[ii] = 0;
//		for(size_t jj=0; jj<D2; jj++) {
//			ov[ii] += m(ii,jj)*iv[jj];
//		}
//	}
//}
//
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
double det(const Matrix<1, 1>& trg)
{
	return trg(0,0);
}

double det(const Matrix<2, 2>& trg)
{
	return trg(0,0)*trg(1,1) - trg(1,0)*trg(0,1);
}

template <int DIM>
double det(const Matrix<DIM, DIM>& trg)
{

	// break up into smaller matrices
	Matrix<DIM/2,DIM/2> A;
	Matrix<DIM/2,(DIM+1)/2> B;
	Matrix<(DIM+1)/2,DIM/2> C;
	Matrix<(DIM+1)/2,(DIM+1)/2> D;
	split(trg, A, B, C, D);
	
	auto AI = inverse(A);
	double a = det(A)*det(D-C*AI*B);

	if(std::isnan(a)) {
		auto DI = inverse(D);
		double b = det(D)*det(A-B*DI*C);
		if(std::isnan(b)) {
			std::cerr << "Determinant failed " << std::endl; 
			return NAN;
		} else
			return b;
	} else 
		return a;
}

template <int D1, int D2>
double norm(const Matrix<D1, D2>& trg)
{
	(void)(trg);
	throw std::length_error("Matrix Passed to Norm Function");
	return 0;
}

template <int DIM>
double norm(const Matrix<DIM, 1>& trg)
{
	double sum = 0;
	for(size_t ii=0; ii<DIM; ii++)
		sum += trg[ii]*trg[ii];
	return sqrt(sum);
}

template <int DIM>
double norm(const Matrix<1, DIM>& trg)
{
	double sum = 0;
	for(size_t ii=0; ii<DIM; ii++)
		sum += trg(0,ii)*trg(0,ii);
	return sqrt(sum);
}

template <int D1, int D2>
double Matrix<D1,D2>::norm() const
{
	return npl::norm<D1,D2>(*this);
}

template <int D1, int D2>
double Matrix<D1,D2>::det() const
{
	const size_t DIM = D1 < D2 ? D1 : D2;
	Matrix<DIM,DIM> tmp;
	for(size_t ii=0;ii<DIM; ii++){
		for(size_t jj=0;jj<DIM; jj++){
			tmp(ii,jj) = data[ii][jj];
		}
	}

	return npl::det<DIM>(tmp);
}

template <int D1, int D2>
double Matrix<D1,D2>::sum() const
{
	double sum = 0;
	for(size_t ii=0; ii<D1; ii++) {
		for(size_t jj=0; jj<D2; jj++) {
			sum+=data[ii][jj];
		}
	}

	return sum;
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
