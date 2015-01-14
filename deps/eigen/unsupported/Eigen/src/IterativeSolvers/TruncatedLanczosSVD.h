// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Micah Chambers <micahc.vt@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRUNCATED_LANCZOS_SVD_H
#define EIGEN_TRUNCATED_LANCZOS_SVD_H

#include <Eigen/Eigenvalues>
#include <limits>
#include <vector>
#include <functional>
#include <unordered_map>
#include <cmath>

#include <iostream>
using namespace std;

namespace Eigen {

	// TODO
	// fix interface
	// stopping condition
	// clean up
	// change system to lists in bandlanczos
namespace internal
{

/**
 * @brief Less Function that prioritizes row (row major)
 */
struct RowLess
{
	bool operator()(const std::pair<int,int>& lhs, const std::pair<int,int>& rhs)
	{
		if(lhs.first < rhs.first)
			return true;
		if(lhs.first > rhs.first)
			return false;
		if(lhs.second < rhs.second)
			return true;
		if(lhs.second > rhs.second)
			return false;
		return false;
	};
};

/**
 * @brief A less function that prioritizes col. So iterating will
 * iterate down the col.
 */
struct ColLess
{
	bool operator()(const std::pair<int,int>& lhs, const std::pair<int,int>& rhs)
	{
		if(lhs.second<rhs.second)
			return true;
		if(lhs.second>rhs.second)
			return false;
		if(lhs.first<rhs.first)
			return true;
		if(lhs.first> rhs.first)
			return false;
		return false;
	};
};
}

/**
 * @brief SVD decomposition of input matrix
 *
 * \f[
        A = USV^*
 * \f]
 *
 * Where A is N-by-M, U is N-by-R, V is M-by-R and S is the diagonal matrix of
 * singular values (R-by-R). Both U and V are unitary (inverse = transpose).
 *
 * TODO: documentation with usage examples
 * TODO: documentation of settings
 * TODO: Test complex Scalars
 */
template <typename _MatrixType>
class TruncatedLanczosSVD
{
  public:

    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef typename MatrixType::Index Index;
    enum {
        RowsAtCompileTime = MatrixType::RowsAtCompileTime,
        ColsAtCompileTime = MatrixType::ColsAtCompileTime,
        DiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_DYNAMIC(
                RowsAtCompileTime,ColsAtCompileTime),
        MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
        MaxDiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(
                MaxRowsAtCompileTime,MaxColsAtCompileTime),
        MatrixOptions = MatrixType::Options
    };

    typedef Matrix<Scalar, RowsAtCompileTime, 1, MatrixOptions,
            MaxRowsAtCompileTime, 1> VectorType;
    typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, MatrixOptions,
            MaxRowsAtCompileTime, MaxDiagSizeAtCompileTime> MatrixUType;
    typedef Matrix<Scalar, ColsAtCompileTime, Dynamic, MatrixOptions,
            MaxColsAtCompileTime, MaxDiagSizeAtCompileTime> MatrixVType;
    typedef Matrix<RealScalar, Dynamic, 1, MatrixOptions,
            MaxDiagSizeAtCompileTime> SingularValuesType;

    // Technically this scalar should be the merging of Scalar and b::Scalar
    typedef Matrix<Scalar, RowsAtCompileTime, 1, MatrixOptions,
            MaxRowsAtCompileTime, MaxRowsAtCompileTime> SolveReturnType;

    // Store M*M or MM*
    typedef Matrix<Scalar, Dynamic, Dynamic, MatrixOptions,
            MaxDiagSizeAtCompileTime, MaxDiagSizeAtCompileTime> WorkMatrixType;

    /**
     * @brief Default constructor
     */
    TruncatedLanczosSVD()
    {
        init();
    }

    /**
     * @brief Construct and
     *
     * @param A Input Matrix to decompose
     * @param computationOptions Logical OR of computation options. Unlike
     * JacobiSVD, there is no difference between ComputeThinU and ComputeFullU
     * since the size of U is determined by the convergence specification. The
     * same goes for ComputeThinV/ComputeFullV
     */
    TruncatedLanczosSVD(const Ref<const MatrixType>& A, unsigned int computationOptions)
    {
        init();
        compute(A, computationOptions);
    };

	int update_TV(const Ref<const MatrixType>& A, int initrank)
	{
		// m_maxiters - number of times we are allowed to recompute in case of
		// bad convergence
		// m_maxsize - maximum size of T, if -1 then no restriction
		// m_trace_stop - stop after sum of eigenvalues reaches this percent of
		// the total trace, values outside [0,1] indicate 1
		// m_tracesq_stop - stop after sum of eigenvalues reaches this percent of
		// the total trace, values outside [0,1] indicate 1

		std::list<VectorType> V; // Store Lanczos Vectors
		std::list<VectorType> P; // Store P vectors (eigenvectors)

		// Radius of Band/Current Candidate Vectors
		int bandrad = initrank;
		int initkept = initrank; // number of non-deflated original vectors

		std::list<int> deflated; // list of deflated indexes, matching Vdef
		std::list<VectorType> Vdef; // list of deflated vectors (Vdef)
		RealScalar Anorm = 1; // Start at 1

		std::map<std::pair<int,int>, Scalar, internal::RowLess> U;
		std::map<std::pair<int,int>, Scalar, internal::ColLess> rho;

		// Diagonal Matrix
		std::vector<Scalar> Delta;

		// Iterator Matching nn
		typename std::list<VectorType>::iterator vn_it = V.begin();
		typename std::list<VectorType>::iterator vj_it;
		typename std::list<VectorType>::iterator pj_it; // iterate over eigenvectors
		std::list<int>::iterator def_it; // iterate over deflited indexes

		int nn;
		for(nn=0; (m_maxiters < 0 || nn<m_maxiters) ; nn++, ++vn_it) {

			// (3) compute ||v_j||
			double vnorm = vn_it->norm();

			cerr <<nn<<"/"<<bandrad<<"/"<<vnorm<<"/"<<m_deftol*Anorm<<endl;
			/*****************************************************************
			 * Perform Deflation until we find a large enough column of V
			 ******************************************************************/
			while(!(vnorm > m_deftol*Anorm)) {
				// If Vj was NOT one of the original candidates, add to the
				// most recent Krylov column to the list of deflated, IE if we
				// are at A^2R_2 then deflate A^1R_2
				if(nn >= bandrad) {
					deflated.push_back(nn-bandrad);

					// Move V_{n} from V to V_{def}
					vj_it = vn_it;
					vn_it++;
					Vdef.splice(Vdef.end(), V, vj_it);
				} else {
					vn_it = V.erase(vn_it);
				}

				bandrad--;
				if(bandrad == 0) {
					// TODO will this have the right size?
					assert(vn_it == V.end());
					break;
				}
				vnorm = vn_it->norm();
			}
			if(bandrad == 0) break;

			// Normalize V_n
			*vn_it /= vnorm;

			if(nn >= bandrad)
				U[{nn-bandrad, nn}] = vnorm/Delta[nn-bandrad];
			else
				rho[{nn,nn-bandrad+initrank}] = vnorm;

			// Orthogonalize the vectors v_{n+j}, 1<=j<mc against vn
			// (orthogonalize all the candidate vectors)
			vj_it = vn_it; vj_it++;
			for(int jj=nn+1; vj_it != V.end(); jj++, ++vj_it) {
				double tau = vn_it->dot(*vj_it);
				*vj_it -= tau*(*vn_it);

				if(jj >= bandrad)
					U[{jj-bandrad, nn}] = tau/Delta[jj-bandrad];
				else
					rho[{nn, jj+initrank-bandrad}] = tau;
			}

			// Update the Spiked Part of U_n
			for(vj_it=Vdef.begin(), def_it=deflated.begin(); vj_it!=Vdef.end();
					++def_it, ++vj_it) {
				int jj = *def_it;
				double tau = vn_it->dot(*vj_it)/Delta[jj];;
				U[{jj, nn}] = tau;
			}

			// Compute the vector P_n
			P.push_back(*vn_it);

			// subtract p_j u_jn for j in I
			pj_it = P.begin();
			int jj = 0;
			for(def_it=deflated.begin(); def_it!=deflated.end(); ++def_it) {
				// Find j p_j, for j \in I
				while(pj_it != P.end() && jj < *def_it) {
					++jj;
					++pj_it;
				}

				P.back() -= (*pj_it)*U[{jj,nn}];
			}

			// subtract p_j u_jn for j in previous mc, start at nn-1
			pj_it = P.end(); --pj_it; --pj_it;
			for(jj=nn-1; jj >= nn-bandrad && jj >= 0; jj--, --pj_it)
				P.back() -= (*pj_it)*U[{jj, nn}];
			U[{nn, nn}] = 1;

			// Advance the Krylov Subspace
			V.push_back(A*P.back());
			Delta.push_back(P.back().dot(V.back()));

			if(Delta.back() == 0) {
				break; // TODO look ahead
			}

			V.back() -= Delta.back()*(*vn_it);

			// If this is the last in the starting block,
			if(nn+1 == bandrad) {
				initkept = bandrad; // number of non-deflated starting vectors

				// Compute estimate of Anorm
				Anorm = 0;
				for(pj_it = P.begin(); pj_it != P.end(); ++pj_it) {
					Anorm = std::max(Anorm, (A*(*pj_it)).norm()/pj_it->norm());
				}
			}

		}

		MatrixType upper(nn, nn);
		upper.setZero();
		for(auto it=U.begin(); it != U.end(); it++) {
			int r = it->first.first;
			int c = it->first.second;
			assert(r >= 0 && r < nn);
			assert(c >= 0 && c < nn);
			upper(r, c) = it->second;
		}

//		cerr << "U Matrix:\n" << upper << endl;
//
		VectorType dvec(nn);
		for(size_t ii=0; ii<nn; ii++)
			dvec[ii] = Delta[ii];
//		cerr << "Delta Vector:\n"<<dvec<<endl;
//
		MatrixType T = upper.transpose()*dvec.asDiagonal()*upper;
		cerr<<"T Matrix:\n"<<T<<endl;

//		pj_it = P.begin();
//		MatrixType pmatrix(A.rows(), nn);
//		for(size_t ii=0; ii<nn; ii++, ++pj_it) {
//			assert(pj_it != P.end());
//			double nrm = pj_it->squaredNorm();
//			pmatrix.col(ii) = *pj_it/std::sqrt(nrm);
//			dvec[ii] *= nrm;
//		}
//
//		cerr<<"P Matirx:\n"<<pmatrix<<endl;
//		cerr<<"dvec Matirx:\n"<<dvec<<endl;
//
		cerr << "Eigenvalues of T:"<<endl;
		eig.compute(T);
		cerr<<"EigenValues:\n"<<eig.eigenvalues().transpose()<<endl;

		MatrixType Vmatrix(A.rows(), T.cols());
		auto vit = V.begin();
		for(size_t cc=0; cc<T.cols(); cc++, ++vit)
			Vmatrix.col(cc) = *vit;

		MatrixType eigenvectors = Vmatrix*eig.eigenvectors();;
		cerr<<"EigenVectors:\n"<<eigenvectors<<endl;

		cerr<<"Full Matrix Eigenvectors:\n"<<endl;
		cerr<<"Full Matrix:\n"<<A<<endl;
		eig.compute(A);
		cerr<<"EigenValues:\n"<<eig.eigenvalues()<<endl;
		cerr<<"EigenVectors:\n"<<eig.eigenvectors()<<endl;

//		// compute order of delta
//		std::vector<int> order(Delta.size());
//		for(int ii=0; ii<order.size(); ii++)
//			order[ii] = ii;
//		std::sort(order.begin(), order.end(),
//				[&Delta](int lhs, int rhs) { return Delta[lhs] < Delta[rhs]; });
//		m_singvals.resize(Delta.size());
//
//		if(save_V) {
//			matrixV.resize(A.rows(), P.size());
//			int ii = 0;
//			pj_it = P.begin();
//			typename std::vector<Scalar>::iterator sit = Delta.begin();
//			for(size_t kk=0; kk<order.size(); kk++) {
//				// iterate to order's location
//				advance(pj_it, order[kk]-ii);
//				advance(sit, order[kk]-ii);
//				matrixV.col(kk) = *pj_it;
//				m_singvals[kk] = *sit;
//			}
//		} else {
//			matrixU.resize(A.cols(), P.size());
//			pj_it = P.begin();
//			int ii = 0;
//			typename std::vector<Scalar>::iterator sit = Delta.begin();
//			for(size_t kk=0; kk<order.size(); kk++) {
//				// iterate to order's location
//				advance(pj_it, order[kk]-ii);
//				advance(sit, order[kk]-ii);
//				matrixU.col(kk) = *pj_it;
//				m_singvals[kk] = *sit;
//			}
//		}

	}
    /**
     * @brief Band Lanczos Methof for Hessian Positive Definite Matrices
     *
	 * Input A, R
	 *
     * @param A Matrix to decompose
	 * @param initrank initial rank to start with
	 * @param save_V save the eigenvectors in m_V (if false then saves in m_U)
     */
    void _compute_ev(const Ref<const MatrixType>& A, int initrank, bool save_V)
	{

		bool mtm = false;
		int psize = 0;
		if(A.rows() >= A.cols()) {
			mtm = true;
			psize = A.cols();
		} else if(A.cols() > A.rows()) {
			mtm = false;
			psize = A.rows();
		}

		MatrixXd V(psize, initrank);
		// Generate Initial Vectors
		for(size_t cc=0; cc<initrank; cc++) {
			V.col(cc).setRandom();
			V.col(cc).normalize();
		}

		MatrixXd T;
		for(int ii=0; ii<m_maxiters; ii++) {
			update_TV(A, mtm, &T, &V);

			// check eigenvalues

		}

        m_status = 1;
    };

    /**
     * @brief Maximum rank to compute (sets max size of T matrix). < 0 will
     * remove all limits.
     */
    int m_maxiters;

    /**
     * @brief Status of computation
     * 0 nothing has happened yet
     * 1 success
     * -1 invalid input
     * -2 non-convergnence
     * -3 Numerical Issue
     */
    int m_status;

    /**
     * @brief Computes the SVD decomposition of A, seeding the underlying
     * eigen solver with a bsis of size initbasis.
     *
     * @param A Input Matrix to decompose
     * @param computationOptions Logical OR of computation options. Unlike
     * JacobiSVD, there is no difference between ComputeThinU and ComputeFullU
     * since the size of U is determined by the convergence specification. The
     * same goes for ComputeThinV/ComputeFullV
     *
     * return 0 if successful, -1 if there is a failure (may need to have more
     * basis vectors)
     */
    void compute(const Ref<const MatrixType>& A, unsigned int computationOptions)
    {
        if(computationOptions & (ComputeThinV|ComputeFullV))
            m_computeV = true;
        if(computationOptions & (ComputeThinU|ComputeFullU))
            m_computeU = true;

        WorkMatrixType C;
        if(A.rows() > A.cols()) {
            // Compute right singular vlaues (V)
            C = A.transpose()*A;
            m_computeV = true;
        } else {
            // Computed left singular values (U)
            C = A*A.transpose();
            m_computeU = true;
        }

		// sets
		// m_singvals
        int initrank = std::min<int>(A.rows(), A.cols())/2;
        if(m_initbasis > 0)
            initrank = m_initbasis;
		_compute(C, initrank, A.rows() > A.cols());

		if(m_status < 0)
			return;
//        // This is a bit hackish, really the user should set this
//
//        BandLanczosSelfAdjointEigenSolver<MatrixType> eig;
//        eig.setMaxIters(m_maxiters);
//        eig.setDeflationTol(m_deftol);
//        eig.compute(C, initrank);
//
//        if(eig.info() == NoConvergence)
//            m_status = -2;
//
//        int eigrows = m_singvals.rows();
//        int rank = 0;
//        m_singvals.resize(eigrows);
//        double maxev = m_singvals[eigrows-1];
//        for(int cc=0; cc<eigrows; cc++) {
//            if(m_singvals[eigrows-1-cc]/maxev < m_sv_thresh)
//                m_singvals[cc] = 0;
//            else {
//                m_singvals[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
//                rank++;
//            }
//        }
//        m_singvals.conservativeResize(rank);
//
//        // Note that because Eigen Solvers usually sort eigenvalues in
//        // increasing order but singular value decomposers do decreasing order,
//        // we need to reverse the singular value and singular vectors found.
//        if(A.rows() > A.cols()) {
//            // Computed right singular vlaues (V)
//            // A = USV*, U = AVS^-1
//
//            // reverse
//            m_V.resize(eig.eigenvectors().rows(), rank);
//            for(int cc=0; cc<rank; cc++)
//                m_V.col(cc) = eig.eigenvectors().col(eigrows-1-cc);
//
//            // Compute U if needed
//            if(m_computeU)
//                m_U = A*m_V*(m_singvals.cwiseInverse()).asDiagonal();
//        } else {
//            // Computed left singular values (U)
//            // A = USV*, A^T = VSU*, V = A^T U S^-1
//
//            m_U.resize(eig.eigenvectors().rows(), rank);
//            for(int cc=0; cc<rank; cc++)
//                m_U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);
//
//            if(m_computeV)
//                m_V = A.transpose()*m_U*(m_singvals.cwiseInverse()).asDiagonal();
//        }
//
        m_status = 1;
    };

    /**
     * @brief Return vector of selected singular values
     *
     * @return Vector of singular values
     */
    inline
    const SingularValuesType& singularValues()
    {
        eigen_assert(m_status == 1 && "SVD Not Complete.");
        return m_singvals;
    };

    /**
     * @brief Get matrix U
     *
     * @return matrix U
     */
    inline
    const MatrixUType& matrixU()
    {
        eigen_assert(computeU() && "ComputeThinU not set.");
        eigen_assert(m_status == 1 && "SVD Not Complete.");
        return m_U;
    };

    /**
     * @brief Get matrix V
     *
     * @return matrix V
     */
    inline
    const MatrixVType& matrixV()
    {
        eigen_assert(computeV() && "ComputeThinV not set.");
        eigen_assert(m_status == 1 && "SVD Not Complete.");
        return m_V;
    };

    /**
     * @brief Returns true if the U matrix has been computed
     *
     * @return Whether the U matrix has been computed
     */
    inline
    bool computeU() const { return m_computeU; };

    /**
     * @brief Returns true if the V matrix has been computed
     *
     * @return Whether the V matrix has been computed
     */
    inline
    bool computeV() const { return m_computeV; };

    /**
     * @brief Return number of rows from the last compute operation
     *
     * @return Number of rows in input matrix
     */
    inline Index rows() const { return m_U.rows(); }

    /**
     * @brief Return number of cols from the last compute operation
     *
     * @return Number of cols in input matrix
     */
    inline Index cols() const { return m_V.rows(); }

    /**
     * @brief Return the pseudo-inverse
     *
     * @return Pseudo-inverse matrix
     */
    MatrixType inverse() const
    {
        eigen_assert(m_status == 1 && "SVD is not initialized.");
        eigen_assert(computeU() && computeV() && "SVD::solve() requires both "
                "unitaries U and V to be computed (thin unitaries suffice).");
        return m_V*m_singvals.cwiseInverse()*m_U.transpose();
    }

    /**
     * @brief Solve the equation \f[Ax = b\f] for the given b vector and the
     * approximation: \f[ x = VS^{-1}U^*b \f]
     *
     * @param b Solution to \f$ Ax = b \f$
     *
     * @return Solution
     */
    template <typename Rhs>
    SolveReturnType solve(const Ref<const Rhs>& b) const
    {
        eigen_assert(m_status == 1 && "SVD is not initialized.");
        eigen_assert(computeU() && computeV() && "SVD::solve() requires both "
                "unitaries U and V to be computed (thin unitaries suffice).");
        return m_V*m_singvals.cwiseInverse()*m_U.transpose()*b;
    }

    /**
     * @brief Maximum number of iterations to perform in the
     * BandLanczosSelfAdjointEigenSolver. This is roughly the maximum rank
     * computed, but be wary of setting it too low.
     *
     * @param maxiters Maximum number of iterations in underlying Eigen Solver
     */
    void setMaxIters(int maxiters) { m_maxiters = maxiters; };

    /**
     * @brief Set maximum iterations to infinity, which is triggered by any
     * value less than 0. This constrols the maximum number of iterations to
     * perform in the BandLanczosSelfAdjointEigenSolver.
     *
     * @param maxiters Maximum number of iterations in underlying Eigen Solver
     */
    void setMaxIters(Default_t d) { m_maxiters = -1; };

    /**
     * @brief Get maximum iterations. This constrols the maximum number of
     * iterations to perform in the BandLanczosSelfAdjointEigenSolver.
     */
    int maxIters() { return m_maxiters; };

    /**
     * @brief Set the tolerance for deflation. Algorithm stops when #of
     * deflations hits the number of basis vectors. A higher tolerance will
     * thus cause faster convergence and DOES NOT AFFECT ACCURACY, it may
     * affect the number of found eigenvalues though. Recommended value is
     * sqrt(epsilon), approx 1e-8.
     *
     * @param tol Tolerance
     */
    void setDeflationTol(RealScalar tol) { m_deftol = tol; };

    /**
     * @brief Set the default deflation tolerance, which is sqrt of epsilon.
     *
     * @param Default_t d
     */
    void setDeflationTol(Default_t d)
    {
        m_deftol = sqrt(std::numeric_limits<RealScalar>::epsilon());
    };

    /**
     * @brief Get the current deflation tolerance for BandLanczos
     *
     * @return deflation tolerance
     */
    RealScalar deflationTol() { return m_deftol; }

    /**
     * @brief Allows to prescribe a threshold to be used by Lanczos algorithm
     * to determine when to stop. Unlike in the JacobiSVD this is used during
     * the SVD decomposition, and will affect output of solve() and rank()
     * until compute() is called again. By default the algorithm stops
     * when the Krylov subspace is exhausted.
     *
     * @param threshold Ratio of total variance to account for in underlying
     * eigenvalue problem.
     */
    void setThreshold(const Scalar& threshold)
    {
        m_sv_thresh = threshold;
    };

    /**
     * @brief Resets the default behavorior of rank determination which is to
     * wait until the Krylov subspace is exhausted.
     *
     * @param Eigen::Default_t default, only 1 option
     */
    void setThreshold(Default_t Default)
    {
        m_sv_thresh = std::numeric_limits<Scalar>::epsilon();
    };

    /**
     * @brief Returns the current threshold used for truncating the SVD.
     *
     * @return The current threshold (inf if not set)
     */
    Scalar threshold()
    {
        return m_sv_thresh;
    }

    /**
     * @brief Returns number of singular values that are not exactly zero.
     * Note that since truncation is part of the process, this ends up being
     * the same as the number of rows in the singularValues() vector.
     *
     * @return Number of nonzero singular values.
     */
    int nonzeroSingularValues()
    {
        eigen_assert(m_status == 1 && "SVD Not Complete.");
        return m_singvals.rows();
    };

    /**
     * @brief Returns estimated rank of A.
     */
    int rank()
    {
        eigen_assert(m_status == 1 && "SVD Not Complete.");
        return m_singvals.rows();
    };


    /**
     * @brief Size of random basis vectors to build Krylov basis
     * from in Eigenvalue decomposition of MM* or M*M. This should be roughly
     * the number of clustered large eigenvalues. If results are not
     * sufficiently good, it may be worth increasing this
     *
     * @param basis New random Lanczos basis size
     */
    void setLanczosBasis(int basis)
    {
        m_initbasis = basis;
    };

    /**
     * @brief Size of random basis vectors to build Krylov basis
     * from in Eigenvalue decomposition of MM* or M*M. This should be roughly
     * the number of clustered large eigenvalues. If results are not
     * sufficiently good, it may be worth increasing this
     *
     * @return Current random Lanczos basis size.
     */
    int lanczosBasis()
    {
        return m_initbasis;
    }

    /**
     * @brief Returns information on the results of compute.
     *
     * If you are getting NoConvergence, you should increase m_initbasis
     *
     * @return Success if nothing has been computed yet/computation was
     * successful, or NumericalIssue if the the quality of the results is
     * suspect.
     */
    ComputationInfo info() const
    {
        if(m_status == 0 || m_status == 1) {
            return Success;
        } else if(m_status == -3) {
            return NumericalIssue;
        } else if(m_status == -2) {
            return NoConvergence;
        } else if(m_status == -1) {
            return InvalidInput;
        }

        return InvalidInput;
    };

private:

    void init()
    {
        m_initbasis = -1; // <= 0 -> .1*input rows
        m_sv_thresh = std::numeric_limits<RealScalar>::epsilon();
        m_deftol = std::sqrt(std::numeric_limits<RealScalar>::epsilon());
        m_maxiters = -1;

        m_computeU = false;
        m_computeV = false;

        m_status = 0;
    };

    MatrixVType m_V; // V in USV*
    MatrixUType m_U; // U in USV*
    SingularValuesType m_singvals; // diag(S) in USV*

    /**
     * @brief Number of basis vectors to start BandLanczos Algorithm with
     */
    int m_initbasis;

    /**
     * @brief Stop after this amount of the sum of squared eigenvalues have
     * been found in the MM* or M*M matrix
     */
    RealScalar m_sv_thresh;

    /**
    * @brief Deflation tolerance for BandLanczos Algorithm. Larger values will
    * converge faster, default is epsilon
     */
    RealScalar m_deftol;

    bool m_computeU;
    bool m_computeV;
};

} // Eigen

#endif //EIGEN_TRUNCATED_LANCZOS_SVD_H

