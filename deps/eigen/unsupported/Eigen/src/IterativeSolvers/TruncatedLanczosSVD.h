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
#include <cmath>

#ifdef VERYVERBOSE
#include <iostream>
using std::endl;
using std::cerr;
#endif //VERYVERBOSE

namespace Eigen {

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
 * TODO: template over Matrix Type
 * TODO: Fix for complex Scalars
 */
template <typename _Scalar>
class TruncatedLanczosSVD
{
public:
    typedef _Scalar Scalar;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

    /**
     * @brief Default constructor
     */
    TruncatedLanczosSVD()
    {
        init();
    };

    /**
     * @brief Construct and
     *
     * @param A Input Matrix to decompose
     * @param computationOptions Logical OR of computation options. Unlike
     * JacobiSVD, there is no difference between ComputeThinU and ComputeFullU
     * since the size of U is determined by the convergence specification. The
     * same goes for ComputeThinV/ComputeFullV
     */
    TruncatedLanczosSVD(const MatrixType& A, unsigned int computationOptions)
    {
        init();
        compute(A, computationOptions);
    };

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
    void compute(const MatrixType& A, unsigned int computationOptions)
    {
        m_status = 1;
        if(computationOptions & (ComputeThinV|ComputeFullV))
            m_computeV = true;
        if(computationOptions & (ComputeThinU|ComputeFullU))
            m_computeU = true;

        MatrixType C;
        if(A.rows() > A.cols()) {
            // Compute right singular vlaues (V)
            C = A.transpose()*A;
            m_computeV = true;
#ifdef VERYVERBOSE
            cerr << "Computed A^T*A" << endl;
            cerr << endl << C << endl << endl;
#endif //VERYVERBOSE
        } else {
            // Computed left singular values (U)
            C = A*A.transpose();
            m_computeU = true;
#ifdef VERYVERBOSE
            cerr << "Computed A*A^T" << endl;
            cerr << endl << C << endl << endl;
#endif //VERYVERBOSE
        }

        // This is a bit hackish, really the user should set this
        int initrank = std::max<int>(A.rows(), A.cols());
        if(m_initbasis > 0)
            initrank = m_initbasis;

        BandLanczosSelfAdjointEigenSolver<Scalar> eig;
        eig.setTraceStop(m_trace_thresh);
        eig.setRank(m_maxrank);
        eig.compute(C, initrank);

        if(eig.info() == NoConvergence)
            m_status = -2;

#ifdef VERYVERBOSE
        cerr << "Eigenvalues: " << eig.eigenvalues().transpose() << endl;
        cerr << "EigenVectors: " << endl << eig.eigenvectors() << endl << endl;
#endif //VERYVERBOSE
        int eigrows = eig.eigenvalues().rows();
        int rank = 0;
        m_singvals.resize(eigrows);
        for(int cc=0; cc<eigrows; cc++) {
            if(eig.eigenvalues()[eigrows-1-cc] < m_sv_thresh)
                m_singvals[cc] = 0;
            else {
                m_singvals[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
                rank++;
            }
        }

#ifdef VERYVERBOSE
        for(int ee=0; ee<rank; ee++) {
            double err = (eig.eigenvalues()[ee]*eig.eigenvectors().col(ee) -
                    C*eig.eigenvectors().col(ee)).squaredNorm();
            cerr << "Error = " << err << endl;
        }
#endif //VERYVERBOSE

        m_singvals.conservativeResize(rank);
#ifdef VERYVERBOSE
        cerr << "Singular Values: " << m_singvals.transpose() << endl;
#endif //VERYVERBOSE

        // Note that because Eigen Solvers usually sort eigenvalues in
        // increasing order but singular value decomposers do decreasing order,
        // we need to reverse the singular value and singular vectors found.
        if(A.rows() > A.cols()) {
            // Computed right singular vlaues (V)
            // A = USV*, U = AVS^-1

            // reverse
            m_V.resize(eig.eigenvectors().rows(), rank);
            for(int cc=0; cc<rank; cc++)
                m_V.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

#ifdef VERYVERBOSE
            cerr << "V Matrix: " << endl << m_V << endl << endl;
#endif //VERYVERBOSE

            // Compute U if needed
            if(m_computeU)
                m_U = A*m_V*(m_singvals.cwiseInverse()).asDiagonal();
        } else {
            // Computed left singular values (U)
            // A = USV*, A^T = VSU*, V = A^T U S^-1

            m_U.resize(eig.eigenvectors().rows(), rank);
            for(int cc=0; cc<rank; cc++)
                m_U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

#ifdef VERYVERBOSE
            cerr << "U Matrix: " << endl << m_U << endl << endl;
#endif //VERYVERBOSE

            if(m_computeV)
                m_V = A.transpose()*m_U*(m_singvals.cwiseInverse()).asDiagonal();
        }
    };

    /**
     * @brief Return vector of selected singular values
     *
     * @return Vector of singular values
     */
    const VectorType& singularValues()
    {
        eigen_assert(m_status == 1);
        return m_singvals;
    };

    /**
     * @brief Get matrix U
     *
     * @return matrix U
     */
    const MatrixType& matrixU()
    {
        eigen_assert(m_status == 1);
        return m_U;
    };

    /**
     * @brief Get matrix V
     *
     * @return matrix V
     */
    const MatrixType& matrixV()
    {
        eigen_assert(m_status == 1);
        return m_V;
    };

    /**
     * @brief Returns true if the U matrix has been computed
     *
     * @return Whether the U matrix has been computed
     */
    bool computeU() const { return m_computeU; };

    /**
     * @brief Returns true if the V matrix has been computed
     *
     * @return Whether the V matrix has been computed
     */
    bool computeV() const { return m_computeV; };

    /**
     * @brief Solve the equation \f[Ax = b\f] for the given b vector and the
     * approximation: \f[ x = VS^{-1}U^*b \f]
     *
     * @param b Solution to \f$ Ax = b \f$
     *
     * @return Solution
     */
    VectorType solve(const VectorType& b) const
    {
        eigen_assert(m_status == 1);
        return m_V*m_singvals.cwiseInverse()*m_U.transpose()*b;
    };

    /**
     * @brief Stop band lanczos algorithm after the trace of the estimated
     * covariance matrix exceeds the ratio of total sum of 
     * eigenvalues. This is passed directly to the underlying
     * BandLanczosSelfAdjointEigenSolver
     *
	 * @param stop Ratio of variance to account for (0 to 1) with 1 stopping
	 * when ALL the variance has been found and 0 stopping immediately. Set to
	 * INFINITY or NAN to only stop naturally (when the Kyrlov Subspace has
	 * been exhausted).
     */
    void setTraceStop(double stop) { m_trace_thresh = stop; };

    /**
     * @brief Return trace squared stopping condition to default.
     *
     * @param Default_t default
     */
    void setTraceStop(Default_t d) { m_trace_thresh = INFINITY; };

    /**
     * @brief Get stop parameter based on sum of squared eigenvalues.
     * See setTraceSqrStop()
     *
     * @return Get the current stopping condition based on the sum squared
     * eigenvalues (trace squared)
     */
    double traceStop() { return m_trace_thresh; };

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
        eigen_assert(m_status == 1);
        return m_singvals.rows();
    };

    /**
     * @brief Returns estimated rank of A.
     */
    int rank()
    {
        eigen_assert(m_status == 1);
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
        m_sv_thresh = std::numeric_limits<Scalar>::epsilon();
        m_trace_thresh = INFINITY;
        m_maxrank = -1;

        m_computeU = false;
        m_computeV = false;

        m_status = 0;
    };

    MatrixType m_C; // MM* or M*M
    MatrixType m_V; // V in USV*
    MatrixType m_U; // U in USV*
    VectorType m_singvals; // diag(S) in USV*

    /**
     * @brief Number of basis vectors to start BandLanczos Algorithm with
     */
    int m_initbasis;

    /**
     * @brief Maximum number of singular values to compute, if this is <= 0,
     * then the condition is ignored.
     */
    int m_maxrank;

    /**
     * @brief Stop after this amount of the sum of squared eigenvalues have
     * been found in the MM* or M*M matrix
     */
    double m_sv_thresh;

    /**
     * @brief Stop the BandLanczosSelfAdjointEigenSolver after the given
     * percent of squared eigenvalues have been found
     */
    double m_trace_thresh;

    bool m_computeU;
    bool m_computeV;

    int m_status;
};

} // Eigen

#endif //EIGEN_TRUNCATED_LANCZOS_SVD_H

