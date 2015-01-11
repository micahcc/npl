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

        // This is a bit hackish, really the user should set this
        int initrank = std::min<int>(A.rows(), A.cols());
        if(m_initbasis > 0)
            initrank = m_initbasis;

        BandLanczosSelfAdjointEigenSolver<MatrixType> eig;
        eig.setTraceStop(m_trace_thresh);
        eig.setMaxIters(m_maxiters);
        eig.compute(C, initrank);

        if(eig.info() == NoConvergence)
            m_status = -2;

        int eigrows = eig.eigenvalues().rows();
        int rank = 0;
        m_singvals.resize(eigrows);
        double maxev = eig.eigenvalues()[eigrows-1];
        for(int cc=0; cc<eigrows; cc++) {
            if(eig.eigenvalues()[eigrows-1-cc]/maxev < m_sv_thresh)
                m_singvals[cc] = 0;
            else {
                m_singvals[cc] = std::sqrt(eig.eigenvalues()[eigrows-1-cc]);
                rank++;
            }
        }
        m_singvals.conservativeResize(rank);

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

            // Compute U if needed
            if(m_computeU)
                m_U = A*m_V*(m_singvals.cwiseInverse()).asDiagonal();
        } else {
            // Computed left singular values (U)
            // A = USV*, A^T = VSU*, V = A^T U S^-1

            m_U.resize(eig.eigenvectors().rows(), rank);
            for(int cc=0; cc<rank; cc++)
                m_U.col(cc) = eig.eigenvectors().col(eigrows-1-cc);

            if(m_computeV)
                m_V = A.transpose()*m_U*(m_singvals.cwiseInverse()).asDiagonal();
        }

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
     * See setTraceStop()
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
        m_sv_thresh = std::numeric_limits<Scalar>::epsilon();
        m_trace_thresh = INFINITY;
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
     * @brief Maximum number of singular values to compute, if this is <= 0,
     * then the condition is ignored.
     */
    int m_maxiters;

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

