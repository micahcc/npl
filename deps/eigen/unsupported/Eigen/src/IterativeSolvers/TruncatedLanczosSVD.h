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
     * @brief Given the current desired rank (setDesiredRank), desired total
     * variance (setVarThreshold), and threshold (setThreshold), remove extra
     * singular values that go beyond the stated limit (ie are after the
     * maximum rank, beyond the desired threshold of found variance, or below
     * the minimum threshold for zero).
     *
     * Because the underlying solver may need to run for
     * additional iterations to achieve high accuracy, additional singular
     * values may at times be returned 'at no additional cost'. Calling this
     * function removes any additional singular values that may have been
     * returned.
     *
     * Further, if limits are placed AFTER the compute()
     * function, this can be used to further truncate the singular values
     * matrix. Note that after calling hardenLimits compute() must be called
     * again to re-calculate truncated elements.
     */
    void hardenLimits()
    {
        eigen_assert(m_status == 1 && "SVD Not Complete.");

        int r = rank();
        m_singvals.conservativeResize(r);
        m_U = m_U.leftCols(r);
        m_V = m_V.leftCols(r);
    }

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
    TruncatedLanczosSVD(const Ref<const MatrixType>& A,
            unsigned int computationOptions, bool hardlimits = false)
    {
        init();
        compute(A, computationOptions, hardlimits);
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
    void compute(const Ref<const MatrixType>& A, unsigned int computationOptions,
            bool hardlimits = false)
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

        // update totalvar (sum of variances)
        m_totalvar = C.trace();

        // This is a bit hackish, really the user should set this
        int initrank = 10;
        if(m_initbasis > 0)
            initrank = m_initbasis;

        double evalstop = 1;
        if(!isnan(m_var_thresh) && !isinf(m_var_thresh))
            evalstop = m_var_thresh;

        BandLanczosSelfAdjointEigenSolver<MatrixType> eig;
        eig.setDesiredRank(m_nvec);
        eig.setEValStop(evalstop);
        eig.setDeflationTol(m_deftol);
        eig.compute(C, initrank);

        if(eig.info() != Success) {
            m_status = -2;
            return;
        }

        int rows = eig.eigenvalues().rows();
        int rank = 0;
        for(int rr=rows-1; rr>=0; rr--) {
            if(eig.eigenvalues()[rr] > std::numeric_limits<RealScalar>::epsilon())
                rank++;
            else
                break;
        }

        // Note that because Eigen Solvers usually sort eigenvalues in
        // increasing order but singular value decomposers do decreasing order,
        // we need to reverse the singular value and singular vectors found.
        if(A.rows() > A.cols()) {
            // Computed right singular vlaues (V)
            // A = USV*, U = AVS^-1

            // reverse
            m_V.resize(A.cols(), rank);
            m_singvals.resize(rank);
            for(int cc=0; cc<rank; cc++) {
                m_V.col(cc) = eig.eigenvectors().col(rows-1-cc);
                m_singvals[cc] = std::sqrt(eig.eigenvalues()[rows-1-cc]);
            }

            // Compute U if needed
            if(m_computeU)
                m_U = A*m_V*(m_singvals.cwiseInverse()).asDiagonal();
        } else {
            // Computed left singular values (U)
            // A = USV*, A^T = VSU*, V = A^T U S^-1

            m_U.resize(A.rows(), rank);
            m_singvals.resize(rank);
            for(int cc=0; cc<rank; cc++) {
                m_U.col(cc) = eig.eigenvectors().col(rows-1-cc);
                m_singvals[cc] = std::sqrt(eig.eigenvalues()[rows-1-cc]);
            }

            if(m_computeV)
                m_V = A.transpose()*m_U*(m_singvals.cwiseInverse()).asDiagonal();
        }

        if(hardlimits)
            hardenLimits();
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
     * @brief Smallest value of singular values that is not considered zero for
     * matrix rank.
     *
     * @param threshold Value (default is epsilon)
     */
    void setThreshold(RealScalar threshold)
    {
        eigen_assert(threshold >= 0 && "Threshold must be >= 0");
        m_thresh = threshold;
    }

    /**
     * @brief Set the threshold for smallest singular values to the default
     * value (sqrt(epsilon)). Determines matrix rank, sets the threshold to the
     * square root of machine precision, since the square of singular values
     * are initially computed.
     */
    void setThreshold(Default_t)
    {
        m_thresh = std::sqrt(std::numeric_limits<RealScalar>::epsilon());
    }

    /**
     * @brief Get smallest value of singular values that is not considered zero
     * for matrix rank.
     */
    RealScalar threshold()
    {
        return m_thresh;
    }

    /**
     * @brief The number of desired vectors in output. This is only a
     * guideline, but setting it will aid convergence speed by suppressing
     * early checks for convergence. Fewer components may still be returned if
     * the Kyrlov sequence does not have enough variation (in which case
     * increase lanczos basis). More vectors could be returned if additional
     * iterations were needed to provide sufficent accuracy.
     *
     * @param nvec Rough estimate of desired number of vectors
     */
    void setDesiredRank(int nvec) { m_nvec = nvec; };

    /**
     * @brief Remove limits on stopping. This implicity sets the value at 1
     * least 1 and less than infinity.
     *
     */
    void setDesiredRank(Default_t) { m_nvec = -1; };

    /**
     * @brief Get stopping condition based on rank. Negative values indicate no
     * requirement.
     *
     * @return Current number of dsired singular values
     */
    int desiredRank() { return m_nvec; };

    /**
     * @brief Set the deflation tolerance in the Band Lanczos
     * algorithm, which is 0.05. Deflation testing is done in comparison to
     * initial vector norm, so that deflation occurrs more naturally.
     * Setting this smaller may result in more found eigenvalues values, but it
     * may also result in spurious repeated values. Larger values may result in
     * faster convergence, but this must be less thant 1.
     *
     * @param tol Tolerance
     */
    void setDeflationTol(RealScalar tol) { m_deftol = tol; };

    /**
     * @brief Set deflation tolerance to the default in the Band Lanczos
     * algorithm, which is 0.05. Deflation testing is done in comparison to
     * initial vector norm, so that deflation occurrs more naturally.
     * Setting this smaller may result in more found eigenvalues values, but it
     * may also result in spurious repeated values. Larger values may result in
     * faster convergence, but this must be less thant 1.
     *
     * @param Default_t d
     */
    void setDeflationTol(Default_t)
    {
        m_deftol = 0.05;
    };

    /**
     * @brief Get the current deflation tolerance for BandLanczos
     *
     * @return deflation tolerance
     */
    RealScalar deflationTol() { return m_deftol; }

    /**
     * @brief Allows to prescribe a threshold to be used by Lanczos algorithm
     * to determine when to stop. Unlike in the JacobiSVD this is a ratio of
     * the total singular values that is searched for in the output (found by
     * the trace of the covariance matrix). This also must be set BEFORE
     * compute is called to have an effect.
     *
     * @param threshold Ratio of total variance to account for [0,1] or
     * infinite to let the algorithm run to completion.
     */
    void setVarThreshold(const Scalar& threshold)
    {
        m_var_thresh = threshold;
    };

    /**
     * @brief Resets the default behavorior of rank determination which is to
     * wait until the Krylov subspace is exhausted.
     *
     * @param Eigen::Default_t default, only 1 option
     */
    void setVarThreshold(Default_t)
    {
        m_var_thresh = INFINITY;
    };

    /**
     * @brief Returns the current threshold used for truncating the SVD.
     *
     * @return The current threshold (inf if not set)
     */
    Scalar varThreshold()
    {
        return m_var_thresh;
    }

    /**
     * @brief See rank()
     *
     * @return Number of nonzero singular values.
     */
    int nonzeroSingularValues()
    {
        eigen_assert(m_status == 1 && "SVD Not Complete.");
        return rank();
    };

    /**
     * @brief Returns estimated rank of A. This is the minimum of 1) desired
     * rank, 2) # values > threshold 3) # values on the upper end of the
     * spectrum before the sum of the singular values exceeds the Variance
     * Threshold.
     *
     */
    int rank()
    {
        eigen_assert(m_status == 1 && "SVD Not Complete.");

        int desrank = m_nvec > 0 ? m_nvec : std::numeric_limits<int>::max();
        int varrank = 0;
        int threshrank = 0;
        RealScalar sum = 0;
        for(int cc=0; cc<m_singvals.rows(); cc++) {
            // if m_thresh is invalid (NAN, negative) or the value is greater
            if(!(m_thresh>= 0) || m_singvals[cc]/m_singvals[0] > m_thresh)
                threshrank++;
            // if m_var_thresh is invalid (not in [0,1]) or sum < thresh
            if(!(m_var_thresh<=1&&m_var_thresh>=0) ||
                    m_singvals[cc]*m_singvals[cc] + sum <
                    m_totalvar*m_var_thresh) {
                varrank++;
            }
            sum += m_singvals[cc]*m_singvals[cc];
        }

        return std::min(std::min(varrank, threshrank), desrank);
    };

    /**
     * @brief Size of random basis vectors to build Krylov basis
     * from in Eigenvalue decomposition of MM* or M*M. This should be roughly
     * the number of clustered large eigenvalues. If not enough eigenvalues
     * are being found, it may be worth increasing this
     *
     * @param basis Random Lanczos basis size
     */
    void setLanczosBasis(int basis)
    {
        m_initbasis = basis;
    };

    /**
     * @brief Size of random basis vectors to build Krylov basis basck
     * to default (which is max(10, 1/10 min(rows,cols)).
     *
     */
    void setLanczosBasis(Default_t)
    {
        m_initbasis = -1;
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
        setLanczosBasis(Default);
        setDeflationTol(Default);
        setVarThreshold(Default);
        setThreshold(Default);
        setDesiredRank(Default);

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
    int m_nvec;

    /**
     * @brief Stop after this amount of the variance has been accounted for
     */
    RealScalar m_var_thresh;

    /**
     * @brief Minimum singular value that is considered non-zero for rank
     * computation.
     */
    RealScalar m_thresh;

    /**
    * @brief Deflation tolerance for BandLanczos Algorithm. Larger values will
    * converge faster, default is epsilon
     */
    RealScalar m_deftol;

    /**
    * @brief Whether U will/was computed
     */
    bool m_computeU;

    /**
    * @brief Whether V will/was computed
     */
    bool m_computeV;

    /**
    * @brief Status, where
    *
    * 1:  Computation successful
    *
    * 0:  computation not yet run
    *
    * -1: Invalid Input
    *
    * -2: The algorithm did not converge
    *
    * -3: There were numerical issues
    */
    int m_status;

    /**
    * @brief Total variance (sum of singular values) present in the input, used
    * to determine when the algorithm is nearing compleation
    */
    double m_totalvar;
};

} // Eigen

#endif //EIGEN_TRUNCATED_LANCZOS_SVD_H

