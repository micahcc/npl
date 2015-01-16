// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Micah Chambers <micahc.vt@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BAND_LANCZOS_SELF_ADJOINT_EIGEN_SOLVER_H
#define EIGEN_BAND_LANCZOS_SELF_ADJOINT_EIGEN_SOLVER_H

#include <Eigen/Eigenvalues>
#include <limits>
#include <cmath>

#include <iostream>
using std::cerr;
using std::endl;

namespace Eigen {


/**
 * @brief Solves eigenvalues and eigenvectors of a hermitian matrix. Currently
 * it is only really symmetric, Values need to made generic enough for complex
 *
 * TODO: documentation with usage examples
 * TODO: documentation of settings
 * TODO: Test complex scalars
 *
 */
template <typename _MatrixType>
class BandLanczosSelfAdjointEigenSolver
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

    // Vector Type
    typedef Matrix<Scalar, RowsAtCompileTime, 1, MatrixOptions,
            MaxRowsAtCompileTime, 1> VectorType;

    // Similar Matrix Type
    typedef Matrix<Scalar, Dynamic, Dynamic, MatrixOptions,
            MaxRowsAtCompileTime, MaxDiagSizeAtCompileTime> ApproxType;

    // EigenVectors Type
    typedef Matrix<Scalar, RowsAtCompileTime, Dynamic, MatrixOptions,
            MaxRowsAtCompileTime, MaxDiagSizeAtCompileTime> EigenVectorType;

    // EigenValus Type
    typedef Matrix<Scalar, Dynamic, 1, MatrixOptions,
            MaxDiagSizeAtCompileTime> EigenValuesType;


    /**
     * @brief Return vector of selected eigenvalues
     *
     * @return Eigenvalues as vector
     */
    const EigenValuesType& eigenvalues() { return m_evals; };

    /**
     * @brief Return Matrix of selected eigenvectors (columns correspond to
     * values in) matching row of eigenvalues().
     *
     * @return Eigenvectors as matrix, 1 vector per row
     */
    const EigenVectorType& eigenvectors() { return m_evecs; };

    /**
     * @brief Basic constructor
     */
    BandLanczosSelfAdjointEigenSolver()
    {
        init();
    };

    /**
     * @brief Computes the inverse square root of the matrix.
     *
     * This function uses the eigendecomposition \f$A = V D V^{-1}\f$ to compute
     * the inverse square root as \f$VD^{-1/2} V^{-1}\f$. This is cheaper than
     * first computing the square root with operatorSqrt() and then its inverse
     * with MatrixBase::inverse().
     *
     * @return the inverse positive-definite square root of the matrix
     */
    MatrixType operatorInverseSqrt() const
    {
        eigen_assert(m_status == 1 && "Eigenvectors not yet computed.");
        return eigenvectors()*eigenvalues.cwiseInverse().cwiseSqrt()*
                    eigenvectors().transpose();
    };


    /**
     * @brief Computes the positive-definite square root of the matrix.
     *
     * The square root of a positive-definite matrix \f$A\f$ is the
     * positive-definite matrix whose square equals \f$A\f$. This function uses
     * the eigendecomposition \f$ A = V D V^{-1} \f$ to compute the square root as
     * \f[ A^{1/2} = V D^{1/2} V^{-1} \f]
     *
     * @return the positive-definite square root of the matrix
     */
    MatrixType operatorSqrt() const
    {
        eigen_assert(m_status == 1 && "Eigenvectors not yet computed.");
        return eigenvectors()*eigenvalues.cwiseSqrt()*eigenvectors().transpose();
    };

    /**
     * @brief Returns information on the results of compute.
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

    /**
     * @brief Constructor that computes A
     *
     * @param A matrix to compute
     * @param randbasis size of random basis vectors to build Krylov basis
     * from, this should be roughly the number of clustered large eigenvalues.
     * If results are not sufficiently good, it may be worth increasing this
     */
    BandLanczosSelfAdjointEigenSolver(const Ref<const MatrixType>& A, size_t randbasis)
    {
        init();
        compute(A, randbasis);
    };

    /**
     * @brief Constructor that computes A with the initial projection (Krylov
     * basis vectors) set to the columns of V
     *
     * @param A matrix to compute
     * @param V initial projection, to build Krylove basis from. If you have a
     * guess of large eigenvectors that would be good.
     */
    BandLanczosSelfAdjointEigenSolver(const Ref<const MatrixType>& A,
            const Ref<const MatrixType>& V)
    {
        init();
        eigen_assert(A.rows() == V.rows() && "A is not Square.");
        if(A.rows() != V.rows())
            m_status = InvalidInput;
        else
            compute(A, V);
    };

    /**
     * @brief Solves A with the initial projection (Krylov
     * basis vectors) set to the random vectors of dimension set by
     * randbasis.
     *
     * @param A matrix to compute the eigensystem of
     * @param randbasis size of random basis vectors to build Krylov basis
     * from, this should be roughly the number of clustered large eigenvalues.
     * If results are not sufficiently good, it may be worth increasing this
     */
    void compute(const Ref<const MatrixType>& A, size_t randbasis)
    {
        eigen_assert(randbasis > 1 && "Rand Basis Must be > 1");
        if(randbasis <= 1) {
            m_status = InvalidInput;
            return;
        }

        // Create Random Matrix
        m_proj.resize(A.rows(), randbasis);
        m_proj.setRandom();

        _compute(A);
    };

    /**
     * @brief Solves A with the initial projection (Krylov
     * basis vectors) set to the random vectors of dimension set by
     * setRandomBasis()
     *
     * @param A matrix to compute the eigensystem of
     */
    void compute(const Ref<const MatrixType>& A, const Ref<const MatrixType>& V)
    {
        m_proj = V;
        _compute(A);
    };

    /**
     * @brief After this ratio of eigenvalues have been found, stop.
     * Note that this won't work of the matrix is non-positive definite. use
     * TraceSquareStop for that.
     *
     * @param ratio Ratio of total eigenvalues to stop at.
     */
    void setEValStop(Scalar ratio) { m_trace_stop = ratio; };

    /**
     * @brief After this ratio of eigenvalues have been found, stop.
     * Note that this won't work of the matrix is non-positive definite. use
     * TraceSquareStop for that. This resets the default behavior which is 1.
     *
     */
    void setEValStop(Default_t)
    {
        m_trace_stop = INFINITY;
    };

    /**
     * @brief After this ratio of eigenvalues have been found, stop.
     * Note that this won't work of the matrix is non-positive definite. use
     * TraceSquareStop for that. This resets the default behavior which is 1.
     *
     * @return Ratio of total eigenvalues to recover
     */
    Scalar eValStop() { return m_trace_stop; };

    /**
     * @brief After this ratio of squared eigenvalues have been found, stop.
     *
     * @param ratio Ratio of total squared eigenvalues to stop at.
     */
    void setEValSquareStop(Scalar ratio) { m_tracesq_stop = ratio; };

    /**
     * @brief After this ratio of squared eigenvalues have been found, stop.
     *
     * @param ratio Ratio of total squared eigenvalues to stop at.
     * This resets the default behavior which is 1.
     *
     * @param d default (1)
     */
    void setEValSquareStop(Default_t)
    {
        m_tracesq_stop = NAN;
    };

    /**
     * @brief After this ratio of squared eigenvalues have been found, stop.
     *
     * @return Ratio of total squared eigenvalues to stop at.
     * This resets the default behavior which is 1.
     */
    Scalar eValSquareStop() { return m_tracesq_stop ; };

    /**
     * @brief Set the tolerance for deflation. Algorithm stops when #of
     * deflations hits the number of basis vectors. A higher tolerance will
     * thus cause faster convergence and DOES NOT AFFECT ACCURACY, it may
     * affect the number of found eigenvalues though. Recommended value is
     * sqrt(epsilon), approx 1e-8.
     *
     * @param dtol Tolerance for deflation
     */
    void setDeflationTol(RealScalar dtol) { m_deflation_tol = dtol; };

    /**
     * @brief Set the default deflation tolerance, which is sqrt of epsilon.
     *
     * @param Default_t d
     */
    void SetDeflationTol(Default_t)
    {
        m_deflation_tol = sqrt(std::numeric_limits<RealScalar>::epsilon());
    };

    /**
     * @brief Get the tolerance for deflation. See setDeflationTol()
     *
     * @return dtol Tolerance for deflation
     */
    RealScalar DeflationTol() { return m_deflation_tol; };

    /**
     * @brief Set desired number of output vectors. Less may be returned in the
     * matrix is rank deficient. This is more a guideline
     *
     * @param numvecs Expected number of vectors we would like. Note that less
     * maybe returned if the matrix is rank-deficient or more could be returned
     * if as a consequence of a few extra iterations, more are available at
     * respectible accuracy
     */
    void setDesiredRank(int maxiters) { m_outvecs = maxiters; };

    /**
     * @brief Set desired number of output vectors. Less may be returned in the
     * matrix is rank deficient. This is more a guideline
     *
     * @param numvecs Expected number of vectors we would like. Note that less
     * maybe returned if the matrix is rank-deficient or more could be returned
     * if as a consequence of a few extra iterations, more are available at
     * respectible accuracy
     */
    void setDesiredRank(Default_t) { m_outvecs = -1; };

    /**
     * @brief Get estimated number of vectors
     */
    int desiredVecs() { return m_outvecs; };

private:

    void init()
    {
        m_status = 0;
        m_outvecs= -1;
        m_deflation_tol = std::sqrt(std::numeric_limits<double>::epsilon());
        setEValSquareStop(Default);
        setEValStop(Default);
    };

    /**
     * @brief Checks to see if the BandLanczos Algorithm can stop by checking
     * the number of converged eigenvectors. Also sets m_evals/m_evecs to the
     * eigenvalues and vectors of T.
     *
     * Number of valid eigenpairs is estimated by computing
     * the Eigenvectors for the approximate (T) matrix, then using the approximate
     * error from Ruhe:
     *
     * ||r|| = ||T_{pj}z||
     *
     * where r is the residual/error in the  eigenvector, T_{pj} is the the next
     * p rows of T and z is the eigenvector of T (size jxj). The error is
     * approximate because we don't include the off diagonals (deflated) vectors
     *
     * Ruhe A. Implementation aspects of band Lanczos algorithms for computation of
     * eigenvalues of large sparse symmetric matrices. Math Comput. 1979
     * May 1;33(146):680–680. Available from:
     * http://www.ams.org/jourcgi/jour-getitem?pii=S0025-5718-1979-0521282-9
     *
     * @tparam _MatrixType Type of matrix
     * @param T Matrix that is similar to A (same eigenvalues and vectors). Prior
     * to convergence some of the eigenvalues will be incorrect, the purpose of
     * this function is to measure the approximate error in eigenvectors/values
     * @param V Lanczos Vectors (including the candidates, which are used to
     * estimate the next band rows of T)
     * @param bandrad Current band radius, bandwidth is 2*bandrad+1
     * @param ev_sum_t Threshold for stopping based on sum of eigenvalues, absolute
     * value, ignored if this values is NAN or INF
     * @param ev_sumsq_t Threshold for stopping based on sum of sequared
     * eigenvalues, absolute value, ignored if this values is NAN or INF
     *
     * @return Number of valid eigenvalues, or 0 if not all the convergeance
     * criteria have been met
     */
    int check(Ref<MatrixType> T, const Ref<const MatrixType> V,
            int bandrad, double ev_sum_t, double ev_sumsq_t);
//            const Ref<const MatrixType> A);

    /**
     * @brief Band Lanczos Methof for Hessian Matrices
     *
     * p initial guesses (b_1...b_p)
     * set v_k = b_k for k = 1,2,...p
     * set p_c = p
     * set I = nullset
     * for j = 1,2, ..., until convergence or p_c = 0; do
     * (3) compute ||v_j||
     *     decide if v_j should be deflated, if yes, then
     *         if j - p_c > 0, set I = I union {j-p_c}
     *         set p_c = p_c-1. If p_c = 0, set j = j-1 and STOP
     *         for k = j, j+1, ..., j+p_c-1, set v_k = v_{k+1}
     *         return to step (3)
     *     set t(j,j-p_c) = ||v_j|| and normalize v_j = v_j/t(j,j-p_c)
     *     for k = j+1, j+2, ..., j+p_c-1, set
     *         t(j,k-p_c) = v_j^*v_k and v_k = v_k - v_j t(j,k-p_c)
     *     compute v(j+p_c) = Av_j
     *     set k_0 = max{1,j-p_c}. For k = k_0, k_0+1,...,j-1, set
     *         t(k,j) = conjugate(t(j,k)) and v_{j+p_c} = v_{j+p_c}-v_k t(k,j)
     *     for k in (I union {j}) (in ascending order), set
     *         t(k,j) = v^*_k v_{j+p_c} and v_{j+p_c} = v_{j+p_c} - v_k t(k,j)
     *     for k in I, set s(j,k) = conjugate(t(k,j))
     *     set T_j^(pr) = T_j + S_j = [t(i,k)] + [s(i,k)] for (i,k=1,2,3...j)
     *     test for convergence
     * end for
     *
     * TODO Sparse storage of T
     *
     * @param A Matrix to decompose
     */
    void _compute(const Ref<const MatrixType> A)
    {
        EigenVectorType& V = m_proj;

        // Normalize Inputs, So Deflation Tolerance Makes Sense
        for(size_t ii=0; ii<V.cols(); ii++) {
            V.col(ii).normalize();
        }

        // I in text, the iterators to nonzero rows of T(d) as well as the index
        // of them in nonzero_i
        std::list<int> nonzero;
        int pc = V.cols();

        // We are going to continuously grow these as more Lanczos Vectors are
        // computed
        int csize = V.cols();
        ApproxType approx(csize, csize);
        approx.fill(0);

        // V is the list of candidates
        VectorType band(pc); // store values in the band T[jj,jj-pc] to T[jj, jj-1]
        int jj=0;

        // Estimate of Matrix A's Induced Norm
        RealScalar anorm = 0;

        /* Stopping Conditions */
        // Eigensolver
        SelfAdjointEigenSolver<ApproxType> eig;

        // Trace
        double tr_thresh = NAN;
        if(m_trace_stop >= 0 && m_trace_stop <= 1)
            tr_thresh = m_trace_stop*A.trace();

        // Trace Squared
        double trsq_thresh = NAN;
        if(m_tracesq_stop >= 0 && m_tracesq_stop <= 1)
            trsq_thresh = m_tracesq_stop*(A*A).trace();

        while(pc > 0) {
            if(jj+pc+1 >= csize) {
                // Need to Grow
                int newsize = (jj+1+pc)*2;
                approx.conservativeResize(newsize, newsize);
                approx.topRightCorner(csize, newsize-csize).fill(0);
                approx.bottomRightCorner(newsize-csize, newsize-csize).fill(0);
                approx.bottomLeftCorner(newsize-csize, csize).fill(0);

                V.conservativeResize(NoChange, newsize);
                V.rightCols(newsize-csize).fill(0);
                csize = newsize;
            }

            // (3) compute ||v_j||
            double Vjnorm = V.col(jj).norm();

            /*******************************************************************
             * Perform Deflation if current (jth vector) is linearly dependent
             * on the previous vectors
             ******************************************************************/
            // decide if vj should be deflated
            if(!(Vjnorm > anorm*m_deflation_tol)) {
                // if j-pc > 0 (switch to 0 based indexing), I = I U {j-pc}
                if(jj-pc>= 0)
                    nonzero.push_back(jj-pc);

                // set pc = pc - 1
                if(--pc == 0) {
                    // if pc==0 set j = j-1 and stop
                    jj--;
                    break;
                }

                // for k = j , ... j+pc-1, set v_k = v_{k+1}
                // return to step 3
                // Erase Vj and leave jj the same
                // NOTE THAT THIS DOESN't HAPPEN MUCH SO WE DON'T WORRY AOBUT THE
                // LINEAR TIME NECESSARY
                for(int cc = jj; cc<jj+pc; cc++)
                    V.col(cc) = V.col(cc+1);
                continue;
            }

            // set t_{j,j-pc} = ||V_j||
            band[0] = Vjnorm;
            if(jj-pc >= 0)
                approx(jj, jj-pc) = Vjnorm;

            // normalize vj = vj/t{j,j-pc}
            V.col(jj) /= Vjnorm;

            /************************************************************
             * Orthogonalize Candidate Vectors Against Vj
             * and make T(j,k-pc) = V(j).V(k) for k = j+1, ... jj+pc
             * or say T(j,k) = V(j).V(k+pc) for k = j-pc, ... jj-1
             ************************************************************/
            // for k = j+1, j+2, ... j+pc-1
            for(int kk=jj+1; kk<jj+pc; kk++) {
                // set t_{j,k-pc} = v^T_j v_k
                Scalar vj_vk = V.col(jj).dot(V.col(kk));
                band[kk-pc-(jj-pc)] = vj_vk;

                if(kk-pc >= 0)
                    approx(jj,kk-pc) = vj_vk;

                // v_k = v_k - v_j t_{j,k-p_c}
                V.col(kk) -= V.col(jj)*vj_vk;
            }

            /************************************************************
             * Create a New Candidate Vector by transforming current
             ***********************************************************/
            // compute v_{j+pc} = A v_j
            //            V.col(jj+pc) = A*V.col(jj);
            V.col(jj+pc) = A*V.col(jj);

            // update estimate of ||A||
            anorm = std::max(anorm, V.col(jj+pc).norm());

            /*******************************************************
             * Fill Off Diagonals with reflection T(k,j) =
             *******************************************************/
            // set k_0 = max{1,j-pc} for k = k_0,k_0+1, ... j-1 set
            for(int kk = std::max(0, jj-pc); kk < jj; kk++) {

                // t_kj = conj(t_jk)
                Scalar t_kj;
                t_kj = numext::conj(band[kk-(jj-pc)]);
                approx(kk, jj) = t_kj;

                // v_{j+pc} = v_{j+pc} - v_k t_{k,j}
                V.col(jj+pc) -= V.col(kk)*t_kj;
            }

            /*****************************************************
             * Orthogonalize Future vectors with deflated vectors
             * and the current vector
             ****************************************************/
            // for k in I
            for(std::list<int>::iterator kk = nonzero.begin();
                            kk != nonzero.end(); ++kk) {
                // t_{k,j} = v_k v_{j+pc}
                Scalar vk_vjpc = V.col(*kk).dot(V.col(jj+pc));
                approx(*kk, jj) = vk_vjpc;

                // v_{j+pc} = v_{j+pc} - v_k t_{k,j}
                V.col(jj+pc) -= V.col(*kk)*vk_vjpc;
            }
            // include jj
            {
                // t_{k,j} = v_k v_{j+pc}
                Scalar vk_vjpc = V.col(jj).dot(V.col(jj+pc));
                approx(jj, jj) = vk_vjpc;

                // v_{j+pc} = v_{j+pc} - v_k t_{k,j}
                V.col(jj+pc) -= V.col(jj)*vk_vjpc;
            }

            // for k in I, set s_{j,k} = conj(t_{k,j})
            for(std::list<int>::iterator kk = nonzero.begin();
                            kk != nonzero.end(); ++kk) {
                approx(jj, *kk) = numext::conj(approx(*kk, jj));
            }

            if(check(approx.topLeftCorner(jj+1+pc,jj+1+pc),
                        V.leftCols(jj+1+pc), pc, tr_thresh, trsq_thresh) > 0)
                break;
            jj++;
        }

        // Check Number of Good Eigenvalues, and update eigenpairs for T
        int ndim = check(approx.topLeftCorner(jj+1+pc,jj+1+pc),
                    V.leftCols(jj+1+pc), pc, tr_thresh, trsq_thresh);
        if(ndim <= 0) {
            m_status = -1;
            return;
        }

        // Compute Eigen Solution to Similar Matrix, then project through V
        m_evals = m_evals.tail(ndim);
        m_evecs = V.leftCols(jj+1)*m_evecs.rightCols(ndim);

        m_status = 1;
    }

    /**
     * @brief Maximum rank to compute (sets max size of T matrix). < 0 will
     * remove all limits.
     */
    int m_outvecs;

    /**
     * @brief Tolerance for deflation. Algorithm designer recommends
     * sqrt(epsilon), which is the default
     */
    double m_deflation_tol;

    double m_trace_stop;
    double m_tracesq_stop;

    EigenValuesType m_evals;
    EigenVectorType m_evecs;
    EigenVectorType m_proj; // Computed Projection Matrix (V)

    /**
     * @brief Status of computation
     * 0 nothing has happened yet
     * 1 success
     * -1 invalid input
     * -2 non-convergnence
     * -3 Numerical Issue
     */
    int m_status;
};

/**
 * @brief Checks to see if the BandLanczos Algorithm can stop by checking
 * the number of converged eigenvectors. Also sets m_evals/m_evecs to the
 * eigenvalues and vectors of T.
 *
 * Number of valid eigenpairs is estimated by computing
 * the Eigenvectors for the approximate (T) matrix, then using the approximate
 * error from Ruhe:
 *
 * ||r|| = ||T_{pj}z||
 *
 * where r is the residual/error in the  eigenvector, T_{pj} is the the next
 * p rows of T and z is the eigenvector of T (size jxj). The error is
 * approximate because we don't include the off diagonals (deflated) vectors
 *
 * Ruhe A. Implementation aspects of band Lanczos algorithms for computation of
 * eigenvalues of large sparse symmetric matrices. Math Comput. 1979
 * May 1;33(146):680–680. Available from:
 * http://www.ams.org/jourcgi/jour-getitem?pii=S0025-5718-1979-0521282-9
 *
 * @tparam _MatrixType Type of matrix
 * @param T Matrix that is similar to A (same eigenvalues and vectors). Prior
 * to convergence some of the eigenvalues will be incorrect, the purpose of
 * this function is to measure the approximate error in eigenvectors/values
 * @param V Lanczos Vectors (including the candidates, which are used to
 * estimate the next band rows of T)
 * @param bandrad Current band radius, bandwidth is 2*bandrad+1
 * @param ev_sum_t Threshold for stopping based on sum of eigenvalues, absolute
 * value, ignored if this values is NAN or INF
 * @param ev_sumsq_t Threshold for stopping based on sum of sequared
 * eigenvalues, absolute value, ignored if this values is NAN or INF
 *
 * @return Number of valid eigenvalues, or 0 if not all the convergeance
 * criteria have been met
 */
template <typename _MatrixType>
int BandLanczosSelfAdjointEigenSolver<_MatrixType>::check(
        Ref<MatrixType> T, const Ref<const MatrixType> V,
        int bandrad, double ev_sum_t, double ev_sumsq_t)
//        const Ref<const MatrixType> A)
{
    const double EVTHRESH = 0.0001;

    // Return Not Done if 1) haven't reached the desired number of EV's,
    // 2) trace hasn't reached the desired value, 3) trace of TT has not
    // reached the desired value

    if(T.cols() < 2*bandrad)
        return 0;

    if(m_outvecs > 1 && T.rows() < m_outvecs)
        return 0;

    if(!isinf(ev_sum_t) && !isnan(ev_sum_t) && T.trace()<ev_sum_t)
        return 0;

    if(!isinf(ev_sumsq_t) && !isnan(ev_sumsq_t) && (T*T).trace()<ev_sumsq_t)
        return 0;

    int N = T.rows()-bandrad; // Finished Rows/Columns of T
    SelfAdjointEigenSolver<MatrixType> eig(T.topLeftCorner(N,N));
    m_evals = eig.eigenvalues();
    m_evecs = eig.eigenvectors();

//#ifdef DEBUG
//    cerr<< "\n=======================\nT:\n"<<T<<endl;
//    cerr<<"\nLAMBDA:"<<eig.eigenvalues().transpose()<<endl;
//    cerr<<"EVs:\n"<<eig.eigenvectors()<<endl;
//    cerr<<"Proj EVs:\n"<<V.leftCols(N)*eig.eigenvectors()<<endl;
//#endif //DEBUG
    size_t nvalid = 0;
    for(int rr=N; rr<N+bandrad; rr++) {
        double vnorm = V.col(rr).norm();
        for(int cc=rr-bandrad; cc<N; cc++)
            T(rr, cc) = V.col(rr).dot(V.col(cc+bandrad))/vnorm;
    }
//#ifdef DEBUG
//    cerr << "T Est:\n\n"<<T.bottomLeftCorner(bandrad, N)<<endl;
//#endif //DEBUG
    double sum = 0;
    double sumsq = 0;
    for(int vv=0; vv<N; vv++) {
        double esterr = (T.bottomLeftCorner(bandrad, N)*
                eig.eigenvectors().col(N-1-vv)).norm();
//        VectorXd ev = V.leftCols(N)*eig.eigenvectors().col(N-1-vv);
//        double esterr = (A*ev-eig.eigenvalues()[N-1-vv]*ev).norm();
        if(esterr > EVTHRESH)
            break;

        nvalid++;

        sum += eig.eigenvalues()[N-1-vv];
        sumsq += eig.eigenvalues()[N-1-vv]*eig.eigenvalues()[N-1-vv];
    }

    // Remove Predictions Made (in case deflation happens and the true band
    // ends up being smaller than the predicted one)
    T.bottomLeftCorner(bandrad, N).setZero();

    if((m_outvecs <= 1 || nvalid >= m_outvecs) &&
            (std::isnan(ev_sumsq_t) || std::isinf(ev_sumsq_t)
             || sumsq>ev_sumsq_t) && (std::isnan(ev_sum_t) ||
                 std::isinf(ev_sum_t) || sum>ev_sum_t)) {
        return nvalid;
    }
    return 0;
}

} // end namespace Eigen

#endif // EIGEN_BAND_LANCZOS_SELF_ADJOINT_EIGEN_SOLVER_H

