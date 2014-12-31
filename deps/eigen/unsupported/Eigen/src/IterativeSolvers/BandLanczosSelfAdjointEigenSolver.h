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

namespace Eigen {

#include <Eigen/Eigenvalues>
#include <limits>
#include <cmath>

#include <iostream>
using std::cerr;
using std::endl;

namespace internal {
template <typename T>
inline T conj(const T& v)
{
    return std::conj(v);
}

template <>
inline double conj(const double& v)
{
    return v;
}

template <>
inline float conj(const float& v)
{
    return v;
}
}

/**
 * @brief Solves eigenvalues and eigenvectors of a hermitian matrix. Currently
 * it is only really symmetric, Values need to made generic enough for complex
 *
 * TODO: documentation with usage examples
 * TODO: documentation of settings
 * TODO: make a template over type of Matrix
 * TODO: Fix for complex scalars
 *
 */
template <typename _Scalar>
class BandLanczosSelfAdjointEigenSolver
{
public:
    typedef _Scalar Scalar;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

    /**
     * @brief Return vector of selected eigenvalues
     *
     * @return Eigenvalues as vector
     */
    const VectorType& eigenvalues() { return m_evals; };

    /**
     * @brief Return Matrix of selected eigenvectors (columns correspond to
     * values in) matching row of eigenvalues().
     *
     * @return Eigenvectors as matrix, 1 vector per row
     */
    const MatrixType& eigenvectors() { return m_evecs; };

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
        eigen_assert(m_status == 1);
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
        eigen_assert(m_status == 1);
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
    BandLanczosSelfAdjointEigenSolver(const MatrixType& A, size_t randbasis)
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
    BandLanczosSelfAdjointEigenSolver(const MatrixType& A, const MatrixType& V)
    {
        init();
        eigen_assert(A.rows() == V.rows());
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
    void compute(const MatrixType& A, size_t randbasis)
    {
        eigen_assert(randbasis > 1);
        if(randbasis <= 1) {
            m_status = InvalidInput;
            return;
        }

        // Create Random Matrix
        m_proj.resize(A.rows(), randbasis);
        m_proj.setRandom();

        // Normalize Each Column
        for(int cc=0; cc<m_proj.cols(); cc++) {
            
			// Normalize
            m_proj.col(cc).normalize();
        }

		cerr << m_proj << endl;
        _compute(A);
    };

    /**
     * @brief Solves A with the initial projection (Krylov
     * basis vectors) set to the random vectors of dimension set by
     * setRandomBasis()
     *
     * @param A matrix to compute the eigensystem of
     */
    void compute(const MatrixType& A, const MatrixType& V)
    {
        m_proj = V;
        _compute(A);
    };

    /**
     * @brief Set a hard limit on the number of eigenvectors to compute
     *
     * @param rank of eigenvector output. Set to < 0 to loop infinitely
     */
    void setRank(int rank) { m_rank = rank; };

    /**
     * @brief Get the current limit on the number of eigenvectors
     *
     * @return Maximum rank of eigenvector output
     */
    int getRank() { return m_rank; };

    /**
     * @brief Set the tolerance for deflation. Algorithm stops when #of
     * deflations hits the number of basis vectors. A higher tolerance will
     * thus cause faster convergence and DOES NOT AFFECT ACCURACY, it may
     * affect the number of found eigenvalues though. Recommended value is
     * sqrt(epsilon), approx 1e-8.
     *
     * @param dtol Tolerance for deflation
     */
    void setDeflationTol(double dtol) { m_deflation_tol = dtol; };

    /**
     * @brief Get the tolerance for deflation. See setDeflationTol()
     *
     * @return dtol Tolerance for deflation
     */
    double getDeflationTol() { return m_deflation_tol; };

    /**
     * @brief Stop after the trace of the similar matrix (T^2) exceeds the
     * ratio of total sum of squared eigenvalues.
     *
     * @param stop 0 to 1 with 1 stopping when ALL the variance has been found
     * and 0 stopping immediately. Set to INFINITY or NAN to only stop
     * naturally (when the Kyrlov Subspace has been exhausted).
     */
    void setTraceSqrStop(double stop) { m_tracesqr_stop = stop; };

    /**
     * @brief Return trace squared stopping condition to default.
     *
     * @param Default_t default
     */
    void setTraceSqrStop(Default_t d) { m_tracesqr_stop = INFINITY; };

    /**
     * @brief Get stop parameter based on sum of squared eigenvalues.
     * See setTraceSqrStop()
     *
     * @return Get the current stopping condition based on the sum squared
     * eigenvalues (trace squared)
     */
    double traceSqrStop() { return m_tracesqr_stop; };
private:

    void init()
    {
        m_status = 0;
        m_rank = -1;
        m_deflation_tol = std::sqrt(std::numeric_limits<double>::epsilon());
        m_tracesqr_stop = INFINITY;
    };

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
    void _compute(const MatrixType& A)
    {
        Scalar varsum = 0;
        Scalar vartotal = INFINITY;
        if(!isinf(m_tracesqr_stop) && !isnan(m_tracesqr_stop)) {
            vartotal = (A.transpose()*A).trace();
        }

        MatrixType& V = m_proj;

        // I in text, the iterators to nonzero rows of T(d) as well as the index
        // of them in nonzero_i
        std::list<int> nonzero;
        int pc = V.cols();

        // We are going to continuously grow these as more Lanczos Vectors are
        // computed
        int csize = V.cols()*2;
        MatrixType approx(csize, csize);
        V.conservativeResize(NoChange, V.cols()*2);

        // V is the list of candidates
        VectorType band(pc); // store values in the band T[jj,jj-pc] to T[jj, jj-1]
        int jj=0;

        while(pc > 0 && (m_rank < 0 || jj < m_rank)) {
            if(jj+pc >= csize) {
                // Need to Grow
                csize *= 2;
                approx.conservativeResize(csize, csize);
                V.conservativeResize(NoChange, csize);
            }

            // (3) compute ||v_j||
            double Vjnorm = V.col(jj).norm();

            /*******************************************************************
             * Perform Deflation if current (jth vector) is linearly dependent
             * on the previous vectors
             ******************************************************************/
            // decide if vj should be deflated
            if(Vjnorm < m_deflation_tol) {
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

            /*******************************************************
             * Fill Off Diagonals with reflection T(k,j) =
             *******************************************************/
            // set k_0 = max{1,j-pc} for k = k_0,k_0+1, ... j-1 set
            for(int kk = std::max(0, jj-pc); kk < jj; kk++) {

                // t_kj = conj(t_jk)
                Scalar t_kj;
                t_kj = internal::conj(band[kk-(jj-pc)]);
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
                approx(jj, *kk) = internal::conj(approx(*kk, jj));
            }

            // Compute Trace of M**2 if a stopping point was set based on the
            // total squared eigenvalue sum
            if(!std::isinf(std::abs(vartotal))) {
                varsum += (approx.topLeftCorner(jj+1,jj+1)*
                        approx.topLeftCorner(jj+1,jj+1)).trace();

                if(std::abs(varsum) > std::abs(m_tracesqr_stop*vartotal))
                    break;
            }

            jj++;
        }

        // Set Approximate and Projection to Final Size
        approx.conservativeResize(jj+1, jj+1);
        V.conservativeResize(NoChange, jj+1);

        // Compute Eigen Solution to Similar Matrix, then project through V
        SelfAdjointEigenSolver<MatrixType> computer(approx);
        m_evals = computer.eigenvalues();
        m_evecs = V*computer.eigenvectors();

        // check results to determine numerical issue or non convergence
        for(int ee = 0; ee<m_evals.rows(); ee++) {
            double err = (m_evals[ee]*m_evecs.col(ee) -
                    A*m_evecs.col(ee)).squaredNorm();
            // numerical issue
            if(err > std::sqrt(std::numeric_limits<double>::epsilon())) {
                m_status = -2;
                return ;
            }
        }
        m_status = 1;
    }

    /**
     * @brief Maximum rank to compute (sets max size of T matrix). < 0 will
     * remove all limits.
     */
    int m_rank;

    /**
     * @brief Tolerance for deflation. Algorithm designer recommends
     * sqrt(epsilon), which is the default
     */
    double m_deflation_tol;

    /**
     * @brief Stop after the trace of the similar matrix (T^2) exceeds some
     * value, which may be known from the input matrices' trace.
     * This could be used to stop after the sum of squared eigenvalues
     * exceeds some value.
     */
    double m_tracesqr_stop;

    VectorType m_evals;
    MatrixType m_evecs;
    MatrixType m_proj; // Computed Projection Matrix (V)

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

} // end namespace Eigen

#endif // EIGEN_BAND_LANCZOS_SELF_ADJOINT_EIGEN_SOLVER_H

