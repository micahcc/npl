
#include "statistics.h"

#include <Eigen/SVD>


namespace npl {


/**
 * @brief Computes the Principal Components of input matrix X
 *
 * Outputs reduced dimension (fewer cols) in output. Note that prio to this,
 * the columns of X should be 0 mean. 
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of principal components
 */
Matrix pca(const Matrix& X, double varth);
{
	const double VARTHRESH = .05;
    double totalv = 0; // total variance
    size_t outdim = 0;

#ifndef NDEBUG
    std::cout << "Computing SVD" << endl;
#endif //DEBUG
    JacobiSVD<MatrixXf> svd(m, ComputeThinU);
#ifndef NDEBUG
    std::cout << "Done" << endl;
#endif //DEBUG
	
    const Vector& W = svd.SingularValues();
    const Matrix& U = svd.MatrixU();
    //only keep dimensions with variance passing the threshold
    for(size_t ii=0; ii<W.rows(); ii++)
        totalv = W[ii]*W[ii];

    double sum = 0;
    for(outdim = 0; outdim < W.rows(); outdim++) {
        sum += W[dd]*W[dd];
        if(sum > totalv*(1-VARTHRESH))
            break;
    }
#ifndef NDEBUG
    std::cout << "Output Dimensions: " << odim 
            << "\nCreating Reduced Matrix..." << endl;
#endif //DEBUG

    Matrix Xr(X.rows(), odim);
	for(int rr=0; rr<X.rows(); rr++) {
		for(int cc=0; cc<odim; cc++) {
			Xr(rr,cc) = U(rr, cc)*W[cc];
		}
	}
#ifndef NDEBUG
	std::cout  << "  Done" << endl;
#endif 

    return Xr;
}

/**
 * @brief Computes the Independent Components of input matrix X. Note that
 * you should run PCA on X before running ICA.
 *
 * Outputs reduced dimension (fewer cols) in output
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of independent components
 */
Matrix ica(const Matrix& X, double varth)
{
	int samples = X.rows();
	int dims = X.cols();
    int ncomp = min(samples, dims);
	
	double mag = 1;
	Vector projected(samples);
    Vector nonlined(samples);
    Vector expected(dims);
    Vector wold(dims);
	
	Matrix WT(ncomp, dims);
	WT.setlength(ncomp, dims);

    // TODO initialize rng
	std::uniform_real_distribution<double> unif(0, 1);

	for(int pp = 0 ; pp < ncomp ; pp++) {
		//randomize weights
		for(unsigned int ii = 0; ii < dims ; ii++) 
			WT(pp, ii) = unif(rng);
			
        // TODO HERE
		//GramSchmidt Decorrelate
		//sum(w^t_p w_j w_j) for j < p
		//cache w_p for wt_wj mutlication
		alglib::vmove(expected.getcontent(), 1, &WT(pp, 0), 1, dims);
		for(int jj = 0 ; jj < pp; jj++){
			//w^t_p w_j
			double wt_wj = -alglib::vdotproduct(expected.getcontent(), 1, &WT(jj, 0), 1, dims);

			//w_p -= (w^t_p w_j) w_j
			alglib::vadd(&WT(pp, 0), 1, &WT(pp, 0), 1, dims, wt_wj);
		}

		//normalize
		alglib::vmul(&WT(pp, 0), dims, 1./sqrt(alglib::vdotproduct(&WT(pp, 0), 
						1, &WT(pp, 0), 1, dims)));
		
		*olog << "Peforming Fast ICA: " << pp << endl;
		mag = 1;
		for(int ii = 0 ; mag > 0.0001 && ii < 100 ; ii++) {
			
			//move to old
			alglib::vmove(wold.getcontent(), 1, &WT(pp, 0), 1, dims);

			//E(x g(w^tx) ) - E(g'(w^tx))w
			//w^tx
			alglib::rmatrixmv(samples, dims, Xo, 0, 0, 0, wold, 0, projected, 0);

			//g(w^T x)
			apply(nonlined.getcontent(), projected.getcontent(), fastICA_g2, samples);

			//E(x g(w^t x) )
			alglib::rmatrixmv(dims, samples, Xo, 0, 0, 1, nonlined, 0, expected, 0);
			alglib::vmul(expected.getcontent(), dims, 1./samples);

			//g'(w^T x), deriv(tanh) = 1-tanh^2
			apply(nonlined.getcontent(), projected.getcontent(), fastICA_dg2, samples);

			//E(g'(w^t x)) w
			double expected2 = vectorExpectation(nonlined.getcontent(), samples);

			//             -E(g'(w^tx))*w + E(x g(w^tx)) 
			alglib::vadd(expected.getcontent(), 1, wold.getcontent(), 1, dims, -expected2);

			//decorrelate with previous solutions
			//w_p = w_p - sum(w^t_p w_j w_j) for j < p

			//copy the estimated value into WT, from expected
			alglib::vmove(&WT(pp, 0), 1, expected.getcontent(), 1, dims);

			for(int jj = 0 ; jj < pp; jj++){
				//w^t_p w_j (using cached WT(pp,0) aka w_p)
				double wt_wj = -alglib::vdotproduct(expected.getcontent(), 1, &WT(jj, 0), 1, dims);

				//w_p -= (w^t_p w_j) w_j
				alglib::vadd(&WT(pp, 0), 1, &WT(pp, 0), 1, dims, wt_wj);
			}

			//normalize
			alglib::vmul(&WT(pp, 0), dims, 1./sqrt(alglib::vdotproduct(&WT(pp, 0), 
							1, &WT(pp, 0), 1, dims)));
			
			//compare old and new w, store result in old w
			mag = 0;
			for(unsigned int cc = 0; cc < dims ; cc++) {
				mag += fabs(fabs(WT(pp, cc)) - fabs(wold(cc)));
			}
		}

		*olog << "Final (" << pp << "): ";
		for(unsigned int cc = 0 ; cc < dims ; cc++)
			*olog << WT(pp, cc) << ' ';
		*olog << endl;
	}
	
	Xo.setlength(Xo.rows(), ncomp);
	alglib::rmatrixgemm(X.rows(), ncomp, X.cols(), 1.0, X, 0, 0, 0, WT, 0, 
				0, 1, 0, Xo, 0, 0);
	
//	return S;
}


}
