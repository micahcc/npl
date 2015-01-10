/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file generalLinearModel.cpp Tests large scale GLM
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>
#include <random>

#include <Eigen/Dense>
#include "statistics.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "utility.h"
#include "nplio.h"
#include "basic_plot.h"
#include "macros.h"

using std::string;
using Eigen::MatrixXd;

using namespace npl;

int main()
{
	// TODO add intercept

	vector<size_t> voldim({12,17,13});
	vector<size_t> fdim({12,17,13,1024});
	ptr<MRImage> realbeta;
	ptr<MRImage> fmri;

	// create X and fill it with sin waves
	MatrixXd X(fdim[3], 2);
	for(size_t rr=0; rr<X.rows(); rr++ ) {
		for(size_t cc=0; cc<X.cols(); cc++) {
			X(rr,cc) = cos(M_PI*rr/100*cc);
		}
	}

	{
		/*
		 * Create Test Image: 4 Images
		 */

		// make array
		{
			vector<ptr<NDArray>> prebeta;
			prebeta.push_back(createMRImage(voldim.size(), voldim.data(), FLOAT32));
			prebeta.push_back(createMRImage(voldim.size(), voldim.data(), FLOAT32));

			fillCircle(prebeta[0], 3, 1);
			fillCircle(prebeta[1], 3, 10);
			realbeta = dPtrCast<MRImage>(concatElevate(prebeta));
		}

		fmri = createMRImage(fdim.size(), fdim.data(), FLOAT32);
		fillGaussian(fmri);

		// add values from beta*X, cols of X correspond to the highest dim of beta
		Vector3DIter<double> bit(realbeta);
		bit.goBegin();

		// for each voxel in fMRI,
		Vector3DIter<double> it(fmri);
		for(it.goBegin(); !it.eof(); ++it, ++bit) {
			VectorXd b(2);
			b[0] = bit[0];
			b[1] = bit[1];
			VectorXd y = X*b;
			// for each row in X/time in fMRI
			for(size_t rr=0; rr<X.rows(); rr++) {
				it.set(rr, it[rr]+y[rr]);
			}
		}

		writeMRImage(fmri, "signal_noise.nii.gz");
		writeMRImage(realbeta, "betas.nii.gz");
	}


	/* Perform Regression */
	int tlen = fmri->tlen();
	if(fmri->ndim() != 4) {
		throw INVALID_ARGUMENT("Input Image should be 4D!");
	}

	// create output images
	vector<size_t> osize(4, 0);
	for(size_t ii=0; ii<3; ii++) {
		osize[ii] = fmri->dim(ii);
	}
	osize[3] = X.cols();

	auto t_est = dPtrCast<MRImage>(fmri->copyCast(4, osize.data()));
	auto p_est = dPtrCast<MRImage>(fmri->copyCast(4, osize.data()));
	auto b_est = dPtrCast<MRImage>(fmri->copyCast(4, osize.data()));

	vector<int64_t> ind(3);

	// Cache Reused Vectors
	auto Xinv = pseudoInverse(X);
	auto covInv = pseudoInverse(X.transpose()*X);

	const double MAX_T = 20;
	const double STEP_T = 0.01;
	StudentsT stud_dist(X.rows()-1, STEP_T, MAX_T);

	VectorXd signal(tlen);

	RegrResult ret;
	// regress each timesereies
	Vector3DIter<double> tit(t_est), pit(p_est), bit(b_est);
	tit.goBegin();
	pit.goBegin();
	bit.goBegin();
	size_t count=0;
	for(Vector3DConstIter<double> fit(fmri); !fit.eof() && !bit.eof(); ++fit, ++tit, ++pit, ++bit) {

		// copy to signal
		for(size_t tt=0; tt<tlen; tt++)
			signal[tt] = fit[tt];

		regress(ret, signal, X, covInv, Xinv, stud_dist);

		if(count == 10) {
			vector<int64_t> index(4);
			fit.index(index);
			Vector3DView<double> beta_acc(realbeta);
			VectorXd beta(4);
			beta[0] = beta_acc(index[0], index[1], index[2] ,0);
			beta[1] = beta_acc(index[0], index[1], index[2] ,1);
			//            beta[2] = beta_acc(index[0], index[1], index[2] ,2);
			//            beta[3] = beta_acc(index[0], index[1], index[2] ,3);

			cerr << "Beta: Est" << ret.bhat.transpose() << " vs " << beta.transpose()
				<< endl;
			cerr << "Standard Errors: " << ret.std_err.transpose() << endl;
			cerr << "T-Value: " << ret.t.transpose() << endl;
			cerr << "P-value: " << ret.p.transpose() << endl;
			cerr << "DOF: "<< ret.dof << endl;

			Plotter plt;
			plt.addArray(signal.rows(), signal.data());
			plt.addArray(X.rows(), X.col(0).data());
			plt.addArray(X.rows(), X.col(1).data());
			//            plt.addArray(X.rows(), X.col(2).data());
			//            plt.addArray(X.rows(), X.col(3).data());
			plt.write("y_x.svg");

			Plotter plt2;
			plt2.addArray(signal.rows(), signal.data());
			plt2.addArray(ret.yhat.rows(), ret.yhat.data());
			VectorXd x0 = (ret.bhat[0]*X.col(0));
			VectorXd x1 = (ret.bhat[1]*X.col(1));
			//            VectorXd x2 = (ret.bhat[2]*X.col(2));
			//            VectorXd x3 = (ret.bhat[3]*X.col(3));
			plt2.addArray(X.rows(), x0.data());
			plt2.addArray(X.rows(), x1.data());
			//            plt2.addArray(X.rows(), x2.data());
			//            plt2.addArray(X.rows(), x3.data());
			plt2.write("y_vs_yht.svg");
		}

		for(size_t ii=0; ii<X.cols(); ii++) {
			tit.set(ii, ret.t[ii]);
			pit.set(ii, ret.p[ii]);
			bit.set(ii, ret.bhat[ii]);
		}
		count++;
	}

	assert(bit.eof() && pit.eof() && tit.eof());

	/* Test Results */
	writeMRImage(b_est, "beta_est.nii.gz");
	writeMRImage(p_est, "p_est.nii.gz");
	writeMRImage(t_est, "t_est.nii.gz");
	return 0;
}




