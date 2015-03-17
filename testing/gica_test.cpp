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
 * @file gica_test.cpp Test Group ICA.
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>

#include "version.h"
#include "fmri_inference.h"
#include "mrimage.h"
#include "ndarray_utils.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include "macros.h"

using std::string;
using std::shared_ptr;
using std::to_string;

using namespace Eigen;
using namespace npl;


void simulate(size_t xsz, size_t ysz, size_t zsz, size_t tsz, double sd,
		size_t nreg);

int coratleast(double thresh, ptr<const MRImage> img1, ptr<const MRImage> img2);

int main()
{
	cerr << "Version: " << __version__ << endl;

	// creates gica_test_prob.nii,gz and gica_test_fmri.nii.gz
	simulate(20, 20, 20, 500, 2, 5);
	std::vector<std::string> inputs(1, "gica_test_fmri.nii.gz");
	std::vector<std::string> masks;

	{
		gicaCreateMatrices(1, 1, masks, inputs, "gica_test_space", 0.5, true, true);
		gicaReduceFull("gica_test_space", "gica_test_space", 0.1, 0.9, 10, true);
		//gicaReduceProb("gica_test_time", "gica_test_time", 10, 3, 0.1, 0.9, true);
		//gicaTemporalICA("gica_test_space", "gica_test_space", "gica_test_space");
		gicaSpatialICA("gica_test_space", "gica_test_space", "gica_test_space", true);

		auto pmap_real = readMRImage("gica_test_prob.nii.gz");
		auto bmap_est  = readMRImage("gica_test_space_bmap_m0.nii.gz");
		if(coratleast(0.5, pmap_real, bmap_est) != 0)
			return -1;
	}

	{
		gicaCreateMatrices(1, 1, masks, inputs, "gica_test_time", 0.5, true, true);
		gicaReduceFull("gica_test_time", "gica_test_time", 0.1, 0.9, 10, true);
		//gicaReduceProb("gica_test_time", "gica_test_time", 10, 3, 0.1, 0.9, true);
		gicaTemporalICA("gica_test_time", "gica_test_time", "gica_test_time", true);
		//gicaSpatialICA("gica_test_space", "gica_test_space", "gica_test_space", true);

		auto pmap_real = readMRImage("gica_test_prob.nii.gz");
		auto bmap_est  = readMRImage("gica_test_time_bmap_m0.nii.gz");
		if(coratleast(0.5, pmap_real, bmap_est) != 0)
			return -1;
	}

	{
		gicaCreateMatrices(1, 1, masks, inputs, "gica_test_space", 0.5, true, true);
		//gicaReduceFull("gica_test_space", "gica_test_space", 0.1, 0.9, true);
		gicaReduceProb("gica_test_space", "gica_test_space", 0.1, 0.9, 10, 3, true);
		//gicaTemporalICA("gica_test_space", "gica_test_space", "gica_test_space", true);
		gicaSpatialICA("gica_test_space", "gica_test_space", "gica_test_space", true);

		auto pmap_real = readMRImage("gica_test_prob.nii.gz");
		auto bmap_est  = readMRImage("gica_test_space_bmap_m0.nii.gz");
		if(coratleast(0.5, pmap_real, bmap_est) != 0)
			return -1;
	}

	{
		gicaCreateMatrices(1, 1, masks, inputs, "gica_test_time", 0.5, true, true);
		//gicaReduceFull("gica_test_time", "gica_test_time", 0.1, 0.9, true);
		gicaReduceProb("gica_test_time", "gica_test_time", 0.1, 0.9, 10, 3, true);
		gicaTemporalICA("gica_test_time", "gica_test_time", "gica_test_time", true);
		//gicaSpatialICA("gica_test_space", "gica_test_space", "gica_test_space", true);

		auto pmap_real = readMRImage("gica_test_prob.nii.gz");
		auto bmap_est  = readMRImage("gica_test_time_bmap_m0.nii.gz");
		if(coratleast(0.5, pmap_real, bmap_est) != 0)
			return -1;
	}

	return 0;
}

/**
 * @brief Creates a random (poisson) series of events, with average number of
 * events per unit time of lambda. Events occur between time tf and t0.
 *
 * @param lambda Number of events per unit time (per second)
 * @param t0 Time 0 to start simulating
 * @param tf Final time to end on
 *
 * @return vector of spike times
 */
vector<double> createRandAct(double lambda, double t0, double tf)
{
	std::random_device rd;
	std::default_random_engine rng(rd());
	std::uniform_real_distribution<double> dist(0, 1);

	vector<double> out;
	double tr = 0.0001;

	while(t0 < tf) {
		double prob = lambda*tr;
		if(dist(rng) < prob)
			out.push_back(t0);
		t0 += tr;
	}

	return out;
}

/**
 * @brief Takes a set of timepoints at which unit impulses occur and returns a
 * timeseries of simulated BOLD signal, from the given input. Note that up to
 * 15 (T0) seconds BEFORE 0, unit impulses will be included so that t0 won't
 * doesn't * have to be at state 0
 *
 * @param times Timepoints of unit spikes, can be any length and spikes my
 * occur in the range (-15, tlen), and still be included
 * @param tlen Total timeseries length in seconds
 * @param tr Sampling time of output in seconds
 * @param hdtr TR in highres BOLD simulation
 * @param learn The weight of the current point, the previous moving average
 * value is weighted (1-learn), setting learn to 1 removes all habituation
 * This limits the effect of habituation.
 *
 * @return Timeseries for the given spike times smoothed by the BOLD signal
 */
vector<double> sampleBOLD(const vector<double>& times, double tlen, double tr,
		double hdtr, double learn)
{
	const double T0 = 15; // offset upsampled timeseries by negative this

	size_t usz = (tlen+T0)/hdtr; // Size of upsampled timeseries
	size_t osz = tlen/tr; // Size of output timeseries

	vector<double> highres(usz);
	std::fill(highres.begin(), highres.end(), 0);

	// Place all spikes in highres sampling (note the area of each is 1 because
	// tr/upsample is the time of a box, and the value added is upsample/tr)
	for(auto t : times) {
		int i = round((t+T0)/hdtr);
		if(i >= 0 && i < highres.size())
			highres[i] += 1./hdtr;
	}

	// Simulate BOLD with HRF
	boldsim(usz, highres.data(), hdtr, learn);

	// Sample at lower frequency
	vector<double> out(osz);
	std::fill(out.begin(), out.end(), 0);

	for(size_t ii=0; ii<osz; ii++) {
		double t = tr*ii;
		int jj = round((t+T0)/hdtr);
		if(jj >= 0 && jj < highres.size()) {
			out[ii] = highres[jj];
			if(std::isnan(out[ii] || std::isinf(out[ii])))
				throw RUNTIME_ERROR("NAN/INF in Simulation");
		}
	}
	return out;
}

void simulate(size_t xsz, size_t ysz, size_t zsz, size_t tsz, double sd,
		size_t nreg)
{
	double learn = 0.05;
	double hdtr = 0.01;
	double tr = 0.5;
	size_t tlen = tsz;
	double noise_sd = 0.01;

	vector<vector<double>> activate;
	cerr << "Simulating Random Activations: ";
	for(size_t ii=0; ii<nreg; ii++) {
		double lambda = (ii+1)*0.1;
		cerr << "lambda "<<lambda<<", ";
		activate.push_back(createRandAct(lambda, -15, tsz*tr));
	}
	cerr << "Done\n";

	size_t outsz[3] = {xsz, ysz, zsz};
	ptr<MRImage> gmprob = createMRImage(3, outsz, FLOAT32);
	for(FlatIter<double> it(gmprob); !it.eof(); ++it)
		it.set(1);

	ptr<MRImage> labelmap;
	cerr << "Simulating Region Map with "<<activate.size()<<"non-zero labels...";
	labelmap = dPtrCast<MRImage>(createRandLabels(gmprob, activate.size(), 2));
	cerr << "Done\n";
	labelmap->write("gica_test_labels.nii.gz");

	vector<vector<double>> design(nreg);

	/* create timeseries for each of the activation spike trains */
	cerr<<"Simulating timeseries...Correlations:\n";
	for(size_t rr=0; rr<nreg; rr++) {
		design[rr] = sampleBOLD(activate[rr], tr*tlen, tr, hdtr, learn);
		assert(design[rr].size() == tlen);

		for(size_t ii=0; ii<=rr; ii++) {
			if(ii != 0) cerr << ",";
			cerr << correlation(tlen, design[rr].data(),
					design[ii].data());
//			cerr << mutualInformation(tlen, design[rr].data(),
//					design[ii].data(), std::sqrt(tlen));
		}
		cerr << endl;
	}
	cerr<<"Done\n";

	/* create 4D image where each volume is a label probability */
	vector<size_t> tmpsize(labelmap->dim(), labelmap->dim()+labelmap->ndim());
	tmpsize.push_back(nreg);
	auto prob = dPtrCast<MRImage>(labelmap->copyCast(tmpsize.size(),
				tmpsize.data(), FLOAT32));
	for(FlatIter<double> it(prob); !it.eof(); ++it)
		it.set(0);

	cerr << "Creating indivudal probability maps...";
	Vector3DIter<double> pit(prob);
	NDIter<int> lit(labelmap);
	for(pit.goBegin(), lit.goBegin(); !pit.eof(); ++pit, ++lit) {
		int l = *lit;
		if(l < 0 || l > nreg) {
			throw INVALID_ARGUMENT("Error, input labelmap has labels outside "
					"the the range provided by the input spike trains. The "
					"labels should range from 0 (unlabeled), 1, .. N for the "
					"N rows of the spike train input");
		} else if(l != 0) {
			pit.set(l-1, 1);
		}
	}
	cerr << "Done\n";

	/* smooth each of the 4D volumes */
	cerr << "Smoothing probability maps...";
	for(size_t dd=0; dd<3; dd++)
		gaussianSmooth1D(prob, dd, sd);
	cerr << "Done\n";

	cerr << "Merging Probability Map with GM Probability"<<endl;
	NDIter<double> git(gmprob);
	for(pit.goBegin(); !pit.eof(); ++pit,++git){
		// Weight Sum of each regions value
		for(size_t rr=0; rr<nreg; rr++)
			pit.set(rr, pit[rr]*(*git));
	}
	prob->write("gica_test_prob.nii.gz");

	/*
	 * for each pixel sum up the contribution of each timeseries then scale by
	 * greymatter probability
	 */
	tmpsize[3] = tlen;
	auto out = dPtrCast<MRImage>(labelmap->copyCast(tmpsize.size(),
				tmpsize.data(), FLOAT32));

	cerr << "Merging activation maps...";
	Vector3DIter<double> oit(out);
	for(pit.goBegin(); !pit.eof(); ++pit,++oit){
		for(size_t tt=0; tt<tlen; tt++) {
			// Weight Sum of each regions value
			double v = 0;
			for(size_t rr=0; rr<nreg; rr++)
				v += design[rr][tt]*pit[rr];

			// Weigtht by GM Prob
			oit.set(tt, v);
		}
	}
	cerr << "Done\n";

	cerr<<"Adding Noise...";
	std::random_device rd;
	std::default_random_engine rng(rd());
	std::normal_distribution<double> dist(0, noise_sd);
	for(oit.goBegin(); !oit.eof(); ++oit) {
		for(size_t tt=0; tt<tlen; tt++)
			oit.set(tt, oit[tt]+dist(rng));
	}
	cerr<<"Done"<<endl;

	out->write("gica_test_fmri.nii.gz");
}

int coratleast(double thresh, ptr<const MRImage> img1, ptr<const MRImage> img2)
{
	int ret = 0;
	cerr << "Corelation" << endl;
	for(size_t t1 = 0; t1 < img1->tlen(); t1++) {
		double maxcor = 0;
		for(size_t t2 = 0; t2 < img2->tlen(); t2++) {
			double corrval = 0;
			double sd1 = 0;
			double sd2 = 0;
			double mu1 = 0;
			double mu2 = 0;
			size_t count = 0;
			for(Vector3DConstIter<double> it1(img1), it2(img2); !it1.eof(); ++it1, ++it2) {
				mu1 += it1[t1];
				mu2 += it2[t2];
				sd1 += it1[t1]*it1[t1];
				sd2 += it2[t2]*it2[t2];
				corrval += it1[t1]*it2[t2];
				count++;
			}
			double cval = fabs(sample_corr(count, mu1, mu2, sd1, sd2, corrval));
			cerr << t1 << " " << t2 << " = " << cval << endl;
			if(cval > maxcor)
				maxcor = cval;
		}
		if(maxcor < thresh) {
			cerr << "Maximum correlation of component " << t1 << " is only " <<
				maxcor << endl;
			ret = -1;
		}
	}

	return ret;
}
