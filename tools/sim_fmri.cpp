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
 * @file sim_fmri.cpp Tool to create a simulated fMRI with a variety of
 * timecourses throughout the brain.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>
#include <fstream>
#include <algorithm>

#include "nplio.h"
#include "mrimage.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "utility.h"
#include "basic_plot.h"

using namespace npl;
using namespace std;

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
vector<double> createRandAct(double lambda, double t0, double tf);

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
 *
 * @return Timeseries for the given spike times smoothed by the BOLD signal
 */
vector<double> sampleBOLD(const vector<double>& times, double tlen, double tr);

/**
 * @brief Simulate an fMRI with regions specified by labelmap, constrained to
 * GM, (set by gm), using the spike timing for each of the labels (act), with
 * total simulation time tlen, and sampling rate tr
 *
 * @param labelmap Activation regions (regions > 0). Label numbers (1-M) should
 * match the number of rows (M) in act.
 * @param gm Gray matter probability, overall signal is weighted by this.
 * @param act Activation spike timings
 * @param sd Standard deviation of smoothing to apply to individual labels to
 * create a more realistic mixing of activation patterns
 * @param tlen Number of volumes in the output
 * @param tr Sampling time of the output
 * @param plotfile Create a plot of the time-courses in the given file
 * (if not empty)
 * @param hdtr TR in highres BOLD simulation
 * @param learn The weight of the current point, the previous moving average
 * value is weighted (1-learn), setting learn to 1 removes all habituation
 * This limits the effect of habituation.
 *
 * @return simulated fMRI
 */
ptr<MRImage> simulate(ptr<MRImage> labelmap, ptr<MRImage> gm,
		const vector<vector<double>>& act, double sd,
		size_t tlen, double tr, string plotfile, double hdtr, double learn);

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Creates a 4D fMRI image where activations throughout "
			"the brain are given by spikes convolved with the hemodynamic "
			"response function, please noise. ", ' ', __version__ );

	TCLAP::ValueArg<string> a_anatomy("g", "greymatter", "Greymatter "
			"probability image. Signal is weited by GM prob.",
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_regions("r", "region-map", "Labelmap where label 1 "
			"gets the signal from the first column of activation table, label "
			"2 from the second and so on. ", false, "", "*.nii.gz");
	TCLAP::SwitchArg a_randregions("R", "region-rand", "Randomize regions "
			"by smoothing a gaussian random field, then arbitrarily assigning "
			"to fit the needed number of regions");
	cmd.xorAdd(a_regions, a_randregions);

	TCLAP::ValueArg<string> a_actfile("a", "act-file", "Activation spike "
			"train for each label. Lines (1-... ) correspond to labels. "
			"Spike times dhould be separated by commans or spaces.",
			false, "", "*.csv");
	TCLAP::MultiArg<double> a_actrand("A", "act-rand", "Create a random "
			"spike activation profile with poisson distribution. ", false,
			"lambda");
	cmd.xorAdd(a_actrand, a_actfile);

	TCLAP::ValueArg<string> a_plot("p", "plot", "Plot the activations in the "
			"specified file", false, "", "*.svd", cmd);

	TCLAP::ValueArg<double> a_smooth("s", "smooth", "Smoothing standard "
			"deviation to apply to each activation region prior to applying "
			"activation. The mixing should create more realistic 'mixed' "
			"activations. This won't create significant out-of-GM activation "
			"due to the limits of anatomy.", false, 5, "mm", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<size_t> a_tlen("t", "times", "Number of output timepoints "
			"in output image.", false, 1024, "n", cmd);
	TCLAP::ValueArg<double> a_tr("T", "tr", "Output image TR (sampling period).",
			false, 1.5, "sec", cmd);
	TCLAP::ValueArg<double> a_hdres("", "sim-dt", "Sampling rate during "
			"during BOLD simulation. If you are getting NAN's then "
			"decrease this.", false, 0.01, "sec", cmd);
	TCLAP::ValueArg<double> a_learn("", "habrate", "Habituation rate/learning "
			"rate for the exponential moving average that is used to estimate "
			"the habituation factor in BOLD simulation. This is the weight "
			"given to the most recent sample, the current MA is weighted (1-r).",
			false, 0.1, "ratio", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	// read regions
	cerr << "Reading Graymatter Probability...";
	auto gmprob = readMRImage(a_anatomy.getValue());
	cerr << "Done\n";

	vector<vector<double>> activate;
	if(a_actrand.isSet()) {
		cerr << "Simulating Random Activations: ";
		for(size_t ii=0; ii<a_actrand.getValue().size(); ii++) {
			cerr << "lambda "<<a_actrand.getValue()[ii]<<", ";
			activate.push_back(createRandAct(a_actrand.getValue()[ii], -15,
					a_tlen.getValue()*a_tr.getValue()));

		}
		ofstream ofs("randact.csv");
		for(size_t ii=0; ii<activate.size(); ii++) {
			if(ii != 0) ofs << "\n";
			for(size_t jj=0; jj<activate[ii].size(); jj++){
				if(jj != 0) ofs << ",";
				ofs << activate[ii][jj];
			}
		}
		ofs<<"\n";
		cerr << "Done\n";
	} else if(a_actfile.isSet()) {
		cerr << "Reading Activation File...";
		activate = readNumericCSV(a_actfile.getValue());
		cerr << "Done\n";
	}

	ptr<MRImage> labelmap;
	if(a_regions.isSet()) {
		cerr << "Reading Region Map...";
		labelmap = readMRImage(a_regions.getValue());
		cerr << "Done\n";
	} else if(a_randregions.isSet()) {
		cerr << "Simulating Region Map...";
		labelmap = dPtrCast<MRImage>(createRandLabels(gmprob, activate.size(), 5));
		cerr << "Done\n";
	}

	cerr << "Simulating fMRI...";
	auto out = simulate(labelmap, gmprob, activate, a_smooth.getValue(),
			a_tlen.getValue(), a_tr.getValue(), a_plot.getValue(),
			a_hdres.getValue(), a_learn.getValue());
	cerr << "Done\n";

	if(a_out.isSet())
		out->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
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

/**
 * @brief Simulate an fMRI with regions specified by labelmap, constrained to
 * GM, (set by gm), using the spike timing for each of the labels (act), with
 * total simulation time tlen, and sampling rate tr
 *
 * @param labelmap Activation regions (regions > 0). Label numbers (1-M) should
 * match the number of rows (M) in act.
 * @param gm Gray matter probability, overall signal is weighted by this.
 * @param act Activation spike timings
 * @param sd Standard deviation of smoothing to apply to individual labels to
 * create a more realistic mixing of activation patterns
 * @param tlen Number of volumes in the output
 * @param tr Sampling time of the output
 * @param hdtr TR in highres simulation dataset
 * @param learn The weight of the current point, the previous moving average
 * value is weighted (1-learn), setting learn to 1 removes all habituation
 * This limits the effect of habituation.
 *
 * @return simulated fMRI
 */
ptr<MRImage> simulate(ptr<MRImage> labelmap, ptr<MRImage> gm,
		const vector<vector<double>>& act, double sd,
		size_t tlen, double tr, string plotfile, double hdtr, double learn)
{
	if(labelmap->ndim() != 3)
		throw INVALID_ARGUMENT("Non-3D Image Provided as Labelmap");
	if(gm->ndim() != 3)
		throw INVALID_ARGUMENT("Non-3D Image Provided for GM Probability Map");
	if(!labelmap->matchingOrient(gm, true, true))
		throw INVALID_ARGUMENT("Greymatter and Labelmaps provided do not "
				"have the same orientation!");

	size_t nreg = act.size(); // # of regions
	vector<vector<double>> design(nreg);

	/* create timeseries for each of the activation spike trains */
	Plotter plotter;
	for(size_t rr=0; rr<nreg; rr++) {
		cerr<<"Simulating timeseries "<<rr<<"...";
		design[rr] = sampleBOLD(act[rr], tr*tlen, tr, hdtr, learn);
		assert(design[rr].size() == tlen);

		if(!plotfile.empty())
			plotter.addArray(design[rr].size(), design[rr].data());

		cerr<<"Done, Mutual Information with Previous: ";
		for(size_t ii=0; ii<rr; ii++) {
			if(ii != 0) cerr << ",";
			cerr << correlation(tlen, design[rr].data(),
					design[ii].data());
//			cerr << mutualInformation(tlen, design[rr].data(),
//					design[ii].data(), std::sqrt(tlen));
		}
		cerr << endl;
	}

	// Write out plot
	if(!plotfile.empty()) plotter.write(plotfile);

	/* create 4D image where each volume is a label probability */
	vector<size_t> tmpsize(labelmap->dim(), labelmap->dim()+labelmap->ndim());
	tmpsize.push_back(nreg);
	auto prob = dPtrCast<MRImage>(labelmap->copyCast(tmpsize.size(),
				tmpsize.data(), FLOAT32));
	for(FlatIter<double> it(prob); !it.eof(); ++it)
		it.set(0);

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

	prob->write("prob_presmooth.nii.gz");

	/* smooth each of the 4D volumes */
	for(size_t dd=0; dd<3; dd++)
		gaussianSmooth1D(prob, dd, sd);

	/*
	 * for each pixel sum up the contribution of each timeseries then scale by
	 * greymatter probability
	 */
	tmpsize[3] = tlen;
	auto out = dPtrCast<MRImage>(labelmap->copyCast(tmpsize.size(),
				tmpsize.data(), FLOAT32));

	NDIter<double> git(gm);
	Vector3DIter<double> oit(out);
	for(pit.goBegin(); !pit.eof(); ++pit,++oit,++git){
		for(size_t tt=0; tt<tlen; tt++) {
			// Weight Sum of each regions value
			double v = 0;
			for(size_t rr=0; rr<nreg; rr++)
				v += design[rr][tt]*pit[rr];

			// Weigtht by GM Prob
			oit.set(tt, v*(*git));
		}
	}

	return out;
}


