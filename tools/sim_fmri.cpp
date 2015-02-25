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
 * @param hdtr TR in highres BOLD simulation
 * @param learn The weight of the current point, the previous moving average
 * value is weighted (1-learn), setting learn to 1 removes all habituation
 * This limits the effect of habituation.
 *
 * @return Timeseries for the given spike times smoothed by the BOLD signal
 */
vector<double> sampleBOLD(const vector<double>& times, double tlen, double tr,
		double hdtr, double learn);

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
			"response function. There are three basic modes: "
			"a) One that inputs a map of activation levels (-B/--bmap), "
			"b) One that provide a labelmap and grey matter map (-r/--region-map) "
			"and c) one where you specificy a grey matter map and get random labels "
			"(-R/--region-rand). Time-series is handled separately, although the "
			"number of regions in -r labelmap or volumes in the -B map must match "
			"the number of generated regions. To input a signal use -a, and reach "
			"column will be a separate signal or use -A lambda -A lambda ...). "
			" A t2* image (-m) is optional, but make the output look more like "
			"an fMRI. ", ' ', __version__ );

	TCLAP::ValueArg<string> a_anatomy("g", "greymatter", "Greymatter "
			"probability image. Signal is weighted by GM prob.",
			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_t2star("m", "t2-map", "T2* Map to overlay the"
			"signal onto. This is useful, for instance, if you intend to "
			"simulate motion.", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_regions("r", "region-map", "Labelmap where label 1 "
			"gets the signal from the first column of activation table, label "
			"2 from the second and so on. Note that if both -R and -r are set "
			"then this becomes an OUTPUT file to write the new regions to",
			false, "", "*.nii.gz");
	TCLAP::SwitchArg a_randregions("R", "region-rand", "Randomize regions "
			"by smoothing a gaussian random field, then arbitrarily assigning "
			"to fit the needed number of regions");
	TCLAP::ValueArg<string> a_bmap("B", "bmap",
			"Instead of a region map, provide a map of weights at each voxel. "
			"This must be a 4D image with each volume representing weight for "
			"a particular signal", false, "", "*.nii.gz");
	vector<TCLAP::Arg*> opts({&a_regions, &a_randregions, &a_bmap});
	cmd.xorAdd(opts);

	TCLAP::ValueArg<string> a_actfile("a", "act-file", "Activation spike "
			"train for each label. Lines (1-... ) correspond to labels. "
			"Spike times dhould be separated by commans or spaces.",
			false, "", "*.csv");
	TCLAP::MultiArg<double> a_actrand("A", "act-rand", "Create a random "
			"spike activation profile with poisson distribution. ", false,
			"lambda");
	cmd.xorAdd(a_actrand, a_actfile);

	TCLAP::ValueArg<string> a_oactfile("", "out-spikes", "Output spike "
			"timing. This may be used as input later (-a). Lines "
			"(1-... ) correspond to labels. "
			"Spike times dhould be separated by commans or spaces.",
			false, "", "*.csv", cmd);
	TCLAP::ValueArg<string> a_timeseries("t", "timeseries", "Write BOLD "
			"timeseries' into CSV file, for instance to do regression on later.",
			false, "", "*csv", cmd);

	TCLAP::ValueArg<string> a_probmap("p", "probmaps",
			"Write regional probability maps", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_plot("", "aplot", "Plot the activations in the "
			"specified file", false, "", "*.svg", cmd);

	TCLAP::ValueArg<double> a_noise("n", "noise-sd", "Noise standard "
			"deviation. Note the true signal tends to be < 0.05", false,
			0.01, "ratio", cmd);

	TCLAP::ValueArg<double> a_smooth("s", "smooth", "Smoothing standard "
			"deviation to apply to each activation region prior to applying "
			"activation. The mixing should create more realistic 'mixed' "
			"activations. This won't create significant out-of-GM activation "
			"due to the limits of anatomy.", false, 1, "mm", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<size_t> a_tlen("", "tdim", "Number of output timepoints "
			"in output image.", false, 1024, "n", cmd);
	TCLAP::ValueArg<double> a_tr("", "tr", "Output image TR (sampling period).",
			false, 1.5, "sec", cmd);
	TCLAP::ValueArg<double> a_hdres("", "sim-dt", "Sampling rate during "
			"during BOLD simulation. If you are getting NAN's then "
			"decrease this.", false, 0.01, "sec", cmd);
	TCLAP::ValueArg<double> a_learn("", "habrate", "Habituation rate/learning "
			"rate for the exponential moving average that is used to estimate "
			"the habituation factor in BOLD simulation. This is the weight "
			"given to the most recent sample, the current MA is weighted (1-r).",
			false, 0.05, "ratio", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	// read regions
	vector<vector<double>> activate;
	if(a_actrand.isSet()) {
		cerr << "Simulating Random Activations: ";
		for(size_t ii=0; ii<a_actrand.getValue().size(); ii++) {
			cerr << "lambda "<<a_actrand.getValue()[ii]<<", ";
			activate.push_back(createRandAct(a_actrand.getValue()[ii], -15,
					a_tlen.getValue()*a_tr.getValue()));

		}

		if(a_oactfile.isSet()) {
			ofstream ofs(a_oactfile.getValue().c_str());
			if(!ofs.is_open())
				throw INVALID_ARGUMENT("Could not open "+a_oactfile.getValue());
			for(size_t ii=0; ii<activate.size(); ii++) {
				if(ii != 0) ofs << "\n";
				for(size_t jj=0; jj<activate[ii].size(); jj++){
					if(jj != 0) ofs << ",";
					ofs << activate[ii][jj];
				}
			}
			ofs<<"\n";
		}
		cerr << "Done\n";
	} else if(a_actfile.isSet()) {
		cerr << "Reading Activation File...";
		activate = readNumericCSV(a_actfile.getValue());
		cerr << "Done\n";
	}

	double sd = a_smooth.getValue();
	double learn = a_learn.getValue();
	double hdtr = a_hdres.getValue();
	double tr = a_tr.getValue();
	size_t tlen = a_tlen.getValue();
	size_t nreg = activate.size(); // # of regions
	vector<vector<double>> design(nreg);

	/* create timeseries for each of the activation spike trains */
	Plotter plotter;
	cerr<<"Simulating timeseries...Correlations:\n";
	for(size_t rr=0; rr<nreg; rr++) {
		design[rr] = sampleBOLD(activate[rr], tr*tlen, tr, hdtr, learn);
		assert(design[rr].size() == tlen);

		if(a_plot.isSet())
			plotter.addArray(design[rr].size(), design[rr].data());

		for(size_t ii=0; ii<=rr; ii++) {
			if(ii != 0) cerr << ",";
			cerr << correlation(tlen, design[rr].data(),
					design[ii].data());
		}
		cerr << endl;
	}
	cerr<<"Done\n";

	// Write out plot
	if(a_plot.isSet()) plotter.write(a_plot.getValue());

	/* Write Out BOLD Timeseries */
	if(a_timeseries.isSet()) {
		ofstream ofs(a_timeseries.getValue());
		if(!ofs.is_open()) {
			cerr<<"Error, could not open "<<a_timeseries.getValue()
				<<" for writing"<<endl;
		} else {
			for(size_t rr=0; rr<tlen; rr++) {
				if(rr != 0) ofs << "\n";
				for(size_t cc=0; cc<nreg; cc++) {
					if(cc != 0) ofs << ",";
					ofs << design[cc][rr];
				}
			}
			ofs << "\n";
		}
	}

	vector<size_t> tmpsize(4);
	tmpsize[3] = nreg;
	ptr<MRImage> t2star, gmprob, labelmap, bmap;
	if(a_randregions.isSet() || a_regions.isSet()) {
		cerr << "Reading Graymatter Probability...";
		auto gmprob = readMRImage(a_anatomy.getValue());
		cerr << "Done\n";
		if(a_randregions.isSet()) {
			cerr << "Simulating Region Map...";
			labelmap = dPtrCast<MRImage>(createRandLabels(gmprob, activate.size(), 5));
			cerr << "Done\n";
			if(a_regions.isSet()) {
				cerr << "Provided both rand regions (-R) and region file (-r), "
					"so writing to "<<a_regions.getValue()<<"(from -r)"<<endl;
				labelmap->write(a_regions.getValue());
			}
		} else if(a_regions.isSet()) {
			cerr << "Reading Region Map...";
			labelmap = readMRImage(a_regions.getValue());
			cerr << "Done\n";
		}

		/* Check Images for Matching Orientation */
		if(labelmap->ndim() != 3)
			throw INVALID_ARGUMENT("Non-3D Image Provided as Labelmap");
		if(gmprob->ndim() != 3)
			throw INVALID_ARGUMENT("Non-3D Image Provided for GM Probability Map");
		if(!labelmap->matchingOrient(gmprob, true, true))
			throw INVALID_ARGUMENT("Greymatter and Labelmaps provided do not "
					"have the same orientation!");

		/* create 4D image where each volume is a label probability */
		for(size_t ii=0; ii<3; ii++)
			tmpsize[ii] = labelmap->dim(ii);
		bmap = dPtrCast<MRImage>(labelmap->copyCast(tmpsize.size(),
					tmpsize.data(), FLOAT32));
		for(FlatIter<double> it(bmap); !it.eof(); ++it)
			it.set(0);

		cerr << "Creating indivudal probability maps...";
		Vector3DIter<double> pit(bmap);
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
			gaussianSmooth1D(bmap, dd, sd);
		cerr << "Done\n";

		cerr << "Merging Probability Map with GM Probability"<<endl;
		NDIter<double> git(gmprob);
		for(pit.goBegin(); !pit.eof(); ++pit,++git){
			// Weight Sum of each regions value
			for(size_t rr=0; rr<nreg; rr++)
				pit.set(rr, pit[rr]*(*git));
		}

		if(a_probmap.isSet())
			bmap->write(a_probmap.getValue());
	} else if(a_bmap.isSet()) {
		cerr<<"Using input bmap"<<endl;
		bmap = readMRImage(a_bmap.getValue());
		cerr<<"Done Reading"<<endl;
		if(bmap->tlen() != nreg) {
			cerr << "Input map must have same number of regions (volums) "
				"as simulated  timeseries ("<<bmap->tlen()<<" in --bmap image "
				"versus "<<nreg<<" simulated regions) "<<endl;
			return -1;
		}
	}

	if(a_t2star.isSet())
		t2star = readMRImage(a_t2star.getValue());

	/*
	 * for each pixel sum up the contribution of each timeseries then scale by
	 * greymatter probability
	 */
	for(size_t ii=0; ii<3; ii++)
		tmpsize[ii] = bmap->dim(ii);
	tmpsize[3] = tlen;
	auto out = dPtrCast<MRImage>(bmap->copyCast(tmpsize.size(),
				tmpsize.data(), FLOAT32));

	cerr << "Merging activation maps..."<<endl;
	Vector3DIter<double> oit(out);
	for(Vector3DIter<double> bit(bmap); !bit.eof(); ++bit,++oit){
		for(size_t tt=0; tt<tlen; tt++) {
			// Weight Sum of each regions value
			double v = 0;
			for(size_t rr=0; rr<nreg; rr++)
				v += design[rr][tt]*bit[rr];

			oit.set(tt, v);
		}
	}
	cerr << "Done"<<endl;

	cerr<<"Adding Noise..."<<endl;
	std::random_device rd;
	std::default_random_engine rng(rd());
	std::normal_distribution<double> dist(0, a_noise.getValue());
	for(oit.goBegin(); !oit.eof(); ++oit) {
		for(size_t tt=0; tt<tlen; tt++)
			oit.set(tt, oit[tt]+dist(rng));
	}
	cerr<<"Done"<<endl;

	if(a_t2star.isSet()) {
		cerr << "Overlaying on T2* Image...";
		NDIter<double> tit(t2star);
		for(oit.goBegin(); !tit.eof(); ++tit, ++oit) {
			for(size_t tt=0; tt<tlen; tt++)
				oit.set(tt, (1+oit[tt])*tit.get());
		}
		cerr << "Done" <<endl;
	}

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
