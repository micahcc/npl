/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file fibCount.cpp Tool for counting fibers between regions.
 ******************************************************************************/

#include "version.h"
#include <tclap/CmdLine.h>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <set>

#include "graph.h"
#include "tracks.h"
#include "nplio.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using namespace npl;
using namespace std;

/**
 * @brief Reads a Mesh File and produces a KD-Tree of points, with integer
 * labels at each point
 *
 * @param filename
 *
 * @return
 */
size_t readLabelMesh(string filename, KDTree<3, 1, float, int64_t>* tree);

/**
 * @brief Reads a label file and produces a KD-Tree of points with an integer
 * label at each point
 *
 * @param filename
 *
 * @return
 */
std::set<int64_t> readLabelMap(string filename, KDTree<3, 1, float, int64_t>* tree);

/**
 * @brief Computes average scalar statistic along each tract then accumulates
 * the average statistic for all labels along the tract. This produces a
 * weighted average of the stastic between pairs of points
 *
 * @param tracts
 * @param map
 * @param mask
 */
//void computeTractAvg(TrackSet& tracts, ptr<MRImage> map, ptr<MRImage> mask);
//
//void createGraphs(const TrackSet& tracts,

/**
* @brief Computes and add scalars to tract dataset
*
* By default adds length, if faimg is defined then the scalars
* will have two-tuple, first with fiber length, second with average fa
*
* @param tracts Tract to modify and add scalars to
* @param interps list of Scalar Field Interpolators to average over
* @param maskLabels list of labels to average scalar fields over
* @param labelinter interpolator for labelmap
*/
vector<vector<double>> computeScalars(const TrackSet& tractData,
		const vector<ptr<MRImage>>& simgs);

/**
 * @brief Reduce each track to the longest masked arc
 *
 * @param mask mask image
 * @param tractData tracts (will be modified)
 */
void cropTracks(ptr<MRImage> mask, TrackSet* trackData, double lenthresh = 0);

/**
 * @brief Computes scalars for each edge in the graph. This is done by averaging
 * over each scalar (which is associated with a track) for all the tracks
 * connecting a pair of regions
 *
 * @param tractData Fiber tracks
 * @param labeltree Labelmap (used to determine what regions are connected by a
 * track)
 * @param scalars Scalars, one per track
 * @param labelToVertex Mapping from labels to vertices in the graphs
 * @param tdist distance to search in tree to see if a fiber is connected
 * @param cgraph Graph consisting of counts
 * @param lgraph Graph consisting of average length
 * @param sgraphs Graph consisting of averaged scalars
 */
void computePerEdgeScalars(const TrackSet& tractData,
		const KDTree<3,1,float,int64_t>& labeltree, const vector<vector<double>>& scalars,
		const std::map<int64_t, size_t>& labelToVertex, double tdist,
		Graph<size_t>* cgraph, Graph<double>* lgraph, vector<Graph<double>>* sgraphs);

int main(int argc, char * argv[])
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Calculates an adjacency matrix from fiber tract and "
		"either A) a mesh of points or B) a labelmap. This looks for the "
		"nearest label at the end of each tract then counts that as a"
		"connection between the ending of each tract. Note that either a"
		"labelmap or labelmesh is required.", ' ', __version__ );


	TCLAP::ValueArg<string> a_tracts("t", "tracts", "Input tracts that will "
			"be tabulated", true, "", "*.trk|*.dfc", cmd);
	TCLAP::ValueArg<string> a_trackref("r", "tract-ref", "*.dft files need a "
			"separate image to determine their orientation, provide that "
			"here", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_wmmask("w", "wm", "White matter mask. This is "
			"needed if you want to remove tracks (or sections of tracks) that "
			"are not inside the brain", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_labelmap("l", "labelmap", "Input label image to "
			"calculate connectivity over. Should be grey-matter only (ie "
			"no white matter).", true, "", "*.nii.gz", cmd);
//	TCLAP::ValueArg<string> a_mesh("m", "labelmesh", "Input mesh to calculate "
//			"connectivity over.", false, "", "*.dfs");
//	cmd.xorAdd(a_labelmap, a_mesh);
	TCLAP::ValueArg<string> a_mask("m", "mask", "Brain mask",
			true, "", "*.nii.gz");

	TCLAP::ValueArg<string> a_cgraph("c", "count-ofile", "Adjacency matrix "
			"containing fiber counts between pairs of regions.", false, "",
			"graph", cmd);
	TCLAP::MultiArg<string> a_fgraphs("f", "field-graph",
			"Adjacency matrix containing average field between pairs of "
			"regions. Repeat for multiple, number of args should match -F",
			false, "graph", cmd);
	TCLAP::ValueArg<string> a_lengraph("L", "length-ofile",
			"Adjacency matrix containing average length of fibers between "
			"pairs of regions.", false, "", "graph", cmd);
	TCLAP::ValueArg<double> a_lenthresh("T", "thresh",
			"Length threshold for fibers (ignore any fibers shorter [5]",
			false, 5, "mm", cmd);
	TCLAP::ValueArg<double> a_dist("d", "dist",
			"Distance from a label that is considered connected to a tract [5]",
			false, 5, "mm", cmd);

	TCLAP::MultiArg<string> a_scalars("F", "field",
			"Scalar image whose values will be averaged over the tracts "
			"connecting pairs of regions. This intended to calculate the "
			"average FA of tracts connecting two regions, but could be "
			"used for any scalar image. Must match number of -f", false,
			"*.nii.gz", cmd);


	cmd.parse(argc, argv);

	std::set<int64_t> labelset;
	std::map<int64_t, size_t> labelToVertex;
	KDTree<3, 1, float, int64_t> tree;
	size_t nlabel;
	{
//		if(a_mesh.isSet())
//			labelset = readLabelMesh(a_mesh.getValue(), &tree);
//		else if(a_labelmap.isSet())
		labelset = readLabelMap(a_labelmap.getValue(), &tree);
		nlabel = labelset.size();

		std::set<int64_t>::iterator it = labelset.begin();
		for(size_t ii=0; it != labelset.end(); ++it, ++ii)
			labelToVertex[*it] = ii;
	}

	// Compute the Attached Labels for each Tract
	cerr<<"Reading Tracts"<<endl;
	TrackSet tractData;
	if(a_trackref.isSet()) {
		tractData = readTracks(a_tracts.getValue(), a_trackref.getValue());
	} else {
		cerr<<"NOTE: Using "<<a_labelmap.getValue()<<" as track "
			"reference. This means the orientation and gridding should be "
			"identical to the image used to generate the original tracks (if "
			"you are using DFT)"<<endl;
		tractData = readTracks(a_tracts.getValue(), a_labelmap.getValue());
	}

	cerr<<"Done"<<endl;

	if(a_wmmask.isSet()) {
		cerr<<"Note: Cropping Tracks to Masked (label != 0) Region. "
			<<a_wmmask.getValue()<<"should be a white-matter mask"<<endl;
		ptr<MRImage> mask = readMRImage(a_wmmask.getValue());
		cropTracks(mask, &tractData, a_lenthresh.getValue());
		cerr<<"Done"<<endl;
	}

	cerr<<"Reading Scalars"<<endl;
	vector<ptr<MRImage>> simgs;
	for(auto fname : a_scalars.getValue())
		simgs.push_back(readMRImage(fname));
	cerr<<"Done"<<endl;

	cerr<<"Creating Output Graphs"<<endl;
	// create Graphs
	Graph<size_t> cgraph(nlabel);
	Graph<double> lengraph(nlabel);
	std::vector<Graph<double>> fgraphs(a_scalars.getValue().size());
	for(size_t ii=0; ii<fgraphs.size(); ii++)
		fgraphs[ii].init(nlabel);

	size_t ii=0;
	for(auto ll : labelset) {
		cgraph.name(ii) = to_string(ll);
		lengraph.name(ii) = to_string(ll);
		for(size_t gg=0; gg<fgraphs.size(); gg++)
			fgraphs[gg].name(ii) = to_string(ll);
		ii++;
	}
	cerr<<"Done"<<endl;

	// scalars[tract][point][scalar]
	vector<vector<double>> scalars;
	if(a_scalars.isSet()) {
		cerr<<"Computing per-track scalars"<<endl;
		scalars = computeScalars(tractData, simgs);
		cerr<<"Done"<<endl;
	}

	cerr<<"Averaging Scalars over Graphs"<<endl;
	computePerEdgeScalars(tractData, tree, scalars, labelToVertex,
			a_dist.getValue(), &cgraph, &lengraph, &fgraphs);
	cerr<<"Done"<<endl;

	if(a_cgraph.isSet())
		cgraph.save(a_cgraph.getValue());

	if(a_lengraph.isSet())
		lengraph.save(a_lengraph.getValue());

	for(size_t ii=0; ii<a_fgraphs.getValue().size() && ii<fgraphs.size(); ii++){
		fgraphs[ii].save(a_fgraphs.getValue()[ii]);
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return EXIT_SUCCESS;
}

template <typename F>
F distance(F a[3], F b[3])
{
	F tmp = 0;
	for(int ii = 0 ; ii < 3 ; ii++)
		tmp += pow(b[ii] - a[ii], 2);
	return sqrt(tmp);
}

template <typename T>
typename T::value_type distance(T a, T b)
{
	typename T::value_type tmp = 0;
	for(int ii = 0 ; ii < a.size(); ii++)
		tmp += pow(b[ii] - a[ii], 2);
	return sqrt(tmp);
}

/**
 * @brief Reduce each track to the longest masked arc
 *
 * @param mask mask image
 * @param tractData tracts (will be modified)
 */
void cropTracks(ptr<MRImage> mask, TrackSet* trackData, double lenthresh)
{
	NNInterp3DView<int64_t> minterp(mask);
	minterp.m_ras = true;
	std::array<float, 3> pt, ppt;
	std::list<std::array<float,3>> pts;
	for(size_t tt=0; tt<trackData->size(); tt++) {
		if((*trackData)[tt].empty()) continue;

		double maxlen = 0;
		double curlen = 0;
		int64_t maxbeg = -1;
		int64_t maxend = -1;
		int64_t curbeg = -1;

		// initial point
		pt = (*trackData)[tt][0];
		if(minterp(pt[0], pt[1], pt[2]) != 0)
			curbeg = 0;

		for(size_t pp=1; pp<(*trackData)[tt].size(); pp++) {
			// update current and previous points
			ppt = pt;
			pt = (*trackData)[tt][pp];

			// currently inside masked region
			if(curbeg != -1) {
				if(minterp(pt[0], pt[1], pt[2]) == 0) {
					// exit masked region
					if(curlen > maxlen) {
						maxbeg = curbeg;
						maxend = pp-1;
						maxlen = curlen;
					}
					curbeg = -1;
					curlen = 0;
				} else {
					// maintain inside
					curlen += distance(pt, ppt);
				}
			} else if(minterp(pt[0], pt[1], pt[2]) != 0) {
				// entering masked region
				curbeg = pp;
				curlen = 0;
			}
		}
		// get if we finish inside the brain
		if(curlen > maxlen) {
			maxbeg = curbeg;
			maxend = (*trackData)[tt].size()-1;
			maxlen = curlen;
		}

		vector<std::array<float,3>> keep;
		if(lenthresh > 0 && maxlen > lenthresh) {
			keep.resize(maxend+1-maxbeg);
			for(size_t ii=maxbeg; ii<=maxend; ii++)
				keep[ii-maxbeg] = (*trackData)[tt][ii];
		}
		std::swap((*trackData)[tt], keep);
	}
}

/**
* @brief Computes and add scalars to tract dataset. Each track will have
* a group of scalars (1 scalar per input image) in it
*
* By default adds length, if faimg is defined then the scalars
* will have two-tuple, first with fiber length, second with average fa
*
* @param tracts Tract to modify and add scalars to
* @param simgs Vector of scalar images
*/
vector<vector<double>> computeScalars(const TrackSet& tractData,
		const vector<ptr<MRImage>>& simgs)
{
	std::array<float,3> prevpt, pt;
	vector<double> sums(simgs.size());
	vector<vector<double>> outscalars(tractData.size());
	vector<LinInterp3DView<double>> interps(simgs.size());

	for(size_t ii=0; ii<simgs.size(); ii++) {
		interps[ii].setArray(simgs[ii]);
		interps[ii].m_ras = true;
	}

	for(size_t tt=0; tt<tractData.size(); ++tt) {
		// create scalars for tract
		double len = 0;
		std::fill(sums.begin(), sums.end(), 0);

		for(size_t pp=1; pp<tractData[tt].size(); ++pp) {
			// interpolate at point
			prevpt = tractData[tt][pp-1];
			pt = tractData[tt][pp];

			// compute step size
			double dlen = distance(pt, prevpt);
			len += dlen;

			// weight scalar by step size
			for(size_t ss=0; ss<sums.size(); ss++) {
				double v = interps[ss].get(pt[0], pt[1], pt[2]);
				sums[ss] += v*dlen;
			}
		}

		// set the scalars for the current track
		outscalars[tt].resize(sums.size());
		for(size_t ss=0; ss<sums.size(); ss++)
			outscalars[tt][ss] = sums[ss]/len;
	}

	return outscalars;
}

void computePerEdgeScalars(const TrackSet& trackData,
		const KDTree<3,1,float,int64_t>& labeltree,
		const vector<vector<double>>& scalars,
		const std::map<int64_t, size_t>& labelToVertex, double treed,
		Graph<size_t>* cgraph, Graph<double>* lgraph,
		vector<Graph<double>>* sgraphs)
{
	double stepsize = 1;
	std::array<float, 3> pt, ppt; // points
	std::unordered_set<int64_t> conlabels; // labels connected by track

	for(size_t ii=0; ii<cgraph->nodes(); ii++) {
		for(size_t jj=0; jj<cgraph->nodes(); jj++) {
			(*cgraph)(ii,jj)=0;
			(*lgraph)(ii,jj)=0;
			for(size_t kk=0; kk<sgraphs->size(); kk++)
				(*sgraphs)[kk](ii,jj) = 0;
		}
	}

	// Iterate through tracks finding connections tracks establish, then
	// summing up properties of tracks connecting regions
	for(size_t tt=0; tt<trackData.size(); tt++) {
		if(trackData[tt].empty()) continue;

		// iterate through points to find all connections made by track and
		double len = 0;
		double udist = stepsize;
		pt = trackData[tt][0];
		conlabels.clear();
		for(size_t pp=1; pp<trackData[tt].size(); pp++) {
			ppt = pt;
			pt = trackData[tt][pp];
			double dlen = distance(pt, ppt);
			len += dlen;

			// udist is distance since last update of labels, just so that we
			// don't waste too much time doing tree-searches
			udist -= dlen;
			if(udist < 0) {
				auto result = labeltree.withindist(pt.size(), pt.data(), treed);
				for(const auto& node : result)
					conlabels.insert(node->m_data[0]);
				udist = stepsize;
			}
		}

		// assign length, count to pairs
		for(auto it1 = conlabels.begin(); it1 != conlabels.end(); ++it1) {
			size_t ii = labelToVertex.at(*it1);
			auto it2 = it1;
			for(; it2 != conlabels.end(); ++it2) {
				size_t jj = labelToVertex.at(*it2);
				// ii->jj
				(*lgraph)(ii,jj) += len;
				(*cgraph)(ii,jj)++;
				for(size_t kk=0; kk<sgraphs->size(); kk++)
					(*sgraphs)[kk](ii,jj) += scalars[tt][kk];
				// jj->ii
				(*lgraph)(jj,ii) += len;
				(*cgraph)(jj,ii)++;
				for(size_t kk=0; kk<sgraphs->size(); kk++)
					(*sgraphs)[kk](jj,ii) += scalars[tt][kk];
			}
		}
	}

	// now divide by the counts
	for(size_t ii=0; ii<cgraph->nodes(); ii++) {
		for(size_t jj=0; jj<cgraph->nodes(); jj++) {
			(*lgraph)(ii,jj) /= (*cgraph)(ii,jj);
			for(size_t kk=0; kk<sgraphs->size(); kk++)
				(*sgraphs)[kk](ii,jj) /= (*cgraph)(ii,jj);
		}
	}
}

/**
 * @brief Reads a Mesh File and produces a KD-Tree of points, with integer
 * labels at each point
 *
 * @param filename
 *
 * @return
 */
size_t readLabelMesh(string filename, KDTree<3,1,float,int64_t>* tree)
{
	(void)filename;
	(void)tree;
	throw std::invalid_argument("readLabelMesh not yet implemented");
	return 0;
};

/**
 * @brief Reads a label file and produces a KD-Tree of points with an integer
 * label at each point
 *
 * @param filename
 *
 * @return
 */
std::set<int64_t> readLabelMap(string filename, KDTree<3,1,float,int64_t>* tree)
{
	std::set<int64_t> alllabels;
	auto labelmap = readMRImage(filename);
	tree->clear();
	double pt[3];
	vector<float> fpt(3);
	vector<int64_t> label(1);
	for(NDIter<int> it(labelmap); !it.eof(); ++it) {
		if(*it != 0) {
			it.index(3, pt);
			labelmap->indexToPoint(3, pt, pt);
			for(size_t ii=0; ii<3; ii++)
				fpt[ii] = pt[ii];
			label[0] = *it;
			alllabels.insert(*it);
			tree->insert(fpt, label);
		}
	}
	tree->build();
	return alllabels;
}

