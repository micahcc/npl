/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file fibCount.cpp Tool for counting fibers between regions.
 ******************************************************************************/

#include "version.h"
#include <tclap/CmdLine.h>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "tracts.h"
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
KDTree<3, 1, float, int> readLabelMesh(string filename);

/**
 * @brief Reads a label file and produces a KD-Tree of points with an integer
 * label at each point
 *
 * @param filename
 *
 * @return
 */
KDTree<3, 1, float, int> readLabelMap(string filename);

/**
 * @brief Reads tracts into a vector of tracts, where each tract is a vector of
 * float[3].
 *
 * @param filename
 *
 * @return
 */
TractSet readTracts(string filename);

/**
 * @brief Computes average scalar statistic along each tract then accumulates
 * the average statistic for all labels along the tract. This produces a
 * weighted average of the stastic between pairs of points
 *
 * @param tracts
 * @param map
 * @param mask
 */
void computeTractAvg(TractSet& tracts, ptr<MRImage> map, ptr<MRImage> mask);

void createGraphs(const TractSet& tracts,

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

	TCLAP::ValueArg<string> a_labelmap("l", "labelmap", "Input label image to "
			"calculate connectivity over.", false, "", "*.nii.gz");
	TCLAP::ValueArg<string> a_mesh("m", "labelmesh", "Input mesh to calculate "
			"connectivity over.", false, "", "*.dfs");
	cmd.xorAdd(a_labelmap, a_mesh);

	TCLAP::ValueArg<string> a_countfile("c", "count-ofile", "Adjacency matrix "
			"containing fiber counts between pairs of regions.", false, "",
			"graph", cmd);
	TCLAP::MultiArg<string> a_fafile("f", "field-graph",
			"Adjacency matrix containing average field between pairs of "
			"regions. Repeat for multiple, number of args should match -F",
			false, "", "graph", cmd);

	TCLAP::MultiArg<string> a_scalars("F", "field",
			"Scalar image whose values will be averaged over the tracts "
			"connecting pairs of regions. This intended to calculate the "
			"average FA of tracts connecting two regions, but could be "
			"used for any scalar image. Must match number of -f", false, "",
			"*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_scalarmasks("M", "field-mask",
			"Mask images to match scalar fields. "
			"If this argument is provided more than once, then the number "
			"should match the number of inputs for -F, and each field will "
			"be masked with its own set of labels.", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_lengthfile("L", "length-ofile",
			"Adjacency matrix containing average length of fibers between "
			"pairs of regions.", false, "", "graph", cmd);

	TCLAP::ValueArg<double> a_radius("r", "fiber-radius",
				"Maximum distance from a fiber for a fiber to be considered "
				"to be passing through the region. (And thus connected to other"
				" regions long the fiber. ", false, 5, "mm", cmd );

	TCLAP::SwitchArg a_endcount("e", "endcount", "Use the endpoints"
				" when counting fibers (rather than connectiving all regions "
				"along each fibers' length", cmd);


	cmd.parse(argc, argv);

	size_t nlabel;
	KDTree<3, 1, float, int> tree;
	if(a_mesh.isSet()) {
		nlabel = readLabelMesh(a_mesh.getValue(), tree);
	} else if(a_labelmap.isSet()) {
		nlabel = readLabelMap(a_labelmap.getValue(), tree);
	}

	// Compute the Attached Labels for each Tract
	TractSet tractData = readTracts(a_tracts.getValue());

	// Compute Connections, note it might be worth storing the position of
	// where regions fall on connecting fibers then limiting average FA only to
	// the values that fall between connections.
	vector<vector<int>> conns;
	if(a_endcount.isSet())
		conns = computeEndConnections(tractData, tree, a_radius.getValue());
	else
		conns = computeConnections(tractData, tree, a_radius.getValue());

	// Compute Average Statistics for each tract then average that average over
	// all tracts connecting regions
	for(size_t ii=0; ii<a_scalars.size(); ii++) {
		ptr<MRImage> simg = readMRImage(a_scalars.getValue()[ii]);
		ptr<MRImage> smask = NULL;
		if(ii < a_scalarmasks.size())
			smask = readMRImage(a_scalarmasks.getValue()[ii]);

		if(ii < a_fafile.size()) {
			auto out = computeTractAvg(tractData, conns, simg, smask);
			out->save(a_fafile.getValue()[ii]);
		}
	}

	if(a_countfile.isSet()) {
		cerr << "Computing Count" << endl;
		Graph<double> count(nlabel);
		for(size_t ii=0; ii<nlabel; ii++) {
			for(size_t jj=0; jj<nlabel; jj++) {
				count(ii, jj) = 0;
			}
		}

		for(size_t tt=0; tt<conns.size(); tt++) {
			for(size_t ll=0; ll<cons[tt].size(); ++ll){
				for(size_t kk=ll; kk<cons[tt].size(); ++kk) {
					count(conns[tt][ll], conns[tt][kk])++;
				}
			}
		}
		count.save(a_countfile.getValue());
	}

	if(a_lengthfile->count > 0) {
		cerr << "Computing Length" << endl;
		auto lengths = computeLengthAvg(tractData, conns);
		lengths->save(a_lengthfile.getValue());
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
void computeScalars(vtkSmartPointer<vtkPolyData> tractData,
			const list<FInterp::Pointer>& interps,
			const list<set<int>>& maskLabels,
			const LInterp::Pointer labelinterp = NULL)
{
	assert(maskLabels.size() == 1 || maskLabels.size() == interps.size() ||
				!labelinterp);

    vtkIdType npts;
    vtkIdType* ids;
	double pt1[3];
	double pt2[3];
	FImageT::PointType fpt;

	size_t numfield = interps.size();

	//only iterate through the sets of labels if we have multiple
	bool iterateLabels = (numfield == maskLabels.size());
	vector<double> fieldPtCount(numfield);

	//create the scalar output, (1 value for length + 1 for each scalar field)
	vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
	scalars->SetNumberOfComponents(1 + numfield);

	//to calculate average along the length we will multiply
	//the length*value for each segment:
	//v[0]*.5*l + v[1]*.5*l
	float* tuple = new float[1 + numfield];
	float* pfield = new float[numfield];

	// set up interpolators for the input scalar fields
	list<FInterp::Pointer>::const_iterator interp_it;
	list<set<int>>::const_iterator label_it;

    /*
	 *  iterate through fibers, calculating scalar fields and length of each
	 */
	cerr << "Initializing Traversal" << endl;
    vtkSmartPointer<vtkCellArray> tracts = tractData->GetLines();
    tracts->InitTraversal();
    while(tracts->GetNextCell(npts, ids)) {
		//initialize point 0
		tractData->GetPoint(ids[0], pt1);
		fpt[0] = -pt1[0];
		fpt[1] = -pt1[1];
		fpt[2] = pt1[2];

		//zero counts
		fill(fieldPtCount.begin(), fieldPtCount.end(), 0);
		for(unsigned int ii = 0 ; ii < numfield; ii++)
			pfield[ii] = NAN;
		for(unsigned int ii = 0 ; ii < numfield+1; ii++)
			tuple[ii] = 0;

		//calculate scalars for point 0
		interp_it = interps.begin() ;
		label_it = maskLabels.begin() ;
		for(int jj = 0; interp_it != interps.end(); jj++, interp_it++) {
			bool valid_field = (*interp_it)->IsInsideBuffer(fpt);

			//valid if we have no labelmap OR we have a labelmap, the point is
			//in the image, and the set matching the current field contains the label
			bool valid_label = (!labelinterp ||
						(labelinterp && labelinterp->IsInsideBuffer(fpt) &&
						 (*label_it).count(labelinterp->Evaluate(fpt))));

			if(valid_field && valid_label)
				pfield[jj] = (*interp_it)->Evaluate(fpt);

			if(iterateLabels)
				label_it++;
		}

		//calculate scalars at each point, and add up length
		for(int ii = 1 ; ii < npts ; ii++) {
			tractData->GetPoint(ids[ii], pt2);

			//evaluate FA, at point in RAS
			fpt[0] = -pt2[1];
			fpt[1] = -pt2[2];
			fpt[2] = pt2[0];

			//sum length
			double dist = distance(pt1, pt2);
			tuple[0] += dist;

			label_it = maskLabels.begin() ;
			interp_it = interps.begin() ;
			for(int jj = 0; interp_it != interps.end(); jj++, interp_it++) {
				bool valid_field = (*interp_it)->IsInsideBuffer(fpt);

				//valid if we have no labelmap OR we have a labelmap, the point is
				//in the image, and the set matching the current field contains the label
				bool valid_label = (!labelinterp ||
						(labelinterp && labelinterp->IsInsideBuffer(fpt) &&
						 (*label_it).count(labelinterp->Evaluate(fpt))));

				double s = 0;

				//sum for first half of segment (if its valid)
				if(!isnan(pfield[jj])) {
					s += dist*.5*pfield[jj];
					fieldPtCount[jj] += dist*.5;
				}

				//sum for second half of segment (if its valid)
				if(valid_field && valid_label) {
					fieldPtCount[jj] += dist*.5;
					s += dist*.5*(*interp_it)->Evaluate(fpt);

					//set pfield as valid for next point
					pfield[jj] = (*interp_it)->Evaluate(fpt);
				} else {
					//set pfield as invalid for next point
					pfield[jj] = NAN;
				}

				tuple[jj+1] += s;

				if(iterateLabels)
					label_it++;
			}

			//step to next point and make pt1 previous
			copy(pt2, pt2+3, pt1);
		}

		//average over length, but only for the length within the scalars label
		//groups
		for(unsigned int jj = 0; jj < numfield; jj++) {
			*olog << "Labeled Length: " << fieldPtCount[jj] << " : "
						<< tuple[jj+1] << " : "
						<< tuple[jj+1]/fieldPtCount[jj] << endl;
			tuple[jj+1] /= fieldPtCount[jj];
		}

		scalars->InsertNextTupleValue(tuple);
    }

	delete[] tuple;

	tractData->GetCellData()->SetScalars(scalars);
}

/**
* @brief Computes graphs of counts, fiber length, variance of fiber length and FA
* 			but unlike createGraphs adds connectivity for any region encountered
* 			along the fiber.
*
* @param tractData Input tracts to count over
* @param labelTree Labelmap Tree to lookup values from
* @param radius distance from fiber to look for connected labels
* @param minlength minimum acceptable fiber length
* @param labelLookup Label to index lookup
* @param ographs Output graphs (0 - count, 1 - length, 2 ... - data scalar image)
*
* @return
*/
int createGraphsInclusive(vtkSmartPointer<vtkPolyData> tractData,
			vtkSmartPointer<TreeT> labelTree, double radius, double minlength,
			const map<int, int>& labelLookup, vector<Graph*>& ographs)
{
	vtkSmartPointer<vtkIntArray> pointLabels = vtkIntArray::SafeDownCast(
				vtkPolyData::SafeDownCast(labelTree->GetDataSet())->
				GetPointData()->GetScalars());
	vtkSmartPointer<vtkFloatArray> tractStats = vtkFloatArray::SafeDownCast(
				tractData->GetCellData()->GetScalars());

    vtkIdType npts;
    vtkIdType* ids;
	vtkSmartPointer<vtkIdList> idlist = vtkSmartPointer<vtkIdList>::New();
	int i1, i2; //index 1, index 2, corresponding to label 1 and 2
	int l1; //label value
	double pt1[3];
	double pt2[3];
	Graph::iterator itc; //count graph iterator
	Graph::iterator its; //count graph iterator
	int nlabels = labelLookup.size();
	map<int, int>::const_iterator mit;
	double tmp;
	map<int, int> conlabels; //connected labels, map to indices in graph

	//store counts for each metric
	vector<Graph*> counts(ographs.size()-1);
	for(unsigned int ii = 0 ; ii < ographs.size()-1 ; ii++)
		counts[ii] = new Graph(nlabels, false, true);

	/* Initialize Graphs */

	//allocate
	for(unsigned int ii = 0 ; ii < ographs.size() ; ii++)
		ographs[ii] = new Graph(nlabels, false, true);

	//labels
	mit = labelLookup.begin();
	for(int ii = 0 ; ii < nlabels; ii++, mit++) {
		for(unsigned int jj = 0; jj < ographs.size() ; jj++)
			ographs[jj]->setLabel(ii, mit->first);
	}

	cerr << "initializating traversal" << endl;
	/*
	 * for each fiber, map the start/end labels
	 */
	int match_label = 0; //note that GetTraversal Location goes by points
	int match_all = 0; //note that GetTraversal Location goes by points
	int cellnum = -1; //note that GetTraversal Location goes by points
    vtkSmartPointer<vtkCellArray> tracts = tractData->GetLines();
    tracts->InitTraversal();
    while(tracts->GetNextCell(npts, ids)) {
		cellnum++;

		//travel down the fiber for looking the matches
		//add matches to set
		tractData->GetPoint(ids[0], pt1);
		conlabels.clear();
		for(int ii = 0; ii < npts; ii++) {
			tractData->GetPoint(ids[ii], pt2);

			idlist->Reset();
        	labelTree->FindPointsWithinRadius(radius, pt2, idlist);
			for(int jj = 0 ; jj < idlist->GetNumberOfIds(); jj++) {
				i1 = idlist->GetId(jj);
				l1 = pointLabels->GetValue(i1);
				conlabels[l1] = -1;;
			}

			//copy pt2 into prev point (pt1)
			copy(pt2, pt2+3, pt1);
		}

		//lookup label and convert to an index
		//save the index in the conlabels map
		for(auto it = conlabels.begin(); it != conlabels.end() ; it++) {
			mit = labelLookup.find(it->first);
			if(mit == labelLookup.end()) {
				cerr << "ERROR WIERD STUFF" << endl;
				return -1;
			}
			it->second = mit->second;
		}

		if(conlabels.size() > 0)
			match_label++;

		// skip tracts that aren't long enough
		if(tractStats->GetComponent(cellnum, 0) < minlength)
			continue;

		/*
		 * Update the Values, averaging len, and fa
		 */

		match_all++;

		//for each pair of regions on the line, add to the graphs
		for(auto it1 = conlabels.begin(); it1 != conlabels.end() ; it1++) {
			for(auto it2 = it1; it2 != conlabels.end() ; it2++) {
				i1 = it1->second;
				i2 = it2->second;

				//count graphs
				ographs[0]->set(i1, i2, ographs[0]->get(i1, i2)+1);
				ographs[0]->set(i2, i1, ographs[0]->get(i2, i1)+1);

				for(unsigned int ii = 1 ; ii < ographs.size() ; ii++) {
					tmp = tractStats->GetComponent(cellnum, ii-1);
					if(!isnan(tmp) && !isinf(tmp)) {
						//counts for this metric
						counts[ii-1]->set(i1, i2, counts[ii-1]->get(i1, i2)+1);
						counts[ii-1]->set(i2, i1, counts[ii-1]->get(i2, i1)+1);

						ographs[ii]->set(i1, i2, ographs[ii]->get(i1, i2)+tmp);
						ographs[ii]->set(i2, i1, ographs[ii]->get(i2, i1)+tmp);
					}
				}

			}
		}
    }
	cerr << "Done Traversing." << endl;
	cerr << match_label<< "/" << cellnum+1 << " Labeled" << endl;
	cerr << match_all << "/" << cellnum+1 << " Passed All Requirements" << endl;

	//average of scalars
	for(unsigned int ii = 1 ; ii < ographs.size() ; ii++) {
		its = ographs[ii]->begin();
		while(its != ographs[ii]->end()) {
			int i1, i2;
			its.geti(i1, i2);
			int c = counts[ii-1]->get(i1, i2);
			if(c != 0)
				*its = (*its)/(double)c;
			++its;
		}
	}

	//free the temporary count matrices
	for(unsigned int ii = 0 ; ii < ographs.size()-1 ; ii++)
		delete counts[ii];

	return 0;
}

/**
* @brief Computes graphs of counts, fiber length, variance of fiber length and FA
*
* @param tractData Input tracts to count over
* @param labelTree Labelmap Tree to lookup values from
* @param leeway distance from endpoints to look for a label
* @param radius distance from fiber to look for connected labels
* @param minlength minimum acceptable fiber length
* @param labelLookup Label to index lookup
* @param ographs Output graphs (0 - count, 1 - length, 2 ... - data scalar image)
*
* @return
*/
int createGraphs(vtkSmartPointer<vtkPolyData> tractData,
			vtkSmartPointer<TreeT> labelTree,
			double leeway, double radius, double minlength,
			const map<int, int>& labelLookup, vector<Graph*>& ographs)
{
	vtkSmartPointer<vtkIntArray> pointLabels = vtkIntArray::SafeDownCast(
				vtkPolyData::SafeDownCast(labelTree->GetDataSet())->
				GetPointData()->GetScalars());
	vtkSmartPointer<vtkFloatArray> tractStats = vtkFloatArray::SafeDownCast(
				tractData->GetCellData()->GetScalars());

    vtkIdType npts;
    vtkIdType* ids;
	vtkIdType id;
	int l1, l2; //label 1, label 2
	int i1, i2; //index 1, index 2, corresponding to label 1 and 2
	double pt1[3];
	double pt2[3];
	double dist = 0;
	Graph::iterator itc; //count graph iterator
	Graph::iterator its; //count graph iterator
	int nlabels = labelLookup.size();
	map<int, int>::const_iterator mit;
	double tmp;

	//used to search for the nearest id along the tracts length
	vtkIdType best_id;
	double best_dist;
	double alongfib;

	/* Initialize Graphs */

	//store counts for each metric
	vector<Graph*> counts(ographs.size()-1);
	for(unsigned int ii = 0 ; ii < ographs.size()-1 ; ii++)
		counts[ii] = new Graph(nlabels, false, true);

	//allocate
	for(unsigned int ii = 0 ; ii < ographs.size() ; ii++)
		ographs[ii] = new Graph(nlabels, false, true);

	//labels
	mit = labelLookup.begin();
	for(int ii = 0 ; ii < nlabels; ii++, mit++) {
		for(unsigned int jj = 0; jj < ographs.size() ; jj++)
			ographs[jj]->setLabel(ii, mit->first);
	}

	cerr << "initializating traversal" << endl;
	/*
	 * for each fiber, map the start/end labels
	 */
	int match_label = 0; //note that GetTraversal Location goes by points
	int match_all = 0; //note that GetTraversal Location goes by points
	int cellnum = -1; //note that GetTraversal Location goes by points
    vtkSmartPointer<vtkCellArray> tracts = tractData->GetLines();
    tracts->InitTraversal();
    while(tracts->GetNextCell(npts, ids)) {
		cellnum++;

		/*
		 * Begining of tract
		 */

		//travel down the fiber for /leeway/ distance looking the closest match
		alongfib = 0;
		best_dist = leeway+1;
		best_id = -1;
		tractData->GetPoint(ids[0], pt1);
		for(int ii = 0; alongfib < leeway && ii < npts; ii++) {
			tractData->GetPoint(ids[ii], pt2);
        	id = labelTree->FindClosestPointWithinRadius(radius, pt2, dist);

			//update closest region
			if(id >= 0 && dist < best_dist) {
				best_dist = dist;
				best_id = id;
			}

			//calculate distance along tract
			alongfib += distance(pt1, pt2);

			//copy pt2 into prev point (pt1)
			copy(pt2, pt2+3, pt1);
		}
		if(best_id < 0) //if there is no nearby point
			continue;

		l1 = pointLabels->GetValue(best_id);

		//lookup label and convert to an index
		mit = labelLookup.find(l1);
		if(mit == labelLookup.end()) {
			cerr << "ERROR WIERD STUFF" << endl;
			return -1;
		}
		i1 = mit->second;

		/*
		 * End of tract
		 */

		//travel down the fiber for /leeway/ distance looking the closest match
		alongfib = 0;
		best_dist = leeway+1;
		best_id = -1;
		tractData->GetPoint(ids[npts-1], pt1);
		for(int ii = npts-1; alongfib < leeway && ii >= 0; ii--) {
			tractData->GetPoint(ids[ii], pt2);
        	id = labelTree->FindClosestPointWithinRadius(radius, pt2, dist);

			//update closest region
			if(id >= 0 && dist < best_dist) {
				best_dist = dist;
				best_id = id;
			}

			//calculate distance along tract
			alongfib += distance(pt1, pt2);

			//copy pt2 into prev point (pt1)
			copy(pt2, pt2+3, pt1);
		}

		if(best_id < 0) //if there is no nearby point
			continue;

		l2 = pointLabels->GetValue(best_id);

		//lookup label and convert to an index
		mit = labelLookup.find(l2);
		if(mit == labelLookup.end()) {
			cerr << "ERROR WIERD STUFF" << endl;
			return -1;
		}
		i2 = mit->second;

		match_label++;

		// skip tracts that aren't long enough
		if(tractStats->GetComponent(cellnum, 0) < minlength)
			continue;

		/*
		 * Update the Values, averaging len, and fa
		 */
		match_all++;
		ographs[0]->set(i1, i2, ographs[0]->get(i1, i2)+1);
		ographs[0]->set(i2, i1, ographs[0]->get(i2, i1)+1);

		for(unsigned int ii = 1 ; ii < ographs.size() ; ii++) {
			tmp = tractStats->GetComponent(cellnum, ii-1);
			if(!isnan(tmp) && !isinf(tmp)) {
				counts[ii-1]->set(i1, i2, counts[ii-1]->get(i1, i2)+1);
				counts[ii-1]->set(i2, i1, counts[ii-1]->get(i2, i1)+1);

				ographs[ii]->set(i1, i2, ographs[ii]->get(i1, i2)+tmp);
				ographs[ii]->set(i2, i1, ographs[ii]->get(i2, i1)+tmp);
			}
		}
    }
	cerr << "Done Traversing." << endl;
	cerr << match_label<< "/" << cellnum+1 << " Labeled" << endl;
	cerr << match_all << "/" << cellnum+1 << " Passed All Requirements" << endl;

	//average of scalars
	for(unsigned int ii = 1 ; ii < ographs.size() ; ii++) {
		its = ographs[ii]->begin();
		while(its != ographs[ii]->end()) {
			int i1, i2;
			its.geti(i1, i2);
			int c = counts[ii-1]->get(i1, i2);
			if(c != 0)
				*its = (*its)/(double)c;
			++its;
		}
	}

	//free the temporary count matrices
	for(unsigned int ii = 0 ; ii < ographs.size()-1 ; ii++)
		delete counts[ii];

	return 0;
}

/**
 * @brief Reads a Mesh File and produces a KD-Tree of points, with integer
 * labels at each point
 *
 * @param filename
 *
 * @return
 */
KDTree<3, 1, float, int> readLabelMesh(string filename)
{

};

/**
 * @brief Reads a label file and produces a KD-Tree of points with an integer
 * label at each point
 *
 * @param filename
 *
 * @return
 */
KDTree<3, 1, float, int> readLabelMap(string filename)
{
	auto labelmap = readMRImage(filename);

	KDTree<3, 1, float, int> tree;
	double pt[3];
	vector<float> fpt(3);
	vector<int> label(1);
	for(NDIter<int> it(labelmap); !it.eof(); ++it) {
		if(*it != 0) {
			it.index(3, pt);
			it.indexToPoint(3, pt, pt);
			for(size_t ii=0; ii<3; ii++)
				fpt[ii] = pt[ii];
			label[0] = *it;
			tree.insert(fpt, label);
		}
	}
	tree.build();
	return tree;
}

