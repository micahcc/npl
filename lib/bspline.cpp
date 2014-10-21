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
 * @file bspline.cpp Implmementation of B-Spline parameterization of an image
 *
 *****************************************************************************/

struct CubicBSpline
{
	CubicBSpline();

	CubicBSpline(ptr<const MRImage> overlay, double spacing)
	{
		createOverlay(overlay, spacing);
	};

	createOverlay(ptr<const MRImage> overlay, double bspace)
	{
		size_t ndim = overlay->ndim();
		VectorXd spacing(overlay->ndim());
		VectorXd origin(overlay->ndim());
		vector<size_t> osize(ndim, 0);

		// get spacing and size
		for(size_t dd=0; dd<osize.size(); ++dd) {
			osize[dd] = 4+ceil(overlay->dim(dd)*overlay->spacing(dd)/bspace);
			spacing[dd] = bspace;
		}

		params = dPtrCast<MRImage>(overlay->createAnother(
					osize.size(), osize.data(), FLOAT64));
		params->setDirection(in->getDirection(), false);
		params->setSpacing(spacing, false);

		// compute center of input
		VectorXd indc(ndim); // center index
		for(size_t dd=0; dd<ndim; dd++) 
			indc[dd] = (in->dim(dd)-1.)/2.;
		VectorXd ptc(ndim); // point center
		in->indexToPoint(ndim, indc.array().data(), ptc.array().data());

		// compute origin from center index (x_c) and center of input (c): 
		// o = c-R(sx_c)
		for(size_t dd=0; dd<ndim; dd++) 
			indc[dd] = (osize[dd]-1.)/2.;
		origin = ptc - in->getDirection()*(spacing.asDiagonal()*indc);
		params->setOrigin(origin, false);

	};

	bool sample(size_t len, double* incindex, double* v, double* dv,
			bool ras = false)
	{
		NDView<double> pvw(params);

		// initialize variables
		int ndim = params->ndim();
        const size_t* dim = params->dim();

        // convert RAS to index
        vector<double> cindex(ndim, 0);
		for(size_t dd=0; dd<len; dd++)
			cindex[dd] = incindex[dd];

        if(ras) 
            params->pointToIndex(len, cindex.data(), cindex.data());

		vector<int64_t> center(ndim, 0);
		for(size_t dd=0; dd<ndim; dd++)
			center[dd] = round(cindex[dd]);
		vector<int64_t> index(ndim, 0);
		const int KPOINTS = pow(5, ndim);

		bool bounded = false;
		bool iioutside = false;

		// compute weighted pixval by iterating over neighbors, which are
		// combinations of KPOINTS
		double pixval = 0;
		double weight = 0;
		div_t result;
		for(int ii = 0 ; ii < KPOINTS; ii++) {
			weight = 1;

			//set index
			result.quot = ii;
			iioutside = false;
			for(int dd = 0; dd < ndim; dd++) {
				result = std::div(result.quot, 5);
				int offset = ((int64_t)results.rem) - 2; //[-2, 2]
				index[dd] = center + offset;
				weight *= B3kern(index[dd] - cindex[dd]);
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			pixval += weight*pvw[index];
		}
		return pixval;
	};

	double sample(size_t len, double* incindex, bool ras = false)
	{
		NDView<double> pvw(params);

		// initialize variables
		int ndim = params->ndim();
        const size_t* dim = params->dim();

        // convert RAS to index
        vector<double> cindex(ndim, 0);
		for(size_t dd=0; dd<len; dd++)
			cindex[dd] = incindex[dd];

        if(ras) 
            params->pointToIndex(len, cindex.data(), cindex.data());

		vector<int64_t> center(ndim, 0);
		for(size_t dd=0; dd<ndim; dd++)
			center[dd] = round(cindex[dd]);
		vector<int64_t> index(ndim, 0);
		const int KPOINTS = pow(5, ndim);

		bool iioutside = false;

		// compute weighted pixval by iterating over neighbors, which are
		// combinations of KPOINTS
		double pixval = 0;
		double weight = 0;
		div_t result;
		for(int ii = 0 ; ii < KPOINTS; ii++) {
			weight = 1;

			//set index
			result.quot = ii;
			iioutside = false;
			for(int dd = 0; dd < ndim; dd++) {
				result = std::div(result.quot, 5);
				int offset = ((int64_t)results.rem) - 2; //[-2, 2]
				index[dd] = center + offset;
				weight *= -dB3kern(index[dd] - cindex[dd]);
				iioutside = iioutside || index[dd] < 0 || index[dd] >= dim[dd];
			}

			// if the current point maps outside, then we need to deal with it
			if(iioutside) {
				if(m_boundmethod == ZEROFLUX) {
					// clamp
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				} else if(m_boundmethod == WRAP) {
					// wrap
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = wrap<int64_t>(0, dim[dd]-1, index[dd]);
				} else {
					// set wieght to zero, then just clamp
					weight = 0;
					for(size_t dd=0; dd<ndim; dd++)
						index[dd] = clamp<int64_t>(0, dim[dd]-1, index[dd]);
				}
			}

			pixval += weight*pvw[index];
		}
		return pixval;
	};

	ptr<MRImage> reconstruct(ptr<const MRImage> input)
	{
		if(params->getDirection() != input->getDirection()) {
			throw INVALID_ARGUMENT("Input parameters and sample image do "
					"not have identical direction matrices!");
		}

		auto out = dPtrCast<MRImage>(input->createAnother());

		// for each kernel, iterate over the points in the neighborhood
		size_t ndim = input->ndim();
		vector<pair<int64_t,int64_t>> roi(ndim);
		NDIter<double> pit(out); // iterator of pixels
		vector<int64_t> pind(ndim); // index of pixel
		vector<int64_t> ind(ndim); // index
		vector<double> pt(ndim);   // point
		vector<double> cind(ndim); // continuous index

		vector<int> winsize(ndim);
		vector<vector<double>> karray(ndim);
		vector<vector<int>> iarray(ndim);
		for(size_t dd=0; dd<ndim; dd++) {
			winsize[dd] = 1+4*ceil(biasparams->spacing(dd)/out->spacing(dd));
			karray[dd].resize(winsize[dd]);
		}

		// We go through each parameter, and compute the weight of the B-spline
		// parameter at each pixel within the range (2 indexes in parameter
		// space, 2*S_B/S_I indexs in pixel space)
		for(NDConstIter<double> bit(biasparams); !bit.eof(); ++bit) {

			// get continuous index of pixel
			bit.index(ind.size(), ind.data());
			biasparams->indexToPoint(ind.size(), ind.data(), pt.data());
			out->pointToIndex(pt.size(), pt.data(), cind.data());

			// construct weights / construct ROI
			double dist = 0;
			for(size_t dd=0; dd<ndim; dd++) {
				pind[dd] = round(cind[dd]); //pind is the center
				for(int ww=-winsize[dd]/2; ww<=winsize[dd]/2; ww++) {
					dist = (pind[dd]+ww-cind[dd])*out->spacing(dd)/biasparams->spacing(dd);
					karray[dd][ww+winsize[dd]/2] = B3kern(dist);
				}
				roi[dd].first = pind[dd]-winsize[dd]/2;
				roi[dd].second = pind[dd]+winsize[dd]/2;
			}

			pit.setROI(roi);
			for(pit.goBegin(); !pit.eof(); ++pit) {
				pit.index(ind);
				double w = 1;
				for(size_t dd=0; dd<ndim; dd++)
					w *= karray[dd][ind[dd]-pind[dd]+winsize[dd]/2];
				pit.set(*pit + w*(*bit));
			}
		}

		return out;
	}

	ptr<MRImage> params;

};
