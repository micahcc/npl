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
 * @file tga_test1.cpp Tests function plotting, 
 *
 *****************************************************************************/

#include <iostream>
#include <cstdlib>
#include <tuple>
#include <fstream>
#include <cassert>
#include <cmath>

#include "basic_plot.h"

namespace npl {

TGAPlot::TGAPlot(size_t xres, size_t yres)
{
	clear();
	res[0] = xres;
	res[1] = yres;
}

void TGAPlot::clear()
{
	res[0] = 1024;
	res[1] = 768;
	xrange[0] = NAN;
	xrange[1] = NAN;
	yrange[0] = NAN;
	yrange[1] = NAN;
	
	colors.clear();
	colors.push_back(StyleT("r"));
	colors.push_back(StyleT("g"));
	colors.push_back(StyleT("b"));
	colors.push_back(StyleT("y"));
	colors.push_back(StyleT("c"));
	colors.push_back(StyleT("p"));

	curr_color = colors.begin();

	funcs.clear();
	arrs.clear();
}

/**
 * @brief Writes the output image to the given file
 *
 * @param fname File name to write to.
 */
void TGAPlot::write(std::string fname)
{
	write(res[0], res[1], fname);
}

/**
 * @brief If the ranged haven't been provided, then autoset them
 */
void TGAPlot::computeRange(size_t xres)
{
	bool pad_x = false;;
	bool pad_y = false;;

	// compute range
	if(std::isnan(xrange[0]) || std::isinf(xrange[0])) {
		// compute minimum
		xrange[0] = INFINITY;
		for(auto& arr : arrs) {
			auto& xarr = std::get<1>(arr);
			for(auto& v: xarr) {
				if(v < xrange[0]) 
					xrange[0] = v;
			}
		}

		pad_x = true;
	}
	if(std::isnan(xrange[1]) || std::isinf(xrange[1])) {
		// compute minimum
		xrange[1] = -INFINITY;
		for(auto& arr : arrs) {
			auto& xarr = std::get<1>(arr);
			for(auto& v: xarr) {
				if(v > xrange[1]) 
					xrange[1] = v;
			}
		}

		double pad = (xrange[1]-xrange[0])*.1;
		if(pad_x) 
			xrange[0] -= pad/2;
		xrange[1] += pad/2;
	}

	if(std::isnan(yrange[0]) || std::isinf(yrange[0])) {
		// compute minimum
		yrange[0] = INFINITY;

		// from arrays
		for(auto& arr : arrs) {
			auto& yarr = std::get<2>(arr);
			for(auto& v: yarr) {
				if(v < yrange[0]) 
					yrange[0] = v;
			}
		}
		
		// from functions, use breaking up x range
		for(auto& functup: funcs) {
			auto& func = std::get<1>(functup);
			double step = (xrange[1]-xrange[0])/xres;
			for(int64_t ii=0; ii<xres; ii++) {
				double x = xrange[0]+ii*step;
				double y = func(x);
				if(y < yrange[0])
					yrange[0] = y;
			}
		}
		pad_y = true;
	}
	
	if(std::isnan(yrange[1]) || std::isinf(yrange[1])) {
		// compute minimum
		yrange[1] = -INFINITY;

		// from arrays
		for(auto& arr : arrs) {
			auto& yarr = std::get<2>(arr);
			for(auto& v: yarr) {
				if(v > yrange[1]) 
					yrange[1] = v;
			}
		}
		
		// from functions, use breaking up x range
		for(auto& functup: funcs) {
			auto& func = std::get<1>(functup);
			double step = (xrange[1]-xrange[0])/xres;
			for(int64_t ii=0; ii<xres; ii++) {
				double x = xrange[0]+ii*step;
				double y = func(x);
				if(y > yrange[1])
					yrange[1] = y;
			}
		}

		double pad = (yrange[1]-yrange[0])*.1;
		if(pad_y) 
			yrange[0] -= pad/2;
		yrange[1] += pad/2;
	}

	if(fabs(yrange[1]-yrange[0]) < 0.00001)
		yrange[1] = yrange[0]+0.00001;
	if(fabs(xrange[1]-xrange[0]) < 0.00001)
		xrange[1] = xrange[0]+0.00001;
}

/**
 * @brief Write the output image with the given (temporary) resolution.
 * Does not affect the internal resolution
 *
 * @param xres X resolution
 * @param yres Y resolution
 * @param fname Filename
 */
void TGAPlot::write(size_t xres, size_t yres, std::string fname)
{
	std::ofstream o(fname.c_str(), std::ios::out | std::ios::binary);

	//Write the header
	o.put(0); //ID
	o.put(0); //Color Map Type
	o.put(10); // run length encoded truecolor
	
	// color map
	o.put(0);
	o.put(0);
	o.put(0);
	o.put(0);
	o.put(0);
	
	//X origin
	o.put(0);
	o.put(0);

	//Y origin
	o.put(0);
	o.put(0);

	//width
	o.put((xres & 0x00FF));
	o.put((xres & 0xFF00) / 256);
	
	//height
	o.put((yres & 0x00FF));
	o.put((yres & 0xFF00) / 256);
	
	//depth
	o.put(32); /* 8 bit bitmap */
	
	//descriptor
	o.put(8); // 8 for RGBA

	// before performing run-length encoding, we need to fill a buffer 
	rgba* buffer = new rgba[xres*yres];
	for(size_t ii=0; ii<xres*yres; ii++) {
		buffer[ii][0] = 255;
		buffer[ii][1] = 255;
		buffer[ii][2] = 255;
		buffer[ii][3] = 255;
	}

	//////////////////////////////////////////////////////////////////////////
	// fill buffer
	//////////////////////////////////////////////////////////////////////////
	computeRange(xres);

	double xstep = (xrange[1]-xrange[0])/xres;
	double ystep = (yrange[1]-yrange[0])/yres;
	// start with buffers, interpolating between points
	for(auto& arr: arrs) {
		auto& sty = std::get<0>(arr);
		auto& xarr = std::get<1>(arr);
		auto& yarr = std::get<2>(arr);
		assert(xarr.size() == yarr.size());

		for(size_t ii=1; ii<xarr.size(); ii++) {
			double xp = (xarr[ii-1]-xrange[0])/xstep;
			double xf = (xarr[ii]-xrange[0])/xstep;
			double dx = xf-xp;
			double yp = (yarr[ii-1]-yrange[0])/ystep;
			double yf = (yarr[ii]-yrange[0])/ystep;
			double dy = yf-yp;
			
			// we want to take steps less than 1 in the fastest moving direction
			if(fabs(dx) > fabs(dy)) {
				dy /= (fabs(dx)+1);
				dx /= (fabs(dx)+1);
			} else {
				dx /= (fabs(dy)+1);
				dy /= (fabs(dy)+1);
			}

			bool done = false;
			while(!done) {
				int64_t xi = std::max<int>(std::min<int>(xres-1, round(xp)), 0);
				int64_t yi = std::max<int>(std::min<int>(yres-1, round(yp)), 0);
				buffer[yi*xres+xi][0] = sty.rgba[0];
				buffer[yi*xres+xi][1] = sty.rgba[1];
				buffer[yi*xres+xi][2] = sty.rgba[2];
				buffer[yi*xres+xi][3] = sty.rgba[3];

				// step
				xp+=dx;
				yp+=dy;
				if(fabs(xp) > fabs(yp)) {
					if(dx >= 0 && xp >= xf)
						done = true;
					else if(dx < 0 && xp <= xf)
						done = true;
				} else {
					if(dy >= 0 && yp >= yf)
						done = true;
					else if(dy < 0 && yp <= yf)
						done = true;
				}
			}
		}
	}
	
	for(auto& func: funcs) {
		auto& sty = std::get<0>(func);
		auto& foo = std::get<1>(func);

		double yip = NAN; // previous y index
		double yi; // y index
		double xx = xrange[0];
		while(xx < xrange[1]) {
			double xbase = xx;
			double dx = xstep;
			double yy;
			do {
				xx = xbase + dx;
				yy = foo(xx);
				yi = (yy-yrange[0])/ystep;
				dx /= 2;
			} while((yip - yi) > 1);
			yip = yi;
			int64_t yind = round(yi);
			int64_t xind = round((xx-xrange[0])/xstep);
			
			buffer[yind*xres + xind][0] = sty.rgba[0];
			buffer[yind*xres + xind][1] = sty.rgba[1];
			buffer[yind*xres + xind][2] = sty.rgba[2];
			buffer[yind*xres + xind][3] = sty.rgba[3];
		}
	}

	for(size_t ii=0; ii<xres*yres; ) {

		/* 
		 * Determine Run Length
		 */
		
		// find longest run from here
		size_t runlen = 1;
		while(ii+runlen<xres*yres && runlen<=128) {
			if(buffer[ii][0] != buffer[ii+runlen][0] ||
						buffer[ii][1] != buffer[ii+runlen][1] || 
						buffer[ii][2] != buffer[ii+runlen][2] || 
						buffer[ii][3] != buffer[ii+runlen][3]) {
				break;
			} else {
				runlen++;
			}
		}
		
		if(runlen > 128)
			runlen = 128;

		if(runlen > 1) {
			// run length encode
			unsigned char packet = 128+(runlen-1);
			o.put(packet);
			o.put(buffer[ii][0]);
			o.put(buffer[ii][1]);
			o.put(buffer[ii][2]);
			o.put(buffer[ii][3]);
		} else {
			// determine how long things are changing for
			runlen = 1;
			while(ii+runlen < xres*yres && runlen<=128) { 
				if(buffer[ii+runlen-1][0] == buffer[ii+runlen][0] &&
						buffer[ii+runlen-1][1] == buffer[ii+runlen][1] &&
						buffer[ii+runlen-1][2] == buffer[ii+runlen][2] &&
						buffer[ii+runlen-1][3] == buffer[ii+runlen][3]) {
					break;
				} else {
					runlen++;
				}
			}
			if(runlen > 128) runlen = 128;
			unsigned char packet = (runlen-1);
			o.put(packet);
			for(size_t jj=ii; jj<ii+runlen; jj++) {
				o.put(buffer[jj][0]);
				o.put(buffer[jj][1]);
				o.put(buffer[jj][2]);
				o.put(buffer[jj][3]);
			}
		}
		ii += runlen;
	}
	
	//close the file
	o.close();
}

/**
 * @brief Sets the x range. To use the extremal values from input arrays
 * just leave these at the default (NAN's)
 *
 * @param low Lower bound
 * @param high Upper bound
 */
void TGAPlot::setXRange(double low, double high)
{
	xrange[0] = low;
	xrange[1] = high;
}

/**
 * @brief Sets the y range. To use the extremal values from input arrays
 * and computed yvalues from functions, just leave these at the default
 * (NAN's)
 *
 * @param low Lower bound
 * @param high Upper bound
 */
void TGAPlot::setYRange(double low, double high)
{
	yrange[0] = low;
	yrange[1] = high;
}

/**
 * @brief Sets the default resolution
 *
 * @param xres Width of output image
 * @param yres Height of output image
 */
void TGAPlot::setRes(size_t xres, size_t yres)
{
	res[0] = xres;
	res[1] = yres;
}

void TGAPlot::addFunc(Function f)
{
	this->addFunc(*curr_color, f);
	curr_color++;
	if(curr_color == colors.end())
		curr_color = colors.begin();
}

void TGAPlot::addFunc(const std::string& style, Function f)
{
	StyleT tmps(style);
	addFunc(tmps, f);
}

void TGAPlot::addFunc(const StyleT& style, Function f)
{
	funcs.push_back(std::make_tuple(style, f));
}

void TGAPlot::addArray(size_t sz, const double* array)
{
	std::vector<double> tmpx(sz);
	std::vector<double> tmpy(sz);
	for(size_t ii=0; ii<sz; ii++) {
		tmpx[ii] = ii;
		tmpy[ii] = array[ii];
	}
	
	arrs.push_back(std::make_tuple(*curr_color, tmpx, tmpy));

	curr_color++;
	if(curr_color == colors.end())
		curr_color = colors.begin();
}

void TGAPlot::addArray(size_t sz, const double* xarr, const double* yarr)
{
	std::vector<double> tmpx(sz);
	std::vector<double> tmpy(sz);
	for(size_t ii=0; ii<sz; ii++) {
		tmpx[ii] = xarr[ii];
		tmpy[ii] = yarr[ii];
	}

	arrs.push_back(std::make_tuple(*curr_color, tmpx, tmpy));
	
	curr_color++;
	if(curr_color == colors.end())
		curr_color = colors.begin();
};

void TGAPlot::addArray(const std::string& style, size_t sz, const double* array)
{
	std::vector<double> tmpx(sz);
	std::vector<double> tmpy(sz);
	for(size_t ii=0; ii<sz; ii++) {
		tmpx[ii] = ii;
		tmpy[ii] = array[ii];
	}
	StyleT tmpstyle(style);
	arrs.push_back(std::make_tuple(tmpstyle, tmpx, tmpy));
}

void TGAPlot::addArray(const StyleT& style, size_t sz, const double* xarr, const double* yarr)
{
	std::vector<double> tmpx(sz);
	std::vector<double> tmpy(sz);
	for(size_t ii=0; ii<sz; ii++) {
		tmpx[ii] = xarr[ii];
		tmpy[ii] = yarr[ii];
	}

	arrs.push_back(std::make_tuple(style, tmpx, tmpy));
}

}
