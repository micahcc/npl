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
 * @file utility.cpp
 *
 *****************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include "utility.h"

#include <string>
#include <cassert>
#include <list>
#include <vector>

using std::endl;
using std::string;
using std::list;
using std::vector;
using std::cerr;
using std::numeric_limits;

namespace npl {

/**
 * @brief Reads a file and returns true if its entirely made up of printable ascii
 *
 * @param filename name of file to read
 *
 * @return if the file is text
 */
bool isTxt(string filename)
{
	std::ifstream ifs(filename.c_str());
	if(!ifs.is_open()) {
		return false;
	}

	char a = ifs.get();
	while(!ifs.eof()) {
		if(a != '\t' && a != '\n' && a != '\r' && !(a <= 126 && a >= 32)) {
			return false;
		}
		a = ifs.get();
	}

	return true;
}

/**
 * @brief Returns true if a file exists and is possible to open
 *
 * @param filename to read
 *
 * @return if the file exists
 */
bool fileExists(std::string filename)
{
	std::ifstream f(filename.c_str());
	if (f.good()) {
		f.close();
		return true;
	} else {
		f.close();
		return false;
	}
}

/**
 * @brief Removes whitespace at the beginning and end of a string
 *
 * @param str string to chomp
 *
 * @return chomped string
 */
string chomp(string str)
{
	int begin, end;
	for(begin = 0; begin < (int)str.size() && isspace(str[begin]); begin++)
		continue;

	if(begin == (int)str.size())
		return string("");
	
	for(end = str.size()-1; end >= 0 && isspace(str[end]); end--)
		continue;

	if(end < 0) //this can't happen
		end = str.length()-1;
	
	return str.substr(begin, end-begin+1);
}

/**
 * @brief Given a delimiter splits the line based on the delmiter and removes
 * extra white space as necessary
 *
 * Note that repeated white space characters will be removed but other delimiters
 * wont be
 *
 * TODO handle values inside quoates like ", " which should technically be
 * ignored and maybe \, but not \\, etc
 *
 * @param line string holding the line
 * @param delim character delimiter[s]
 *
 * @return vector of strings, one string per parsed token
 */
vector<string> parseLine(std::string line, string delim)
{
	size_t pos = 0;
	size_t prev = 0;
	vector<string> out;
	while(pos < line.length()) {
		prev = pos;
		pos = line.find_first_of(delim, prev);
		if(pos == string::npos) {
			pos = line.length();
			string tmp = chomp(line.substr(prev));
			if(tmp.length() != 0)
				out.push_back(tmp);
		} else {
			string tmp = chomp(line.substr(prev, pos-prev));
			if(tmp.length() != 0)
				out.push_back(tmp);
			
			//move past delimiter
			pos++;

			//skip remaining white space
			while(pos < line.size() && isspace(line[pos]))
				pos++;
		}
	}
	return out;
};

/**
 * @brief This function parses an input and returns a list of rows
 *
 * This function reads a text file of the form:
 * DvDvDvD....Dv[D]
 * by deciding whether white space or commas or semi-colons is
 * the delimiter and then proceeding to read each line. It does
 * this by looking at the first 10 lines and comparing the line
 * based on each possible deliminter.
 * out
 *
 * @param filename file to read
 * @param comment lines with this first non-white space character will be ignored
 *
 * @return Vector of vectors, where outer vectors are rows, inner are columns
 */
vector<vector<string>> readStrCSV(string filename, char comment)
{
    std::ifstream fin(filename.c_str());

	if(!fin.is_open()) {
		cerr << "Couldn't open:  " << filename << endl;
		return vector<vector<string>>();
	}

    std::string line;
	vector<string> tmparr;

	list<vector<string> > outstore;

	int linenum = 0;
	int minwidth = numeric_limits<int>::max();
	int maxwidth = 0;
	int priority = 2;

	string delims[3] = {";", "\t ", ","};

	/* Start trying delimiters. Priority is in reverse order so the last that
	 * grants the same number of outputs on a line and isn't 1 is given the
	 * highest priority
	 */
	
	//grab the first few lines
	list<string> firstlines;
	for(int ii = 0 ; !fin.eof(); ii++ ) {
		getline(fin, line);
		firstlines.push_back(line);
	}

	//test our possible delimiters
	for(int ii = 0 ; ii < 3 ; ii++) {
		list<string>::iterator it = firstlines.begin();
		minwidth = numeric_limits<int>::max();
		maxwidth = 0;
		for(;it != firstlines.end(); it++) {
			line = *it;
			string tmp = chomp(line);
			if(line[0] == comment || tmp[0] == comment || tmp.size() == 0)
				continue;

			// parse the line, and compute width
			tmparr = parseLine(line, delims[ii]);

			if((int)tmparr.size() < minwidth) {
				minwidth = tmparr.size();
			}
			if((int)tmparr.size() > maxwidth) {
				maxwidth = tmparr.size();
			}
		}
		if(maxwidth > 1 && maxwidth == minwidth) {
			priority = ii;
		}
	}
			
	
	//re-process first 10 lines using the proper delimiter
	list<string>::iterator it = firstlines.begin();
	minwidth = numeric_limits<int>::max();
	maxwidth = 0;
	for(;it != firstlines.end(); it++, linenum++) {
		line = *it;
		string tmp = chomp(line);
		if(line[0] == comment || tmp[0] == comment || tmp.size() == 0) {
			continue;
		}

		tmparr = parseLine(line, delims[priority]);
		if((int)tmparr.size() < minwidth)
			minwidth = tmparr.size();
		if((int)tmparr.size() > maxwidth)
			maxwidth = tmparr.size();

		outstore.push_back(tmparr);
	}

	//process the rest of the input (the reason we don't use get is we might want
	//to parse - as stdin
	for(;!fin.eof(); linenum++) {
		getline(fin, line);
		string tmp = chomp(line);
		if(line[0] == comment || tmp[0] == comment || tmp.size() == 0) {
			continue;
		}
		
		tmparr = parseLine(line, delims[priority]);
		if((int)tmparr.size() < minwidth) {
			minwidth = tmparr.size();
		}
		if((int)tmparr.size() > maxwidth) {
			maxwidth = tmparr.size();
		}

		outstore.push_back(tmparr);
	}

	//copy the output from a list to a vector
	vector<vector<string>> out(outstore.size());

	size_t ii=0;
	for(auto it = outstore.begin(); it != outstore.end(); ++ii, ++it) {
		out[ii] = std::move(*it);
	}

	if(minwidth != maxwidth || minwidth == 0) {
		cerr << "Warning you may want to be concerned that there are "
			<< "differences in the number of fields per line" << endl;
	}

    return out;
}

/**
 * @brief This function parses an input and returns a list of rows
 *
 * This function reads a text file of the form:
 * DvDvDvD....Dv[D]
 * by deciding whether white space or commas or semi-colons is
 * the delimiter and then proceeding to read each line. It does
 * this by looking at the first 10 lines and comparing the line
 * based on each possible deliminter.
 * out
 *
 * @param filename file to read
 * @param out vector of rows (stored in vectors)
 * @param comment lines with this first non-white space character will be ignored
 *
 * @return maximum row width
 */
std::vector<std::vector<double>> readNumericCSV(string filename, char comment)
{

	vector<vector<string>> tmp = readStrCSV(filename, comment);
	
	//allocate output data
	std::vector<std::vector<double>> out(tmp.size());
	for(unsigned int rr = 0 ; rr < out.size(); rr++) {
		out[rr].resize(tmp[rr].size(), 0);
		for(int cc = 0 ; cc < tmp[rr].size(); cc++) {
			out[rr][cc] = atof(tmp[rr][cc].c_str());
		}
	}
	
	return out;
}

/**
 * @brief Takes a count, sum and sumsqr and returns the sample variance.
 * This is slightly different than the variance definition and I can
 * never remember the exact formulation.
 *
 * @param count Number of samples
 * @param sum sum of samples
 * @param sumsqr sum of square of the samples
 *
 * @return sample variance
 */
double sample_var(int count, double sum, double sumsqr)
{
	return (sumsqr-sum*sum/count)/(count-1);
}


/**
 * @brief Creates a TGA file with the data from the input vector.
 *
 * Data in the input vector should be row-major (ie rows should be
 * contiguous in memory). Number of rows = height, number of columns = width.
 *
 * @param filename	output file name *.tga
 * @param in		input vector, should be 2 dimensional, row major
 * @param height	number of rows
 * @param width		number of columns
 * @param log		Perform log transform on data
 */
void writeTGA(std::string filename, const std::vector<double>& in, int height,
			int width, bool log)
{
	assert(in.size() == (size_t)width*height);

	std::ofstream o(filename.c_str(), std::ios::out | std::ios::binary);

	//Write the header
	o.put(0); //ID
	o.put(0); //Color Map Type
	o.put(3); //uncompressed grayscale
	
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
	o.put((width & 0x00FF));
	o.put((width & 0xFF00) / 256);
	
	//height
	o.put((height & 0x00FF));
	o.put((height & 0xFF00) / 256);
	
	//depth
	o.put(8); /* 8 bit bitmap */
	
	//descriptor
	o.put(0);

	//get min and max
	double min = INFINITY;
	for(uint32_t ii = 0 ; ii < in.size() ; ii++)
		min = std::min(min, in[ii]);

	double max = -INFINITY;
	if(log) {
		for(uint32_t ii = 0 ; ii < in.size() ; ii++)
			max = std::max(max, std::log(in[ii]-min+1));
		min = 0;
	} else {
		for(uint32_t ii = 0 ; ii < in.size() ; ii++)
			max = std::max(max, in[ii]);
	}

	assert(min != max);
	
	//compress range to 0-255
	double range = max-min;

	if(range != 0) {
		//Write the pixel data
		if(log) {
			for(uint32_t ii=0; ii < in.size(); ii++)
				o.put((unsigned char)(255*std::log(in[ii]-min+1)/range));
		} else {
			for(uint32_t ii=0; ii < in.size(); ii++)
				o.put((unsigned char)(255*(in[ii]-min)/range));
		}
	} else {
		for(uint32_t ii=0; ii < in.size(); ii++)
			o.put(0);
	}

	//close the file
	o.close();
}

/**
 * @brief Writes a TGA image from the data vector
 *
 * @param filename File name to write
 * @param in array of values to write, values should be in row-major order
 * @param height height of output image
 * @param width width of output image
 * @param log whether to log-transform the data
 */
void writeTGA(std::string filename, const std::vector<float>& in, int height,
			int width, bool log)
{
	assert(in.size() == (size_t)width*height);

	std::ofstream o(filename.c_str(), std::ios::out | std::ios::binary);

	//Write the header
	o.put(0); //ID
	o.put(0); //Color Map Type
	o.put(3); //uncompressed grayscale
	
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
	o.put((width & 0x00FF));
	o.put((width & 0xFF00) / 256);
	
	//height
	o.put((height & 0x00FF));
	o.put((height & 0xFF00) / 256);
	
	//depth
	o.put(8); /* 8 bit bitmap */
	
	//descriptor
	o.put(0);

	//get min and max
	float min = INFINITY;
	for(uint32_t ii = 0 ; ii < in.size() ; ii++)
		min = std::min(min, in[ii]);

	float max = -INFINITY;
	if(log) {
		for(uint32_t ii = 0 ; ii < in.size() ; ii++)
			max = std::max(max, std::log(in[ii]-min+1));
		min = 0;
	} else {
		for(uint32_t ii = 0 ; ii < in.size() ; ii++)
			max = std::max(max, in[ii]);
	}

	assert(min != max);
	
	//compress range to 0-255
	double range = max-min;

	if(range != 0) {
		//Write the pixel data
		if(log) {
			for(uint32_t ii=0; ii < in.size(); ii++)
				o.put((unsigned char)(255*std::log(in[ii]-min+1)/range));
		} else {
			for(uint32_t ii=0; ii < in.size(); ii++)
				o.put((unsigned char)(255*(in[ii]-min)/range));
		}
	} else {
		for(uint32_t ii=0; ii < in.size(); ii++)
			o.put(0);
	}

	//close the file
	o.close();
}

/**
 * @brief Creates a TGA file with the x and y values plotted from x and y
 *
 * @param filename	output file name *.tga
 * @param x			x values of each point
 * @param y			y values of each point
 */
void writePlot(std::string filename, const std::vector<double>& y, int ysize)
{
	if(ysize < 0)
		ysize = y.size();
	size_t XSIZE = y.size();

	std::vector<double> image(XSIZE*ysize, 0);

	double yrange[2] = {INFINITY, -INFINITY};

	//find min and max values of x and y
	for(unsigned int ii = 0 ; ii < XSIZE; ii++) {
		yrange[0] = std::min(y[ii], yrange[0]);
		yrange[1] = std::max(y[ii], yrange[1]);
	}

	for(unsigned int ii = 0 ; ii < XSIZE; ii++) {
		int ypos = ysize*(y[ii]-yrange[0])/(yrange[1]-yrange[0]+.000001);
		image[ypos*XSIZE + ii] = 1;
	}

	writeTGA(filename, image, ysize, XSIZE);
}

/**
 * @brief Creates a TGA file with the y values plotted from multiple
 * y's
 *
 * @param filename	output file name *.tga
 * @param y			y values of each point
 */
void writePlot(std::string filename, const std::list<std::vector<double>>& y)
{
	if(y.empty())
		return;

	size_t ysize = 0;
	double yrange[2] = {INFINITY, -INFINITY};
	//find min and max values of x and y
	for(auto it=y.begin(); it!=y.end(); it++) {
		ysize = std::max<size_t>(ysize, it->size());
		for(unsigned int ii = 0 ; ii < it->size(); ii++) {
			yrange[0] = std::min((*it)[ii], yrange[0]);
			yrange[1] = std::max((*it)[ii], yrange[1]);
		}
	}
	int XSIZE = y.size();

	std::vector<double> image(XSIZE*ysize, 0);
	for(auto it=y.begin(); it!=y.end(); it++) {
		for(unsigned int ii = 0 ; ii < it->size(); ii++) {
			int ypos = ysize*((*it)[ii]-yrange[0])/(yrange[1]-yrange[0]+.000001);
			image[ypos*XSIZE + ii] = 1;
		}
	}

	writeTGA(filename, image, ysize, XSIZE);
}

/**
 * @brief Creates a TGA file with the y values plotted from multiple
 * y's
 *
 * @param filename	output file name *.tga
 * @param y			y values of each point
 */
void writePlot(std::string filename, const std::vector<double>& y)
{
	size_t ysize = 0;
	double yrange[2] = {INFINITY, -INFINITY};
	//find min and max values of x and y
	ysize = std::max<size_t>(ysize, y.size());
	for(unsigned int ii = 0 ; ii < y.size(); ii++) {
		yrange[0] = std::min(y[ii], yrange[0]);
		yrange[1] = std::max(y[ii], yrange[1]);
	}
	int XSIZE = y.size();

	std::vector<double> image(XSIZE*ysize, 0);
	for(unsigned int ii = 0 ; ii < y.size(); ii++) {
		int ypos = ysize*(y[ii]-yrange[0])/(yrange[1]-yrange[0]+.000001);
		image[ypos*XSIZE + ii] = 1;
	}

	writeTGA(filename, image, ysize, XSIZE);
}


/**
 * @brief Creates a TGA file with the x and y values plotted from x and y
 *
 * @param filename	output file name *.tga
 * @param x			x values of each point
 * @param y			y values of each point
 * TODO actually use x values, rather than just plotting with continuous ii
 */
void writePlot(std::string filename, const std::vector<double>& x,
		const std::vector<double>& y, int ysize)
{
	if(ysize < 0)
		ysize = y.size();
	int XSIZE = x.size();

	std::vector<double> image(x.size()*ysize, 0);

	double xrange[2] = {INFINITY, -INFINITY};
	double yrange[2] = {INFINITY, -INFINITY};

	//find min and max values of x and y
	for(unsigned int ii = 0 ; ii < x.size(); ii++) {
		xrange[0] = std::min(x[ii], xrange[0]);
		xrange[1] = std::max(x[ii], xrange[1]);
		yrange[0] = std::min(y[ii], yrange[0]);
		yrange[1] = std::max(y[ii], yrange[1]);
	}

	for(unsigned int ii = 0 ; ii < x.size(); ii++) {
		int ypos = ysize*(y[ii]-yrange[0])/(yrange[1]-yrange[0]+.000001);
		image[ypos*x.size() + ii] = 1;
	}

	writeTGA(filename, image, ysize, XSIZE);
}

/**
 * @brief Creates a TGA file with the x and y values plotted based
 * on the input function.
 *
 * @param filename	output file name *.tga
 * @param xrange	min and max x values (start and stop points)
 * @param xres		resolution (density) of xpoints. Ouptut size is range/res
 */
void writePlot(std::string filename, double(*f)(double),
		double xrange[2], double xres, int ysize)
{
	
	std::vector<double> x((xrange[1]-xrange[0])/xres, 0);
	std::vector<double> y(x.size(), 0);

	double xv = xrange[0];
	for(unsigned int ii = 0 ; ii < x.size(); ii++) {
		y[ii] = f(xv);
		x[ii] = xv;

		xv += xres;
	}

	writePlot(filename, x, y, ysize);
}

/**
 * @brief Rectangle function centered at 0, with radius a, range should be = 2a
 *
 * @param x distance from center
 * @param a radius
 *
 * @return weight
 */
double rectWindow(double x, double a)
{
	if(fabs(x) < a)
		return 1;
	else
		return 0;
}

/**
 * @brief Sinc function centered at 0, with radius a, range should be = 2a
 *
 * @param x distance from center
 * @param a radius
 *
 * @return weight
 */
double sincWindow(double x, double a)
{
	const double PI = acos(-1);

	if(x == 0)
		return 1;
	else if(fabs(x) < a)
		return sin(PI*x/a)/(PI*x/a);
	else
		return 0;
}

/**
 * @brief Lanczos kernel function
 *
 * @param x distance from center
 * @param a radius of kernel
 *
 * @return weight
 */
double lanczosKernel(double x, double a)
{
	const double PI = acos(-1);

	if(x == 0)
		return 1;
	else if(fabs(x) < a)
		return a*sin(PI*x)*sin(PI*x/a)/(PI*PI*x*x);
	else
		return 0;
}


}


