/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file utility.cpp
 *
 *****************************************************************************/

#define _LARGEFILE64_SOURCE

#include <cstdlib>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include "utility.h"
#include "macros.h"
#include "basic_functions.h"

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
 * @brief Returns the directory name for the given file. TODO: windows support
 *
 * @param path Input path
 *
 * @return 				Output directory name of input path
 */
std::string dirname(std::string path)
{
	size_t n = path.find_last_of('/');
	while(true) {
		if(n == 0) {
			return path;
		} else if(path[n-1] == '\\') {
			n = path.find_last_of('/', n);
		} else {
			return path.substr(0, n+1);
		}
	}

	return "";
}

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
vector<vector<string>> readStrCSV(string filename, char& delim, char comment)
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

	std::string delims[] = {";", " \t", ","};

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

	if(delims[priority].length() > 0)
		delim = delims[priority][0];

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
 * @brief Takes a 1 or 3 column format of regressor and produces
 * a timeseries sampeled every tr, starting at time t0, and running
 * for ntimes.
 *
 * 3 column format:
 *
 * onset duration value
 *
 * 1 column format (value of 1 is assumed):
 *
 * onsert
 *
 * @param spec
 * @param tr
 * @param ntimes
 * @param t0
 *
 * @return
 */
vector<double> getRegressor(vector<vector<double>>& spec,
		double tr, size_t ntimes, double t0)
{
	const double upsample = 200;
	assert(!spec.empty());
	if(spec.empty())
		return vector<double>();

	vector<double> out(ntimes);
	vector<double> tseries(ntimes*upsample);

	// fill tseries with 0's
	for(int ii=0; ii < tseries.size(); ii++)
		tseries[ii] = 0;

	if(spec[0].size() == 3) {

		// take each range and fill in the value
		for(size_t ii=0; ii<spec.size(); ii++) {
			bool warned = false;
			double tstart=spec[ii][0];
			double tstop=spec[ii][0]+spec[ii][1];
			int64_t itstart  = lround((tstart-t0)*upsample/tr);
			int64_t itstop = lround((tstop-t0)*upsample/tr);
			for(int64_t it=itstart; it<=itstop; it++) {
				if(it < 0 || it >= tseries.size())
					continue;

				if(!warned && tseries[it] != 0) {
					cerr << "Warning, overlapping specification in 3 column "
						"format: " << spec[ii][0] << " " << spec[ii][1] << " "
						<< spec[ii][2] << endl;
					warned = true;
				}
				tseries[it] = spec[ii][2];
			}
		}
	} else if(spec[0].size() == 1) {

		// find the points closes to the spec points and make them ones
		for(size_t ii=0; ii<spec.size(); ii++) {
			double tstart=spec[ii][0];
			int64_t it = lround((tstart-t0)*upsample/tr);
			if(it >= 0 && it < tseries.size()) {
				tseries[it] = 1;
			}
		}
	} else {
		cerr << "Unrecognized format provided to for "
			<< spec[0].size() << " cols." << endl;
		return vector<double>();
	}

	convolve(tseries, cannonHrf, tr/upsample, 32*upsample);

	for(size_t ii=0 ; ii<ntimes; ii++) {
		double time = ii*tr+t0;
		size_t it = round((time-t0)*upsample/tr);
		out[ii] = tseries[it];
	}

	return out;
}

/**
 * @brief Gamma distribution, used by Cannonical HRF from SPM
 *
 * @param x			Position
 * @param k 		Shape parameter
 * @param theta 	Scale parameter
 *
 * @return
 */
double gammaPDF(double x, double k, double theta)
{
	double a = exp(-x/theta);
	double b = pow(x,k-1);
	double c = pow(theta, k);
	double d = tgamma(k);
	return a*b/(c*d);
}

/**
 * @brief Cannonical hemodynamic response funciton from SPM.
 * delay of response (relative to onset) = 6
 * delay of undershoot (relative to onset) = 16
 * dispersion of response = 1
 * dispersion of undershoot = 1
 * ratio of response to undershoot = 6
 * onset (seconds) = 0
 * length of kernel (seconds) = 32
 *
 * @param t 		Time of sampled value
 * @param rdelay 	Delay of onset (default 6)
 * @param udelay	Delay of undershot (relative to onset, default 16)
 * @param rdisp		Dispersion of response (default 1)
 * @param udisp		Dispersion of undershoot (default 1)
 * @param puRatio	Ratio of response to undershoot (default 6)
 * @param onset		Onset time (default 0)
 * @param total		Kernel time (default 32)
 *
 * @return 			Weight at the given time
 */
double cannonHrf(double t, double rdelay, double udelay, double rdisp,
		double udisp, double puRatio, double onset, double total)
{
	if(t < 0 || t > total)
		return 0;

	double u = t-onset;
	double g1 = gammaPDF(u, rdelay/rdisp, 1/rdisp);
	double g2 = gammaPDF(u, udelay/udisp, 1/udisp);
	return g1 - g2/puRatio;
}

/**
 * @brief The cannoical HRF with all default parameters
 *
 * @param t	Time from stimulus
 *
 * @return 	Response level
 */
double cannonHrf(double t)
{
	return cannonHrf(t, 6, 16, 1, 1, 6, 0, 32);
}

/**
 * @brief In place simulation of BOL timeseries using the balloon model.
 * Habituation is accomplished by subtracting exponential moving average of
 * u, thus if u tends to be on a lot, the effective u will be lower, if it
 * turns on for the first time in a long time, the spike will be greater.
 * Learn is the weight of the current point, the previous moving average
 * value is weighted (1-learn)
 *
 * @param len Length of input/output buffer
 * @param iobuff Input/Output Buffer of values (input stimulus, output signal)
 * @param dt Timestep for simulation
 * @param learn The weight of the current point, the previous moving average
 * value is weighted (1-learn), setting learn to 1 removes all habituation
 */
void boldsim(size_t len, double* iobuff, double dt, double learn)
{
	//const double EPSILON_0 = 1.43;
	//const double NU_0 = 40.3;
	//const double TE = .04; //40ms
	//const double k1 = 4.3*NU_0*TE ;
	//const double k2 = EPSILON_0*25*TE;
	//const double k3 = EPSILON_0-1;

	const double TAU_0 = .98; //Hu = .98, Vakorin = 1.18
	const double ALPHA = .33; //Hu = .33,
	const double E_0 = .34; //Hu = .34
	const double V_0 = 0.33; //Hu = .03, set to 0.33 to create roughly 0-1
	const double TAU_S = 1.54; //Hu = 1.54, Vakorin = 2.72
	const double TAU_F = 2.46; //Hu = 2.46, Vakorin = .56
	const double EPSILON = .7; //Hu = .54
	const double k1 = 7*E_0;
	const double k2 = 2;
	const double k3 = 2*E_0-0.2;

	// state variables
	double vt = 1; // blood volume
	double qt = 1; // deogyenated hemogrlobin
	double st = 0; // derivative of flow
	double ft = 1;  // blood flow
	double dvt, dqt, dst, dft; // derivatives of states

	double ut; // input
	double y; // output value
	double ma = 0; // moving average of ut

	for(size_t ii=0; ii<len; ii++) {
		/* Compute Change in State */
		ut = iobuff[ii];
		ma = learn*ut + (1-learn)*ma;

		// habituate - subtract moving average, but ut must be positive,
		if(ut < ma) ut = 0;

		// Normalized Blood Volume
		//V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha))
		dvt = ((ft - pow(vt, 1./ALPHA))/TAU_0);

		// Normalized Deoxyhaemoglobin Content
		//Q_t* = \frac{1}{tau_0} * (\frac{f_t}{E_0} * (1- (1-E_0)^{1/f_t}) -
		//              \frac{q_t}{v_t^{1-1/\alpha}})
		dqt = ((ft/E_0)*(1 - pow(1-E_0, 1./ft)) - qt/pow(vt, 1-1./ALPHA))/TAU_0;

		// Second Derivative of Cerebral Blood Flow
		//S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
		dst = ut*EPSILON - st/TAU_S - (ft-1.)/TAU_F;

		// Normalized Cerebral Blood Flow
		//f_t* = s_t;
		dft = st;

		/* Integrate / Compute New State */
		vt += dvt*dt;
		qt += dqt*dt;
		st += dst*dt;
		ft += dft*dt;

		y = V_0*(k1*(1-qt) + k2*(1-qt/vt) + k3*(1 - vt));
		iobuff[ii] = y;
	}
}

/**
 * @brief Convolves a signal and a function using loops (not fast)
 * No wrapping is done.
 *
 * for ii in 0...N-1
 * for jj in 0...M-1
 * if jj-ii valid:
 *     s[ii] = s[ii]*kern[jj-ii]
 *
 * @param signal Input signal sampled every (tr)
 * @param foo Function to sample backwards
 * @param tr Sampling time of input signal
 * @param length Number of samples to draw from foo
 */
void convolve(std::vector<double>& signal, double(*foo)(double),
		double tr, double length)
{
	std::vector<double> kern(length/tr);
	for(size_t ii=0; ii<kern.size(); ii++)
		kern[ii] = foo(ii*tr);

	std::vector<double> tmp(signal);
	for(int64_t ii=0; ii<signal.size(); ii++) {
		signal[ii] = 0;
		for(int64_t jj=0; jj<kern.size(); jj++) {
			if(jj-ii >= 0)
				signal[ii] += tmp[ii]*kern[jj-ii];
		}
	}
}

MemMap::MemMap(std::string fn, size_t bsize, bool createNew) :
	m_size(0), m_fd(0), m_data(NULL)
{
	if(createNew) {
		openNew(fn, bsize);
	} else {
		openExisting(fn);
	}
};

int64_t MemMap::openNew(string fn, size_t bsize)
{
	close();

	// Create an Emptry File
	m_fd = ::open(fn.c_str(), O_LARGEFILE|O_CREAT|O_TRUNC|O_RDWR|O_SYNC, S_IRWXU);
	if(m_fd == -1) {
		std::cerr<<"Error Opening file"<< endl;
		return -1;
	}

	// Seek to end
	int result;
	result = lseek(m_fd, bsize-1, SEEK_SET);
	if (result == -1) {
		std::cerr << "Error when seeking "<<bsize<<" into file" << endl;
		::close(m_fd);
		m_size = -1;
		return -1;
	}

	// Write a byte at the end
	result = write(m_fd, "", 1);
	if(result < 0) {
		std::cerr << "Error when writing file" << endl;
		::close(m_fd);
		m_size = -1;
		return -1;
	}

	m_data = mmap(NULL, bsize, PROT_READ|PROT_WRITE, MAP_SHARED, m_fd, 0);
	m_size = bsize;
	return m_size;
};

int64_t MemMap::openExisting(string fn, bool quiet)
{
	close();

	// Check that the file size matches expectations
	struct stat st;
	if(stat(fn.c_str(), &st) != 0) {
		if(!quiet)
			cerr<<"Stat error on input file: "<<fn<<endl;
		return -1;
	}

	m_size = st.st_size;;
	m_fd = ::open(fn.c_str(), O_LARGEFILE|O_RDWR);
	if(m_fd < 0) {
		if(!quiet)
			cerr<<"Error opening existing file: "<<fn<<endl;
		return -1;
	}

	m_data = mmap(NULL, m_size, PROT_READ|PROT_WRITE, MAP_SHARED, m_fd, 0);
	return m_size;
};

void MemMap::close()
{
	if(m_size > 0) {
		munmap(m_data, m_size);
		::close(m_fd);
	}
	m_data = NULL;
	m_fd = -1;
	m_size = 0;
};

MemMap::~MemMap()
{
	close();
};

} // NPL

