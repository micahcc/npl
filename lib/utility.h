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
 * @file utility.h File with random utilities, for string processing, for 
 * determining basic filename information (basename/dirname). 
 *
 *****************************************************************************/
#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <string>
#include <cmath>
#include <list>
#include <vector>

#define __FUNCTION_STR__ std::string(__PRETTY_FUNCTION__)

namespace npl {

/**
 * @brief Returns the directory name for the given file
 *
 * @param path Input path
 *
 * @return 				Output directory name of input path
 */
std::string dirname(std::string path);

/**
 * @brief Reads a file and returns true if its entirely made up of printable ascii
 *
 * @param filename name of file to read
 *
 * @return if the file is text
 */
bool isTxt(std::string filename);

/**
 * @brief Returns true if a file exists and is possible to open
 *
 * @param filename to read
 *
 * @return if the file exists
 */
bool fileExists(std::string filename);

/**
 * @brief Removes whitespace at the beginning and end of a string
 *
 * @param str string to chomp
 *
 * @return chomped string
 */
std::string chomp(std::string str);

/**
 * @brief Given a delimiter splits the line based on the delmiter and removes
 * extra white space as necessary
 *
 * Note that repeated white space characters will be removed but other delimiters
 * wont be
 *
 * TODO handle ", " which should technically be ignored and maybe \, but not \\,
 * etc
 *
 * @param line string holding the line
 * @param delim character delimiter[s]
 *
 * @return vector of strings, one string per parsed token
 */
std::vector<std::string> parseLine(std::string line, std::string delim);

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
 * @return out vector of rows (stored in vectors)
 */
std::vector<std::vector<std::string>> readStrCSV(std::string filename,
			char comment = '#');

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
 * @return out vector of rows (stored in vectors)
 */
std::vector<std::vector<double>> readNumericCSV(std::string filename,
			char comment = '#');

/**
 * @brief Gamma distribution, used by Cannonical HRF from SPM
 *
 * @param x			Position
 * @param k 		Shape parameter
 * @param theta 	Scale parameter
 *
 * @return 
 */
double gammaPDF(double x, double k, double theta);


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
		double udisp, double puRatio, double onset, double total);

/**
 * @brief The cannoical HRF with all default parameters
 *
 * @param t	Time from stimulus
 *
 * @return 	Response level 
 */
double cannonHrf(double t);

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
std::vector<double> getRegressor(std::vector<std::vector<double>>& spec, 
		double tr, size_t ntimes, double t0);

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
std::vector<double> getRegressor(std::vector<std::vector<double>>& spec, 
		double tr, size_t ntimes, double t0);

/**
 * @brief Convolves a signal and a function using loops (not fast)
 *
 * @param signal
 * @param foo
 * @param tr
 * @param length
 */
void convolve(std::vector<double>& signal, double(*foo)(double),
		double tr, double length);

/**
 * @brief Computes the standard deviation from FWHM (because I can never
 * remember the ratio)
 *
 * @param fwhm  Full width-half max that we want to convert to equivalent
 * standard deviation
 *
 * @return  Standard deviation for a gaussian with the specified FWHM
 */
inline
double fwhm_to_sd(double fwhm)
{
    return fwhm/(2*sqrt(2*log(2)));
}

/**
 * @brief Computes the FWHM from from standard deviation (because I can never
 * remember the ratio)
 *
 * @param sd Standard deviation of a Gaussian function. 
 *
 * @return Full-width-half-max size of the same gaussian function
 */
inline
double sd_to_fwhm(double sd)
{
    return sd*(2*sqrt(2*log(2)));
}

}
#endif // UTILITY_FUNCTIONS_H


