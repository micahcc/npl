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
 * @file utility.h
 *
 *****************************************************************************/
#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <string>
#include <cmath>
#include <list>
#include <vector>


namespace npl {


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
inline
double sample_var(int count, double sum, double sumsqr);

/******************************************
 * Tools for writing TGA image files/plots
 *****************************************/

/**
 * @brief Writes a TGA image from the data vector
 *
 * @param filename File name to write
 * @param in array of values to write, values should be in row-major order
 * @param height height of output image
 * @param width width of output image
 * @param log whether to log-transform the data
 */
void writeTGA(std::string filename, const std::vector<double>& in, int height,
			int width, bool log = false);


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
			int width, bool log = false);

/**
 * @brief Creates a TGA file with the x and y values plotted from x and y
 *
 * @param filename	output file name *.tga
 * @param x			x values of each point
 * @param y			y values of each point
 */
void writePlot(std::string filename, const std::vector<double>& y, int ysize);

/**
 * @brief Creates a TGA file with the y values plotted from multiple
 * y's
 *
 * @param filename	output file name *.tga
 * @param y			y values of each point
 */
void writePlot(std::string filename, const std::list<std::vector<double>>& y);

/**
 * @brief Creates a TGA file with the y values plotted from multiple
 * y's
 *
 * @param filename	output file name *.tga
 * @param y			y values of each point
 */
void writePlot(std::string filename, const std::vector<double>& y);

/**
 * @brief Creates a TGA file with the x and y values plotted from x and y
 *
 * @param filename	output file name *.tga
 * @param x			x values of each point
 * @param y			y values of each point
 * TODO actually use x values, rather than just plotting with continuous ii
 */
void writePlot(std::string filename, const std::vector<double>& x,
		const std::vector<double>& y, int ysize);

/**
 * @brief Creates a TGA file with the x and y values plotted based
 * on the input function.
 *
 * @param filename	output file name *.tga
 * @param xrange	min and max x values (start and stop points)
 * @param xres		resolution (density) of xpoints. Ouptut size is range/res
 */
void writePlot(std::string filename, double(*f)(double),
		double xrange[2], double xres, int ysize);

#endif // UTILITY_FUNCTIONS_H


}
