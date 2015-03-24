/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file tga_test1.cpp Tests function plotting,
 *
 *****************************************************************************/
#ifndef BASIC_PLOT_H
#define BASIC_PLOT_H

#include <iostream>
#include <cstdlib>
#include <list>
#include <vector>
#include <tuple>

namespace npl {

template <typename T>
void writePlot(std::string filename, const std::vector<T>& data);

template <typename T>
void writePlot(std::string filename, const std::vector<T>& data,
		size_t xsize, size_t ysize);

typedef char rgba[4];

/**
 * @brief Class for creating basic plots of arrays or functions. An example
 * might be:
 *
 * Plotter plot;
 *
 * vector<double> data;
 *
 * // (fill data)
 *
 * plot.addArray(data.size(), data.data());
 * plot.write("data.svg");
 *
 */
class Plotter
{
public:
	struct StyleT;

	/**
	 * @brief Constructor
	 */
	Plotter(size_t xres = 1024, size_t yres = 768);

	/**
	 * @brief Removes all state variables, including plotted points, and
	 * functions
	 */
	void clear();

	/**
	 * @brief Writes the output image to the given file
	 *
	 * @param fname File name to write to.
	 */
	int write(std::string fname);

	/**
	 * @brief Write the output image with the given (temporary) resolution.
	 * Does not affect the internal resolution
	 *
	 * @param xres X resolution
	 * @param yres Y resolution
	 * @param fname Filename
	 */
	int write(size_t xres, size_t yres, std::string fname);

	/**
	 * @brief Sets the x range. To use the extremal values from input arrays
	 * just leave these at the default (NAN's)
	 *
	 * @param low Lower bound
	 * @param high Upper bound
	 */
	void setXRange(double low, double high);

	/**
	 * @brief Sets the y range. To use the extremal values from input arrays
	 * and computed yvalues from functions, just leave these at the default
	 * (NAN's)
	 *
	 * @param low Lower bound
	 * @param high Upper bound
	 */
	void setYRange(double low, double high);

	/**
	 * @brief Sets the default resolution
	 *
	 * @param xres Width of output image
	 * @param yres Height of output image
	 */
	void setRes(size_t xres, size_t yres);

	// stores style of functions, and functions themselves
	typedef double (*Function)(double x);

	void addFunc(Function f);
	void addFunc(const std::string& style, Function f);
	void addFunc(const StyleT& style, Function f);

	void addArray(size_t sz, const double* array);
	void addArray(size_t sz, const double* xarr, const double* yarr);
	void addArray(const std::string& style, size_t sz, const double* array);
	void addArray(const StyleT& style, size_t sz, const double* xarr, const double* yarr);

	size_t res[2]; //image size
	double xrange[2];
	double yrange[2];
	bool axes;

	std::list<std::tuple<StyleT, Function>> funcs;

	// stores style, x values and y values
	std::list<std::tuple<StyleT, std::vector<double>, std::vector<double>>> arrs;

	// stores potential colors
	std::list<StyleT> colors;
	std::list<StyleT>::iterator curr_color;

	struct StyleT
	{
		StyleT(std::string a) : StyleT("", a) {} ;

		StyleT(std::string name, std::string a)
		{
			this->name = name;
			full = false;
			bold = false;
			dot = false;
			dash = false;
			star = false;
			for(size_t ii=0; ii<a.size(); ii++) {
				if(a[ii] == '.') {
					dot = true;
				} else if(a[ii] == '=') {
					bold = true;
					full = true;
				} else if(a[ii] == '-') {
					full = true;
					dash = true;
				} else if(a[ii] == '*') {
					star = true;
				} else if(a[ii] == '!') {
					full = true;
				} else if(a[ii] == '#') {
					// read color into RGB
					rgba[0]=255;
					rgba[1]=255;
					rgba[2]=255;
					rgba[3]=255;
					for(int jj=0; ii<a.size() && jj<8; jj++,ii++) {

						if(a[ii] >= 'A' && a[ii] <= 'F') {
							if(jj%2 == 0)
								rgba[jj/2] = a[ii]-'A'+10;
							else
								rgba[jj/2] += (a[ii]-'A'+10)*16;
						} else if(a[ii] >= 'a' && a[ii] <= 'f') {
							if(jj%2 == 0)
								rgba[jj/2] = a[ii]-'a'+10;
							else
								rgba[jj/2] += (a[ii]-'a'+10)*16;
						} else if(a[ii] >= '0' && a[ii] <= '9') {
							if(jj%2 == 0)
								rgba[jj/2] = a[ii]-'0';
							else
								rgba[jj/2] += (a[ii]-'0')*16;
						} else
							break;
					}
				} else if(a[ii] == 'r') { // red
					rgba[0] = 255; rgba[1] = 0; rgba[2] = 0; rgba[3] = 255;
				} else if(a[ii] == 'g') { // green
					rgba[0] = 0; rgba[1] = 255; rgba[2] = 0; rgba[3] = 255;
				} else if(a[ii] == 'b') { // blue
					rgba[0] = 0; rgba[1] = 0; rgba[2] = 255; rgba[3] = 255;
				} else if(a[ii] == 'k') { // black
					rgba[0] = 0; rgba[1] = 0; rgba[2] = 0; rgba[3] = 255;
				} else if(a[ii] == 'w') { // white
					rgba[0] = 255; rgba[1] = 255; rgba[2] = 255; rgba[3] = 255;
				} else if(a[ii] == 'y') { // yellow
					rgba[0] = 255; rgba[1] = 255; rgba[2] = 0; rgba[3] = 255;
				} else if(a[ii] == 'p') { // pink
					rgba[0] = 255; rgba[1] = 0; rgba[2] = 255; rgba[3] = 255;
				} else if(a[ii] == 'G') { // grey
					rgba[0] = 128; rgba[1] = 128; rgba[2] = 128; rgba[3] = 255;
				} else if(a[ii] == 'c') { // cyan
					rgba[0] = 0; rgba[1] = 255; rgba[2] = 255; rgba[3] = 255;
				}

			}
		};

		std::string name;
		unsigned char rgba[4];
		bool dash;
		bool dot;
		bool star;
		bool full;
		bool bold;
	};

private:
	void computeRange(size_t xres);
	int writeTGA(size_t xres, size_t yres, std::string fname);
	int writeSVG(size_t xres, size_t yres, std::string fname);
};

/**
 * @brief Writes a plot to the given filename. This is a convience wrapper
 * around Plotter, which for quick-and-dirty use might be too much setup.
 *
 * @tparam T Type of data to plot (will be cast to double)
 * @param filename
 * @param data Vector of data to plot
 */
template <typename T>
void writePlot(std::string filename, const std::vector<T>& data)
{
	Plotter plt;
    std::vector<double> tmp(data.size());
    for(size_t ii=0; ii<data.size(); ii++)
        tmp[ii] = (double)data[ii];

	plt.addArray(tmp.size(), tmp.data());
	plt.write(filename);
}


/**
 * @brief Writes a plot to the given filename. This is a convience wrapper
 * around Plotter, which for quick-and-dirty use might be too much setup.
 *
 * @tparam T Type of data to plot (will be cast to double)
 * @param filename
 * @param data Vector of 1D data to plot
 * @param xsize size output image (only matters for raster images *.tga)
 * @param ysize size output image (only matters for raster images *.tga)
 */
template <typename T>
void writePlot(std::string filename, const std::vector<T>& data, size_t xsize,
		size_t ysize)
{
	Plotter plt(xsize, ysize);

    std::vector<double> tmp(data.size());
    for(size_t ii=0; ii<data.size(); ii++)
        tmp[ii] = (double)data[ii];
	plt.addArray(tmp.size(), tmp.data());
	plt.write(filename);
}

}; //npl

#endif // BASIC_PLOT

