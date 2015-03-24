/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file student_test.cpp Tests students t-score versus p value for several
 * parameters, and compares the results to values from tables
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "version.h"
#include "statistics.h"
#include "utility.h"
#include "basic_plot.h"
#include "macros.h"

using std::string;
using Eigen::MatrixXd;

using namespace std;
using namespace npl;

// From R:
// P:  0.8      DOF:  1    t:  1.376382
// P:  0.8      DOF:  10   t:  0.8790578
// P:  0.8      DOF:  20   t:  0.8599644
// P:  0.8      DOF:  30   t:  0.8537673
// P:  0.9      DOF:  1    t:  3.077684
// P:  0.9      DOF:  10   t:  1.372184
// P:  0.9      DOF:  20   t:  1.325341
// P:  0.9      DOF:  30   t:  1.310415
// P:  0.95     DOF:  1    t:  6.313752
// P:  0.95     DOF:  10   t:  1.812461
// P:  0.95     DOF:  20   t:  1.724718
// P:  0.95     DOF:  30   t:  1.697261
// P:  0.99     DOF:  1    t:  31.82052
// P:  0.99     DOF:  10   t:  2.763769
// P:  0.99     DOF:  20   t:  2.527977
// P:  0.99     DOF:  30   t:  2.457262
// P:  0.999    DOF:  1    t:  318.3088
// P:  0.999    DOF:  10   t:  4.1437
// P:  0.999    DOF:  20   t:  3.551808
// P:  0.999    DOF:  30   t:  3.385185
// P:  0.9999   DOF:  1    t:  3183.099
// P:  0.9999   DOF:  10   t:  5.69382
// P:  0.9999   DOF:  20   t:  4.538521
// P:  0.9999   DOF:  30   t:  4.233986
// P:  0.99999  DOF:  1    t:  31830.99
// P:  0.99999  DOF:  10   t:  7.526954
// P:  0.99999  DOF:  20   t:  5.542839
// P:  0.99999  DOF:  30   t:  5.054032

double prows[7] = {0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999};
double dcols[4] = {1, 10, 20, 30};
double ttable[7][4] = {
{1.376382, 0.8790578, 0.8599644, 0.8537673},
{3.077684, 1.372184, 1.325341, 1.310415},
{6.313752, 1.812461, 1.724718, 1.697261},
{31.82052, 2.763769, 2.527977, 2.457262},
{318.3088, 4.1437, 3.551808, 3.385185},
{3183.099, 5.69382, 4.538521, 4.233986},
{31830.99, 7.526954, 5.542839, 5.054032}};

int main()
{
	// Currently we don't do the last (0.99999) one
	const double DT = 0.05;
	const double TMAX = 32000;
	for(size_t dii=0; dii<4; dii++) {
		StudentsT tdist(dcols[dii], DT, TMAX);
		for(size_t pii=0; pii<7; pii++) {
			double t = tdist.icdf(prows[pii]);
			if(fabs(ttable[pii][dii] - t) > 0.01) {
				cerr << "Mismatch for p: "<<prows[pii]<<"/dof:"<<dcols[dii]
					<<endl;
				cerr<<ttable[pii][dii]<<" vs "<<t<<endl;
				return -1;
			}
		}
	}
	return 0;
}

