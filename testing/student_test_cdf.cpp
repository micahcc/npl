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
 * @file student_test_cdf.cpp Tests students t-score versus p value for several
 * parameters, and compares the results to values from R
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
// DOF:  1   t:  0.5  logp:  -0.4345074
// DOF:  10  t:  0.5  logp:  -0.3768001
// DOF:  20  t:  0.5  logp:  -0.3729
// DOF:  30  t:  0.5  logp:  -0.3715877
// DOF:  1   t:  1    logp:  -0.2876821
// DOF:  10  t:  1    logp:  -0.1868678
// DOF:  20  t:  1    logp:  -0.1798785
// DOF:  30  t:  1    logp:  -0.1775183
// DOF:  1   t:  5    logp:  -0.06489374
// DOF:  10  t:  5    logp:  -0.0002687029
// DOF:  20  t:  5    logp:  -3.436573e-05
// DOF:  30  t:  5    logp:  -1.164841e-05
// DOF:  1   t:  10   logp:  -0.03223968
// DOF:  10  t:  10   logp:  -7.947769e-07
// DOF:  20  t:  10   logp:  -1.581891e-09
// DOF:  30  t:  10   logp:  -2.287626e-11
// DOF:  1   t:  100  logp:  -0.003188069
// DOF:  10  t:  100  logp:  -1.224845e-16
// DOF:  20  t:  100  logp:  -8.850866e-29
// DOF:  30  t:  100  logp:  -9.923059e-40
// DOF:  1   t:  1000 logp:  -0.0003183605
// DOF:  10  t:  1000 logp:  -1.230412e-26
// DOF:  20  t:  1000 logp:  -9.019567e-49
// DOF:  30  t:  1000 logp:  -1.036002e-69

double dcols[4] = {1, 10, 20, 30};
double trows[6] = {0.5, 1, 5, 10, 100, 1000};
double ptable[6][4] = {
{-0.4345074,-0.3768001,-0.3729,-0.3715877},
{-0.2876821,-0.1868678,-0.1798785,-0.1775183},
{-0.06489374,-0.0002687029,-3.436573e-05,-1.164841e-05},
{-0.03223968,-7.947769e-07,-1.581891e-09,-2.287626e-11},
{-0.003188069,-1.224845e-16,-8.850866e-29,-9.923059e-40},
{-0.0003183605,-1.230412e-26,-9.019567e-49,-1.036002e-69}};

int main()
{
	// Currently we don't do the last (0.99999) one
	const double DT = 0.05;
	const double TMAX = 1200;
	for(size_t dii=0; dii<4; dii++) {
		StudentsT tdist(dcols[dii], DT, TMAX);
		for(size_t tii=0; tii<6; tii++) {
			double p = tdist.cdf(trows[tii]);
			if(fabs(exp(ptable[tii][dii]) - p) > 0.0000001) {
				cerr << "Mismatch for t: "<<trows[tii]<<"/dof:"<<dcols[dii]
					<<endl;
				cerr<<ptable[tii][dii]<<" vs "<<log(p)<<endl;
				cerr<<exp(ptable[tii][dii])<<" vs "<<p<<endl;
				return -1;
			}
		}
	}
	return 0;
}


