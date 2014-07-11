#ifndef NIFTI_H
#define NIFTI_H

typedef struct __attribute__((packed))
{
	int32_t   sizeof_hdr; // = 348

	// unused
	char  data_type[10];
	char  db_name[18];
	int32_t   extents;
	int16_t session_error;
	char  regular;

	union {
		char byte;
		struct __attribute__((packed)) {
			unsigned int freqdim : 2;
			unsigned int phasedim : 2;
			unsigned int slicedim : 2;
			unsigned int unused : 2;
		} bits;
	} dim_info;

	int16_t ndim;	// number of dimensions
	int16_t dim[7];	// size in each of ndim dimensions
	
	// currently ignored
	float intent_p1;	
	float intent_p2;
	float intent_p3;
	int16_t intent_code;

	int16_t datatype; // pixel type
	int16_t bitpix;
	int16_t slice_start;
	float qfac;			// used for quaternions
	float pixdim[7];	// pixel spacing
	float vox_offset;	// where to start reading pixel data

	float scl_slope;	// y = scl_slope*x + scl_inter
	float scl_inter;
	
	// currently ignored
	int16_t slice_end;
	char  slice_code;
	char  xyzt_units;
	float cal_max;
	float cal_min;
	float slice_duration;
	float toffset;
	int32_t   glmax;
	int32_t   glmin;
	char  descrip[80];
	char  aux_file[24];

	int16_t qform_code ;
	int16_t sform_code ;
	float quatern[3]; //b,c,d
	float qoffset[3]; //x,y,z
	float saffine[12]; //row0 = x, row1 = y, row2 = z
	char intent_name[16];
	char magic[4];
} nifti1_header;

typedef struct __attribute__((packed))
{
   int32_t   sizeof_hdr;
   char  magic[8] ;
   int16_t datatype;
   int16_t bitpix;
   int64_t ndim;
   int64_t dim[7];
   double intent_p1 ;
   double intent_p2 ;
   double intent_p3 ;
   double qfac;
   double pixdim[7];
   int64_t vox_offset;
   double scl_slope ;
   double scl_inter ;
   double cal_max;
   double cal_min;
   double slice_duration;
   double toffset;
   int64_t slice_start;
   int64_t slice_end;
   char  descrip[80];
   char  aux_file[24];
   int32_t qform_code ;
   int32_t sform_code ;
   double quatern[3];
   double qoffset[3];
   double saffine[12] ; // row-major order
   int32_t slice_code ;
   int32_t xyzt_units ;
   int32_t intent_code ;
   char intent_name[16];
   union {
	   char byte;
	   struct __attribute__((packed)) {
		   unsigned int freqdim : 2;
		   unsigned int phasedim : 2;
		   unsigned int slicedim : 2;
		   unsigned int unused : 2;
	   } bits;
   } dim_info;
   char unused_str[15];
} nifti2_header;

enum NIFTI_TYPE {
	// covered by get/set int64
	NIFTI_TYPE_UINT8=2,
	NIFTI_TYPE_INT8=256,
	NIFTI_TYPE_INT16=4,
	NIFTI_TYPE_INT32=8,
	NIFTI_TYPE_UINT16=512,
	NIFTI_TYPE_UINT32=768,
	NIFTI_TYPE_INT64=1024,
	NIFTI_TYPE_UINT64=1280, // overflowing an int64 is hard

	// covered by get/set dbl
	NIFTI_TYPE_FLOAT32=16,
	NIFTI_TYPE_FLOAT64=64,
	
	// covered by get/set cdbl
	NIFTI_TYPE_COMPLEX64=32,
	NIFTI_TYPE_COMPLEX128=1792,
	
	// covered by get/set RGBA
	NIFTI_TYPE_RGB24=128,
	NIFTI_TYPE_RGBA32=2304,

	// NOT SUPPORTED
	// covered by get/set cquad
	NIFTI_TYPE_COMPLEX256=2048,
	
	// covered by get/set quad
	NIFTI_TYPE_FLOAT128=1536
	
};

/*---------------------------------------------------------------------------*/
/* INTERPRETATION OF VOXEL DATA:
   ----------------------------
   The intent_code field can be used to indicate that the voxel data has
   some particular meaning.  In particular, a large number of codes is
   given to indicate that the the voxel data should be interpreted as
   being drawn from a given probability distribution.

   VECTOR-VALUED DATASETS:
   ----------------------
   The 5th dimension of the dataset, if present (i.e., dim[0]=5 and
   dim[5] > 1), contains multiple values (e.g., a vector) to be stored
   at each spatiotemporal location.  For example, the header values
    - dim[0] = 5
    - dim[1] = 64
    - dim[2] = 64
    - dim[3] = 20
    - dim[4] = 1     (indicates no time axis)
    - dim[5] = 3
    - datatype = DT_FLOAT
    - intent_code = NIFTI_INTENT_VECTOR
   mean that this dataset should be interpreted as a 3D volume (64x64x20),
   with a 3-vector of floats defined at each point in the 3D grid.

   A program reading a dataset with a 5th dimension may want to reformat
   the image data to store each voxels' set of values together in a struct
   or array.  This programming detail, however, is beyond the scope of the
   NIFTI-1 file specification!  Uses of dimensions 6 and 7 are also not
   specified here.

   STATISTICAL PARAMETRIC DATASETS (i.e., SPMs):
   --------------------------------------------
   Values of intent_code from NIFTI_FIRST_STATCODE to NIFTI_LAST_STATCODE
   (inclusive) indicate that the numbers in the dataset should be interpreted
   as being drawn from a given distribution.  Most such distributions have
   auxiliary parameters (e.g., NIFTI_INTENT_TTEST has 1 DOF parameter).

   If the dataset DOES NOT have a 5th dimension, then the auxiliary parameters
   are the same for each voxel, and are given in header fields intent_p1,
   intent_p2, and intent_p3.

   If the dataset DOES have a 5th dimension, then the auxiliary parameters
   are different for each voxel.  For example, the header values
    - dim[0] = 5
    - dim[1] = 128
    - dim[2] = 128
    - dim[3] = 1      (indicates a single slice)
    - dim[4] = 1      (indicates no time axis)
    - dim[5] = 2
    - datatype = DT_FLOAT
    - intent_code = NIFTI_INTENT_TTEST
   mean that this is a 2D dataset (128x128) of t-statistics, with the
   t-statistic being in the first "plane" of data and the degrees-of-freedom
   parameter being in the second "plane" of data.

   If the dataset 5th dimension is used to store the voxel-wise statistical
   parameters, then dim[5] must be 1 plus the number of parameters required
   by that distribution (e.g., intent_code=NIFTI_INTENT_TTEST implies dim[5]
   must be 2, as in the example just above).

   Note: intent_code values 2..10 are compatible with AFNI 1.5x (which is
   why there is no code with value=1, which is obsolescent in AFNI).

   OTHER INTENTIONS:
   ----------------
   The purpose of the intent_* fields is to help interpret the values
   stored in the dataset.  Some non-statistical values for intent_code
   and conventions are provided for storing other complex data types.

   The intent_name field provides space for a 15 character (plus 0 byte)
   'name' string for the type of data stored. Examples:
    - intent_code = NIFTI_INTENT_ESTIMATE; intent_name = "T1";
       could be used to signify that the voxel values are estimates of the
       NMR parameter T1.
    - intent_code = NIFTI_INTENT_TTEST; intent_name = "House";
       could be used to signify that the voxel values are t-statistics
       for the significance of 'activation' response to a House stimulus.
    - intent_code = NIFTI_INTENT_DISPVECT; intent_name = "ToMNI152";
       could be used to signify that the voxel values are a displacement
       vector that transforms each voxel (x,y,z) location to the
       corresponding location in the MNI152 standard brain.
    - intent_code = NIFTI_INTENT_SYMMATRIX; intent_name = "DTI";
       could be used to signify that the voxel values comprise a diffusion
       tensor image.

   If no data name is implied or needed, intent_name[0] should be set to 0.
-----------------------------------------------------------------------------*/

 /*! default: no intention is indicated in the header. */

#define NIFTI_INTENT_NONE        0

    /*-------- These codes are for probability distributions ---------------*/
    /* Most distributions have a number of parameters,
       below denoted by p1, p2, and p3, and stored in
        - intent_p1, intent_p2, intent_p3 if dataset doesn't have 5th dimension
        - image data array                if dataset does have 5th dimension

       Functions to compute with many of the distributions below can be found
       in the CDF library from U Texas.

       Formulas for and discussions of these distributions can be found in the
       following books:

        [U] Univariate Discrete Distributions,
            NL Johnson, S Kotz, AW Kemp.

        [C1] Continuous Univariate Distributions, vol. 1,
             NL Johnson, S Kotz, N Balakrishnan.

        [C2] Continuous Univariate Distributions, vol. 2,
             NL Johnson, S Kotz, N Balakrishnan.                            */
    /*----------------------------------------------------------------------*/

  /*! [C2, chap 32] Correlation coefficient R (1 param):
       p1 = degrees of freedom
       R/sqrt(1-R*R) is t-distributed with p1 DOF. */

/*! \defgroup NIFTI1_INTENT_CODES
    \brief nifti1 intent codes, to describe intended meaning of dataset contents
    @{
 */
#define NIFTI_INTENT_CORREL      2

  /*! [C2, chap 28] Student t statistic (1 param): p1 = DOF. */

#define NIFTI_INTENT_TTEST       3

  /*! [C2, chap 27] Fisher F statistic (2 params):
       p1 = numerator DOF, p2 = denominator DOF. */

#define NIFTI_INTENT_FTEST       4

  /*! [C1, chap 13] Standard normal (0 params): Density = N(0,1). */

#define NIFTI_INTENT_ZSCORE      5

  /*! [C1, chap 18] Chi-squared (1 param): p1 = DOF.
      Density(x) proportional to exp(-x/2) * x^(p1/2-1). */

#define NIFTI_INTENT_CHISQ       6

  /*! [C2, chap 25] Beta distribution (2 params): p1=a, p2=b.
      Density(x) proportional to x^(a-1) * (1-x)^(b-1). */

#define NIFTI_INTENT_BETA        7

  /*! [U, chap 3] Binomial distribution (2 params):
       p1 = number of trials, p2 = probability per trial.
      Prob(x) = (p1 choose x) * p2^x * (1-p2)^(p1-x), for x=0,1,...,p1. */

#define NIFTI_INTENT_BINOM       8

  /*! [C1, chap 17] Gamma distribution (2 params):
       p1 = shape, p2 = scale.
      Density(x) proportional to x^(p1-1) * exp(-p2*x). */

#define NIFTI_INTENT_GAMMA       9

  /*! [U, chap 4] Poisson distribution (1 param): p1 = mean.
      Prob(x) = exp(-p1) * p1^x / x! , for x=0,1,2,.... */

#define NIFTI_INTENT_POISSON    10

  /*! [C1, chap 13] Normal distribution (2 params):
       p1 = mean, p2 = standard deviation. */

#define NIFTI_INTENT_NORMAL     11

  /*! [C2, chap 30] Noncentral F statistic (3 params):
       p1 = numerator DOF, p2 = denominator DOF,
       p3 = numerator noncentrality parameter.  */

#define NIFTI_INTENT_FTEST_NONC 12

  /*! [C2, chap 29] Noncentral chi-squared statistic (2 params):
       p1 = DOF, p2 = noncentrality parameter.     */

#define NIFTI_INTENT_CHISQ_NONC 13

  /*! [C2, chap 23] Logistic distribution (2 params):
       p1 = location, p2 = scale.
      Density(x) proportional to sech^2((x-p1)/(2*p2)). */

#define NIFTI_INTENT_LOGISTIC   14

  /*! [C2, chap 24] Laplace distribution (2 params):
       p1 = location, p2 = scale.
      Density(x) proportional to exp(-abs(x-p1)/p2). */

#define NIFTI_INTENT_LAPLACE    15

  /*! [C2, chap 26] Uniform distribution: p1 = lower end, p2 = upper end. */

#define NIFTI_INTENT_UNIFORM    16

  /*! [C2, chap 31] Noncentral t statistic (2 params):
       p1 = DOF, p2 = noncentrality parameter. */

#define NIFTI_INTENT_TTEST_NONC 17

  /*! [C1, chap 21] Weibull distribution (3 params):
       p1 = location, p2 = scale, p3 = power.
      Density(x) proportional to
       ((x-p1)/p2)^(p3-1) * exp(-((x-p1)/p2)^p3) for x > p1. */

#define NIFTI_INTENT_WEIBULL    18

  /*! [C1, chap 18] Chi distribution (1 param): p1 = DOF.
      Density(x) proportional to x^(p1-1) * exp(-x^2/2) for x > 0.
       p1 = 1 = 'half normal' distribution
       p1 = 2 = Rayleigh distribution
       p1 = 3 = Maxwell-Boltzmann distribution.                  */

#define NIFTI_INTENT_CHI        19

  /*! [C1, chap 15] Inverse Gaussian (2 params):
       p1 = mu, p2 = lambda
      Density(x) proportional to
       exp(-p2*(x-p1)^2/(2*p1^2*x)) / x^3  for x > 0. */

#define NIFTI_INTENT_INVGAUSS   20

  /*! [C2, chap 22] Extreme value type I (2 params):
       p1 = location, p2 = scale
      cdf(x) = exp(-exp(-(x-p1)/p2)). */

#define NIFTI_INTENT_EXTVAL     21

  /*! Data is a 'p-value' (no params). */

#define NIFTI_INTENT_PVAL       22

  /*! Data is ln(p-value) (no params).
      To be safe, a program should compute p = exp(-abs(this_value)).
      The nifti_stats.c library returns this_value
      as positive, so that this_value = -log(p). */


#define NIFTI_INTENT_LOGPVAL    23

  /*! Data is log10(p-value) (no params).
      To be safe, a program should compute p = pow(10.,-abs(this_value)).
      The nifti_stats.c library returns this_value
      as positive, so that this_value = -log10(p). */

#define NIFTI_INTENT_LOG10PVAL  24

  /*! Smallest intent_code that indicates a statistic. */

#define NIFTI_FIRST_STATCODE     2

  /*! Largest intent_code that indicates a statistic. */

#define NIFTI_LAST_STATCODE     24

 /*---------- these values for intent_code aren't for statistics ----------*/

 /*! To signify that the value at each voxel is an estimate
     of some parameter, set intent_code = NIFTI_INTENT_ESTIMATE.
     The name of the parameter may be stored in intent_name.     */

#define NIFTI_INTENT_ESTIMATE  1001

 /*! To signify that the value at each voxel is an index into
     some set of labels, set intent_code = NIFTI_INTENT_LABEL.
     The filename with the labels may stored in aux_file.        */

#define NIFTI_INTENT_LABEL     1002

 /*! To signify that the value at each voxel is an index into the
     NeuroNames labels set, set intent_code = NIFTI_INTENT_NEURONAME. */

#define NIFTI_INTENT_NEURONAME 1003

 /*! To store an M x N matrix at each voxel:
       - dataset must have a 5th dimension (dim[0]=5 and dim[5]>1)
       - intent_code must be NIFTI_INTENT_GENMATRIX
       - dim[5] must be M*N
       - intent_p1 must be M (in float format)
       - intent_p2 must be N (ditto)
       - the matrix values A[i][[j] are stored in row-order:
         - A[0][0] A[0][1] ... A[0][N-1]
         - A[1][0] A[1][1] ... A[1][N-1]
         - etc., until
         - A[M-1][0] A[M-1][1] ... A[M-1][N-1]        */

#define NIFTI_INTENT_GENMATRIX 1004

 /*! To store an NxN symmetric matrix at each voxel:
       - dataset must have a 5th dimension
       - intent_code must be NIFTI_INTENT_SYMMATRIX
       - dim[5] must be N*(N+1)/2
       - intent_p1 must be N (in float format)
       - the matrix values A[i][[j] are stored in row-order:
         - A[0][0]
         - A[1][0] A[1][1]
         - A[2][0] A[2][1] A[2][2]
         - etc.: row-by-row                           */

#define NIFTI_INTENT_SYMMATRIX 1005

 /*! To signify that the vector value at each voxel is to be taken
     as a displacement field or vector:
       - dataset must have a 5th dimension
       - intent_code must be NIFTI_INTENT_DISPVECT
       - dim[5] must be the dimensionality of the displacment
         vector (e.g., 3 for spatial displacement, 2 for in-plane) */

#define NIFTI_INTENT_DISPVECT  1006   /* specifically for displacements */
#define NIFTI_INTENT_VECTOR    1007   /* for any other type of vector */

 /*! To signify that the vector value at each voxel is really a
     spatial coordinate (e.g., the vertices or nodes of a surface mesh):
       - dataset must have a 5th dimension
       - intent_code must be NIFTI_INTENT_POINTSET
       - dim[0] = 5
       - dim[1] = number of points
       - dim[2] = dim[3] = dim[4] = 1
       - dim[5] must be the dimensionality of space (e.g., 3 => 3D space).
       - intent_name may describe the object these points come from
         (e.g., "pial", "gray/white" , "EEG", "MEG").                   */

#define NIFTI_INTENT_POINTSET  1008

 /*! To signify that the vector value at each voxel is really a triple
     of indexes (e.g., forming a triangle) from a pointset dataset:
       - dataset must have a 5th dimension
       - intent_code must be NIFTI_INTENT_TRIANGLE
       - dim[0] = 5
       - dim[1] = number of triangles
       - dim[2] = dim[3] = dim[4] = 1
       - dim[5] = 3
       - datatype should be an integer type (preferably DT_INT32)
       - the data values are indexes (0,1,...) into a pointset dataset. */

#define NIFTI_INTENT_TRIANGLE  1009

 /*! To signify that the vector value at each voxel is a quaternion:
       - dataset must have a 5th dimension
       - intent_code must be NIFTI_INTENT_QUATERNION
       - dim[0] = 5
       - dim[5] = 4
       - datatype should be a floating point type     */

#define NIFTI_INTENT_QUATERNION 1010

 /*! Dimensionless value - no params - although, as in _ESTIMATE
     the name of the parameter may be stored in intent_name.     */

#define NIFTI_INTENT_DIMLESS    1011

 /*---------- these values apply to GIFTI datasets ----------*/

 /*! To signify that the value at each location is from a time series. */

#define NIFTI_INTENT_TIME_SERIES  2001

 /*! To signify that the value at each location is a node index, from
     a complete surface dataset.                                       */

#define NIFTI_INTENT_NODE_INDEX   2002

 /*! To signify that the vector value at each location is an RGB triplet,
     of whatever type.
       - dataset must have a 5th dimension
       - dim[0] = 5
       - dim[1] = number of nodes
       - dim[2] = dim[3] = dim[4] = 1
       - dim[5] = 3
    */

#define NIFTI_INTENT_RGB_VECTOR   2003

 /*! To signify that the vector value at each location is a 4 valued RGBA
     vector, of whatever type.
       - dataset must have a 5th dimension
       - dim[0] = 5
       - dim[1] = number of nodes
       - dim[2] = dim[3] = dim[4] = 1
       - dim[5] = 4
    */

#define NIFTI_INTENT_RGBA_VECTOR  2004

 /*! To signify that the value at each location is a shape value, such
     as the curvature.  */

#define NIFTI_INTENT_SHAPE        2005

/* @} */

/*---------------------------------------------------------------------------*/
/* 3D IMAGE (VOLUME) ORIENTATION AND LOCATION IN SPACE:
   ---------------------------------------------------
   There are 3 different methods by which continuous coordinates can
   attached to voxels.  The discussion below emphasizes 3D volumes, and
   the continuous coordinates are referred to as (x,y,z).  The voxel
   index coordinates (i.e., the array indexes) are referred to as (i,j,k),
   with valid ranges:
     i = 0 .. dim[1]-1
     j = 0 .. dim[2]-1  (if dim[0] >= 2)
     k = 0 .. dim[3]-1  (if dim[0] >= 3)
   The (x,y,z) coordinates refer to the CENTER of a voxel.  In methods
   2 and 3, the (x,y,z) axes refer to a subject-based coordinate system,
   with
     +x = Right  +y = Anterior  +z = Superior.
   This is a right-handed coordinate system.  However, the exact direction
   these axes point with respect to the subject depends on qform_code
   (Method 2) and sform_code (Method 3).

   N.B.: The i index varies most rapidly, j index next, k index slowest.
    Thus, voxel (i,j,k) is stored starting at location
      (i + j*dim[1] + k*dim[1]*dim[2]) * (bitpix/8)
    into the dataset array.

   N.B.: The ANALYZE 7.5 coordinate system is
      +x = Left  +y = Anterior  +z = Superior
    which is a left-handed coordinate system.  This backwardness is
    too difficult to tolerate, so this NIFTI-1 standard specifies the
    coordinate order which is most common in functional neuroimaging.

   N.B.: The 3 methods below all give the locations of the voxel centers
    in the (x,y,z) coordinate system.  In many cases, programs will wish
    to display image data on some other grid.  In such a case, the program
    will need to convert its desired (x,y,z) values into (i,j,k) values
    in order to extract (or interpolate) the image data.  This operation
    would be done with the inverse transformation to those described below.

   N.B.: Method 2 uses a factor 'qfac' which is either -1 or 1; qfac is
    stored in the otherwise unused pixdim[0].  If pixdim[0]=0.0 (which
    should not occur), we take qfac=1.  Of course, pixdim[0] is only used
    when reading a NIFTI-1 header, not when reading an ANALYZE 7.5 header.

   N.B.: The units of (x,y,z) can be specified using the xyzt_units field.

   METHOD 1 (the "old" way, used only when qform_code = 0):
   -------------------------------------------------------
   The coordinate mapping from (i,j,k) to (x,y,z) is the ANALYZE
   7.5 way.  This is a simple scaling relationship:

     x = pixdim[1] * i
     y = pixdim[2] * j
     z = pixdim[3] * k

   No particular spatial orientation is attached to these (x,y,z)
   coordinates.  (NIFTI-1 does not have the ANALYZE 7.5 orient field,
   which is not general and is often not set properly.)  This method
   is not recommended, and is present mainly for compatibility with
   ANALYZE 7.5 files.

   METHOD 2 (used when qform_code > 0, which should be the "normal" case):
   ---------------------------------------------------------------------
   The (x,y,z) coordinates are given by the pixdim[] scales, a rotation
   matrix, and a shift.  This method is intended to represent
   "scanner-anatomical" coordinates, which are often embedded in the
   image header (e.g., DICOM fields (0020,0032), (0020,0037), (0028,0030),
   and (0018,0050)), and represent the nominal orientation and location of
   the data.  This method can also be used to represent "aligned"
   coordinates, which would typically result from some post-acquisition
   alignment of the volume to a standard orientation (e.g., the same
   subject on another day, or a rigid rotation to true anatomical
   orientation from the tilted position of the subject in the scanner).
   The formula for (x,y,z) in terms of header parameters and (i,j,k) is:

     [ x ]   [ R11 R12 R13 ] [        pixdim[1] * i ]   [ qoffset_x ]
     [ y ] = [ R21 R22 R23 ] [        pixdim[2] * j ] + [ qoffset_y ]
     [ z ]   [ R31 R32 R33 ] [ qfac * pixdim[3] * k ]   [ qoffset_z ]

   The qoffset_* shifts are in the NIFTI-1 header.  Note that the center
   of the (i,j,k)=(0,0,0) voxel (first value in the dataset array) is
   just (x,y,z)=(qoffset_x,qoffset_y,qoffset_z).

   The rotation matrix R is calculated from the quatern_* parameters.
   This calculation is described below.

   The scaling factor qfac is either 1 or -1.  The rotation matrix R
   defined by the quaternion parameters is "proper" (has determinant 1).
   This may not fit the needs of the data; for example, if the image
   grid is
     i increases from Left-to-Right
     j increases from Anterior-to-Posterior
     k increases from Inferior-to-Superior
   Then (i,j,k) is a left-handed triple.  In this example, if qfac=1,
   the R matrix would have to be

     [  1   0   0 ]
     [  0  -1   0 ]  which is "improper" (determinant = -1).
     [  0   0   1 ]

   If we set qfac=-1, then the R matrix would be

     [  1   0   0 ]
     [  0  -1   0 ]  which is proper.
     [  0   0  -1 ]

   This R matrix is represented by quaternion [a,b,c,d] = [0,1,0,0]
   (which encodes a 180 degree rotation about the x-axis).

   METHOD 3 (used when sform_code > 0):
   -----------------------------------
   The (x,y,z) coordinates are given by a general affine transformation
   of the (i,j,k) indexes:

     x = srow_x[0] * i + srow_x[1] * j + srow_x[2] * k + srow_x[3]
     y = srow_y[0] * i + srow_y[1] * j + srow_y[2] * k + srow_y[3]
     z = srow_z[0] * i + srow_z[1] * j + srow_z[2] * k + srow_z[3]

   The srow_* vectors are in the NIFTI_1 header.  Note that no use is
   made of pixdim[] in this method.

   WHY 3 METHODS?
   --------------
   Method 1 is provided only for backwards compatibility.  The intention
   is that Method 2 (qform_code > 0) represents the nominal voxel locations
   as reported by the scanner, or as rotated to some fiducial orientation and
   location.  Method 3, if present (sform_code > 0), is to be used to give
   the location of the voxels in some standard space.  The sform_code
   indicates which standard space is present.  Both methods 2 and 3 can be
   present, and be useful in different contexts (method 2 for displaying the
   data on its original grid; method 3 for displaying it on a standard grid).

   In this scheme, a dataset would originally be set up so that the
   Method 2 coordinates represent what the scanner reported.  Later,
   a registration to some standard space can be computed and inserted
   in the header.  Image display software can use either transform,
   depending on its purposes and needs.

   In Method 2, the origin of coordinates would generally be whatever
   the scanner origin is; for example, in MRI, (0,0,0) is the center
   of the gradient coil.

   In Method 3, the origin of coordinates would depend on the value
   of sform_code; for example, for the Talairach coordinate system,
   (0,0,0) corresponds to the Anterior Commissure.

   QUATERNION REPRESENTATION OF ROTATION MATRIX (METHOD 2)
   -------------------------------------------------------
   The orientation of the (x,y,z) axes relative to the (i,j,k) axes
   in 3D space is specified using a unit quaternion [a,b,c,d], where
   a*a+b*b+c*c+d*d=1.  The (b,c,d) values are all that is needed, since
   we require that a = sqrt(1.0-(b*b+c*c+d*d)) be nonnegative.  The (b,c,d)
   values are stored in the (quatern_b,quatern_c,quatern_d) fields.

   The quaternion representation is chosen for its compactness in
   representing rotations. The (proper) 3x3 rotation matrix that
   corresponds to [a,b,c,d] is

         [ a*a+b*b-c*c-d*d   2*b*c-2*a*d       2*b*d+2*a*c     ]
     R = [ 2*b*c+2*a*d       a*a+c*c-b*b-d*d   2*c*d-2*a*b     ]
         [ 2*b*d-2*a*c       2*c*d+2*a*b       a*a+d*d-c*c-b*b ]

         [ R11               R12               R13             ]
       = [ R21               R22               R23             ]
         [ R31               R32               R33             ]

   If (p,q,r) is a unit 3-vector, then rotation of angle h about that
   direction is represented by the quaternion

     [a,b,c,d] = [cos(h/2), p*sin(h/2), q*sin(h/2), r*sin(h/2)].

   Requiring a >= 0 is equivalent to requiring -Pi <= h <= Pi.  (Note that
   [-a,-b,-c,-d] represents the same rotation as [a,b,c,d]; there are 2
   quaternions that can be used to represent a given rotation matrix R.)
   To rotate a 3-vector (x,y,z) using quaternions, we compute the
   quaternion product

     [0,x',y',z'] = [a,b,c,d] * [0,x,y,z] * [a,-b,-c,-d]

   which is equivalent to the matrix-vector multiply

     [ x' ]     [ x ]
     [ y' ] = R [ y ]   (equivalence depends on a*a+b*b+c*c+d*d=1)
     [ z' ]     [ z ]

   Multiplication of 2 quaternions is defined by the following:

     [a,b,c,d] = a*1 + b*I + c*J + d*K
     where
       I*I = J*J = K*K = -1 (I,J,K are square roots of -1)
       I*J =  K    J*K =  I    K*I =  J
       J*I = -K    K*J = -I    I*K = -J  (not commutative!)
     For example
       [a,b,0,0] * [0,0,0,1] = [0,0,-b,a]
     since this expands to
       (a+b*I)*(K) = (a*K+b*I*K) = (a*K-b*J).

   The above formula shows how to go from quaternion (b,c,d) to
   rotation matrix and direction cosines.  Conversely, given R,
   we can compute the fields for the NIFTI-1 header by

     a = 0.5  * sqrt(1+R11+R22+R33)    (not stored)
     b = 0.25 * (R32-R23) / a       => quatern_b
     c = 0.25 * (R13-R31) / a       => quatern_c
     d = 0.25 * (R21-R12) / a       => quatern_d

   If a=0 (a 180 degree rotation), alternative formulas are needed.
   See the nifti1_io.c function mat44_to_quatern() for an implementation
   of the various cases in converting R to [a,b,c,d].

   Note that R-transpose (= R-inverse) would lead to the quaternion
   [a,-b,-c,-d].

   The choice to specify the qoffset_x (etc.) values in the final
   coordinate system is partly to make it easy to convert DICOM images to
   this format.  The DICOM attribute "Image Position (Patient)" (0020,0032)
   stores the (Xd,Yd,Zd) coordinates of the center of the first voxel.
   Here, (Xd,Yd,Zd) refer to DICOM coordinates, and Xd=-x, Yd=-y, Zd=z,
   where (x,y,z) refers to the NIFTI coordinate system discussed above.
   (i.e., DICOM +Xd is Left, +Yd is Posterior, +Zd is Superior,
        whereas +x is Right, +y is Anterior  , +z is Superior. )
   Thus, if the (0020,0032) DICOM attribute is extracted into (px,py,pz), then
     qoffset_x = -px   qoffset_y = -py   qoffset_z = pz
   is a reasonable setting when qform_code=NIFTI_XFORM_SCANNER_ANAT.

   That is, DICOM's coordinate system is 180 degrees rotated about the z-axis
   from the neuroscience/NIFTI coordinate system.  To transform between DICOM
   and NIFTI, you just have to negate the x- and y-coordinates.

   The DICOM attribute (0020,0037) "Image Orientation (Patient)" gives the
   orientation of the x- and y-axes of the image data in terms of 2 3-vectors.
   The first vector is a unit vector along the x-axis, and the second is
   along the y-axis.  If the (0020,0037) attribute is extracted into the
   value (xa,xb,xc,ya,yb,yc), then the first two columns of the R matrix
   would be
              [ -xa  -ya ]
              [ -xb  -yb ]
              [  xc   yc ]
   The negations are because DICOM's x- and y-axes are reversed relative
   to NIFTI's.  The third column of the R matrix gives the direction of
   displacement (relative to the subject) along the slice-wise direction.
   This orientation is not encoded in the DICOM standard in a simple way;
   DICOM is mostly concerned with 2D images.  The third column of R will be
   either the cross-product of the first 2 columns or its negative.  It is
   possible to infer the sign of the 3rd column by examining the coordinates
   in DICOM attribute (0020,0032) "Image Position (Patient)" for successive
   slices.  However, this method occasionally fails for reasons that I
   (RW Cox) do not understand.
-----------------------------------------------------------------------------*/

   /* [qs]form_code value:  */      /* x,y,z coordinate system refers to:    */
   /*-----------------------*/      /*---------------------------------------*/

/*! \defgroup NIFTI1_XFORM_CODES
    \brief nifti1 xform codes to describe the "standard" coordinate system
    @{
 */

#define NIFTI_XFORM_UNKNOWN      0
#define NIFTI_XFORM_SCANNER_ANAT 1
#define NIFTI_XFORM_ALIGNED_ANAT 2
#define NIFTI_XFORM_TALAIRACH    3
#define NIFTI_XFORM_MNI_152      4
/* @} */

/*---------------------------------------------------------------------------*/
/* UNITS OF SPATIAL AND TEMPORAL DIMENSIONS:
   ----------------------------------------
   The codes below can be used in xyzt_units to indicate the units of pixdim.
   As noted earlier, dimensions 1,2,3 are for x,y,z; dimension 4 is for
   time (t).
    - If dim[4]=1 or dim[0] < 4, there is no time axis.
    - A single time series (no space) would be specified with
      - dim[0] = 4 (for scalar data) or dim[0] = 5 (for vector data)
      - dim[1] = dim[2] = dim[3] = 1
      - dim[4] = number of time points
      - pixdim[4] = time step
      - xyzt_units indicates units of pixdim[4]
      - dim[5] = number of values stored at each time point

   Bits 0..2 of xyzt_units specify the units of pixdim[1..3]
    (e.g., spatial units are values 1..7).
   Bits 3..5 of xyzt_units specify the units of pixdim[4]
    (e.g., temporal units are multiples of 8).

   This compression of 2 distinct concepts into 1 byte is due to the
   limited space available in the 348 byte ANALYZE 7.5 header.  The
   macros XYZT_TO_SPACE and XYZT_TO_TIME can be used to mask off the
   undesired bits from the xyzt_units fields, leaving "pure" space
   and time codes.  Inversely, the macro SPACE_TIME_TO_XYZT can be
   used to assemble a space code (0,1,2,...,7) with a time code
   (0,8,16,32,...,56) into the combined value for xyzt_units.

   Note that codes are provided to indicate the "time" axis units are
   actually frequency in Hertz (_HZ), in part-per-million (_PPM)
   or in radians-per-second (_RADS).

   The toffset field can be used to indicate a nonzero start point for
   the time axis.  That is, time point #m is at t=toffset+m*pixdim[4]
   for m=0..dim[4]-1.
-----------------------------------------------------------------------------*/

/*! \defgroup NIFTI1_UNITS
    \brief nifti1 units codes to describe the unit of measurement for
           each dimension of the dataset
    @{
 */
#define NIFTI_UNITS_UNKNOWN 0
#define NIFTI_UNITS_METER   1
#define NIFTI_UNITS_MM      2
#define NIFTI_UNITS_MICRON  3
#define NIFTI_UNITS_SEC     8
#define NIFTI_UNITS_MSEC   16
#define NIFTI_UNITS_USEC   24
#define NIFTI_UNITS_HZ     32
#define NIFTI_UNITS_PPM    40
#define NIFTI_UNITS_RADS   48
/* @} */

#undef  XYZT_TO_SPACE
#undef  XYZT_TO_TIME
#define XYZT_TO_SPACE(xyzt)       ( (xyzt) & 0x07 )
#define XYZT_TO_TIME(xyzt)        ( (xyzt) & 0x38 )

#undef  SPACE_TIME_TO_XYZT
#define SPACE_TIME_TO_XYZT(ss,tt) (  (((char)(ss)) & 0x07)   \
                                   | (((char)(tt)) & 0x38) )

/*---------------------------------------------------------------------------*/
/* MRI-SPECIFIC SPATIAL AND TEMPORAL INFORMATION:
   ---------------------------------------------
   A few fields are provided to store some extra information
   that is sometimes important when storing the image data
   from an FMRI time series experiment.  (After processing such
   data into statistical images, these fields are not likely
   to be useful.)

  { freq_dim  } = These fields encode which spatial dimension (1,2, or 3)
  { phase_dim } = corresponds to which acquisition dimension for MRI data.
  { slice_dim } =
    Examples:
      Rectangular scan multi-slice EPI:
        freq_dim = 1  phase_dim = 2  slice_dim = 3  (or some permutation)
      Spiral scan multi-slice EPI:
        freq_dim = phase_dim = 0  slice_dim = 3
        since the concepts of frequency- and phase-encoding directions
        don't apply to spiral scan

    slice_duration = If this is positive, AND if slice_dim is nonzero,
                     indicates the amount of time used to acquire 1 slice.
                     slice_duration*dim[slice_dim] can be less than pixdim[4]
                     with a clustered acquisition method, for example.

    slice_code = If this is nonzero, AND if slice_dim is nonzero, AND
                 if slice_duration is positive, indicates the timing
                 pattern of the slice acquisition.  The following codes
                 are defined:
                   NIFTI_SLICE_SEQ_INC  == sequential increasing
                   NIFTI_SLICE_SEQ_DEC  == sequential decreasing
                   NIFTI_SLICE_ALT_INC  == alternating increasing
                   NIFTI_SLICE_ALT_DEC  == alternating decreasing
                   NIFTI_SLICE_ALT_INC2 == alternating increasing #2
                   NIFTI_SLICE_ALT_DEC2 == alternating decreasing #2
  { slice_start } = Indicates the start and end of the slice acquisition
  { slice_end   } = pattern, when slice_code is nonzero.  These values
                    are present to allow for the possible addition of
                    "padded" slices at either end of the volume, which
                    don't fit into the slice timing pattern.  If there
                    are no padding slices, then slice_start=0 and
                    slice_end=dim[slice_dim]-1 are the correct values.
                    For these values to be meaningful, slice_start must
                    be non-negative and slice_end must be greater than
                    slice_start.  Otherwise, they should be ignored.

  The following table indicates the slice timing pattern, relative to
  time=0 for the first slice acquired, for some sample cases.  Here,
  dim[slice_dim]=7 (there are 7 slices, labeled 0..6), slice_duration=0.1,
  and slice_start=1, slice_end=5 (1 padded slice on each end).

  slice
  index  SEQ_INC SEQ_DEC ALT_INC ALT_DEC ALT_INC2 ALT_DEC2
    6  :   n/a     n/a     n/a     n/a    n/a      n/a    n/a = not applicable
    5  :   0.4     0.0     0.2     0.0    0.4      0.2    (slice time offset
    4  :   0.3     0.1     0.4     0.3    0.1      0.0     doesn't apply to
    3  :   0.2     0.2     0.1     0.1    0.3      0.3     slices outside
    2  :   0.1     0.3     0.3     0.4    0.0      0.1     the range
    1  :   0.0     0.4     0.0     0.2    0.2      0.4     slice_start ..
    0  :   n/a     n/a     n/a     n/a    n/a      n/a     slice_end)

  The SEQ slice_codes are sequential ordering (uncommon but not unknown),
  either increasing in slice number or decreasing (INC or DEC), as
  illustrated above.

  The ALT slice codes are alternating ordering.  The 'standard' way for
  these to operate (without the '2' on the end) is for the slice timing
  to start at the edge of the slice_start .. slice_end group (at slice_start
  for INC and at slice_end for DEC).  For the 'ALT_*2' slice_codes, the
  slice timing instead starts at the first slice in from the edge (at
  slice_start+1 for INC2 and at slice_end-1 for DEC2).  This latter
  acquisition scheme is found on some Siemens scanners.

  The fields freq_dim, phase_dim, slice_dim are all squished into the single
  byte field dim_info (2 bits each, since the values for each field are
  limited to the range 0..3).  This unpleasantness is due to lack of space
  in the 348 byte allowance.

  The macros DIM_INFO_TO_FREQ_DIM, DIM_INFO_TO_PHASE_DIM, and
  DIM_INFO_TO_SLICE_DIM can be used to extract these values from the
  dim_info byte.

  The macro FPS_INTO_DIM_INFO can be used to put these 3 values
  into the dim_info byte.
-----------------------------------------------------------------------------*/

/*! \defgroup NIFTI1_SLICE_ORDER
    \brief nifti1 slice order codes, describing the acquisition order
           of the slices
    @{
 */
#define NIFTI_SLICE_UNKNOWN   0
#define NIFTI_SLICE_SEQ_INC   1
#define NIFTI_SLICE_SEQ_DEC   2
#define NIFTI_SLICE_ALT_INC   3
#define NIFTI_SLICE_ALT_DEC   4
#define NIFTI_SLICE_ALT_INC2  5  /* 05 May 2005: RWCox */
#define NIFTI_SLICE_ALT_DEC2  6  /* 05 May 2005: RWCox */
/* @} */

#endif /* _NIFTI_HEADER_ */
