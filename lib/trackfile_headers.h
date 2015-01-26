
#ifndef TRACKFILE_HEADERS_H
#define TRACKFILE_HEADERS_H

//id_string[6]    char 6       ID string for track file. The first 5
//                             characters must be "TRACK".
//dim[3]          short int    Dimension of the image volume.
//voxel_size[3]   float        Voxel size of the image volume.
//origin[3]       float        Origin of the image volume. This field is not yet
//                             being used by TrackVis. That means the origin is
//                             always (0, 0, 0).
//n_scalars    short int 2     Number of scalars saved at each track point
//                             (besides x, y and z coordinates).
//scalar_name[10][20] char     Name of each scalar. Can not be longer than
//                             20 characters each.
//n_properties    short int    Number of properties saved at each track.
//property_name[10][20] char   Name of each property. Can not be longer than
//                             20 characters each. Can only store up to 10 names.
//vox_to_ras[4][4] float       4x4 matrix for voxel to RAS (crs to xyz)
//                             transformation. If vox_to_ras[3][3] is 0, it
//                             means the matrix is not recorded. This field is
//                             added from version 2.
//reserved[444]    char        Reserved space for future version.
//voxel_order[4]   char        Storing order of the original image data. Explained here.
//pad2[4]    char 4            Paddings.
//image_orientation_patient[6] float 24 Image orientation of the original image.
//                             As defined in the DICOM header.
//pad1[2]    char 2            Paddings.
//invert_x   unsigned char    1 Inversion/rotation flags used to generate
//           this track file. For internal use only.
//invert_y   unsigned char 1 As above.
//invert_z   unsigned char 1 As above.
//swap_xy    unsigned char 1 As above.
//swap_yz    unsigned char 1 As above.
//swap_zx    unsigned char 1 As above.
//n_count    int 4             Number of tracks stored in this track file.
//                             0 means the number was NOT stored.
//version    int 4             Version number. Current version is 1.
//hdr_size    int 4            Size of the header. Used to determine byte swap.
//                             Should be 1000.
//<---------------------BODY-------------------------->
//Track #1    int 4            Number of points in this track, as m.
//        float (3+n_s)*4      Track Point #1. Contains 3 plus n_s float
//                             numbers. First 3 float numbers are the
//                             x/y/z coordinate of this track point,
//                             followed by n_s float numbers representing
//                             each of its scalars.
//        float(3+n_s)*4       Track Point #2. Same as above.
//...    ...     ...
//        float(3+n_s)*4       Track Point #m. Same as above.
//        float    n_p*4       n_p float numbers representing each
//                             of the properties of this track.
//Track #2     Same as above.
//
//...          Same as above.
//
//Track #n     Same as above.

typedef struct
{
	char id_string[6];

	short int dim[3];
	float voxel_size[3];
	float origin[3];

	short int n_scalars;
	char scalar_name[10][20];

	short int n_properties;
	char property_name[10][20];

	float vox_to_ras[4][4];

	char reserved[444];

	char voxel_order[4];
	char padA4[4];
	float image_orientation_patient[6];
	char padB2[2];

	unsigned char invert_x;
	unsigned char invert_y;
	unsigned char invert_z;
	unsigned char swap_xy;
	unsigned char swap_yz;
	unsigned char swap_zx;

	int n_count;
	int version;
	int hdr_size; //shouldbe 1000

	unsigned char data[0];
} TrkHead;

/* reads a Brainsuite .dft file */
//0	magic	char[8]	An 8 character string designating the filetype
//                  (‘DFC_LE\0\0′ or ‘DFC_BE\0\0′*).
//8	version	uint8[4]	4 unsigned characters, representing the version number
//                  (e.g., 1.0.0.2)
//12	header size	int32	size of the stored header
//16	data start	int32	offset from beginning of file to where the curves
//                          are stored
//20	metadata offset	int32	offset from beginning of file to where the
//                          metadata are stored
//24	subject data offset	int32	offset from beginning of file to where the
//                          subject data are stored (not currently used)
//28	# of contours	int32	number of curves stored in file
//32	# of seedpoints int64	number of seedpoints stored in file
//Cuve0	Curve1	…	CurveNC-1
//
//N	x0	y0	z0	x1	y1	z1	…	xN-1	yN-1	zN-1

typedef struct
{
	char id_string[8];	// "DFT_LE  "
	uint8_t version[4];
	int32_t header_size;
	int32_t data_start;
	int32_t metadata_offset;
	int32_t subject_data_offset;
	int32_t num_contours;
	int64_t seedpoints;
	char padding[24];
} DftHead;

#endif //TRACKFILE_HEADERS_H
#define TRACKS_H
