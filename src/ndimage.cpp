#include "ndimage.h"

/* Functions */
NDImage* readNDImage(std::string filename);
void writeNDImage(NDImage* img, std::string filename);

/* Pre-Compile Certain Image Types */
class NDImageStore<1, float>;
class NDImageStore<1, double>;
class NDImageStore<1, int>;

class NDImageStore<2, float>;
class NDImageStore<2, double>;
class NDImageStore<2, int>;

class NDImageStore<3, float>;
class NDImageStore<3, double>;
class NDImageStore<3, int>;

class NDImageStore<4, float>;
class NDImageStore<4, double>;
class NDImageStore<4, int>;

class NDImageStore<5, float>;
class NDImageStore<5, double>;
class NDImageStore<5, int>;

class NDImageStore<6, float>;
class NDImageStore<6, double>;
class NDImageStore<6, int>;

class NDImageStore<7, float>;
class NDImageStore<7, double>;
class NDImageStore<7, int>;

class NDImageStore<8, float>;
class NDImageStore<8, double>;
class NDImageStore<8, int>;


NDImage* readNDImage(std::string filename)
{
	const size_t BSIZE = 1024*1024; //1M
	auto gz = gzopen(filename.c_str(), "rb");
	gzbuffer(gz, BSIZE);
	
	// read the header
	
	return NULL;
}

void writeNDImage(NDImage* img, std::string filename)
{

	// create nifti image class in memory
	

	// compress in memory
	

	// write
}

/* 
 * Nifti Readers 
 */

NDImage* readNifti1Image(gzFile file)
{
	nifti1_header header;
	assert(sizeof(header) == 348);

	gzread(file, &header, sizeof(header));
	if(strcmp(header.magic, "n+1")) {
		gzrewind(file);
		return NULL;
	}

	// skip unused
	// 10+18+4+2+1
	gzread(file, 

	// valid, go ahead and parse
	
}

NDImage* readNifti2Image(gzFile file)
{
	nifti2_header header;
	assert(sizeof(header) == 540);

	// read header size
	gzread(file, &header, sizeof(header));

	// if not n+1, this isn't a nifti file
	if(strcmp(header.magic, "n+2")) {
		gzclearerr(file);
		gzrewind(file);
		return NULL;
	}

	// valid, go ahead and parse
	
}
