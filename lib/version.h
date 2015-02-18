#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.1.0-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.1.0-release"
	#else
		#define __version__ "3.1.0"
	#endif
#endif
