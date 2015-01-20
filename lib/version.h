#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.0.7-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.0.7-release"
	#else
		#define __version__ "3.0.7"
	#endif
#endif
