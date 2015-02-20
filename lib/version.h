#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.1.3-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.1.3-release"
	#else
		#define __version__ "3.1.3"
	#endif
#endif
