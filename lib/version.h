#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.1.2-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.1.2-release"
	#else
		#define __version__ "3.1.2"
	#endif
#endif
