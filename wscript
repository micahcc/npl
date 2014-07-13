#!/bin/env python
from waflib.Utils import subprocess
import os
from waflib import Options, Node, Build, Configure
import re

out = 'build'

def cmakeparse(cmakefile):
    ifile = open(cmakefile)
    ind = ifile.read()
    ifile.close()
    print(ind)
    libs = re.search('LIBRARIES "([a-zA-Z0-9_;-]*)"', ind).group(1)
    libs = libs.split(";")
    return libs

def configure(conf):
    join = os.path.join
    isabs = os.path.isabs
    abspath = os.path.abspath
    
    opts = vars(conf.options)
    conf.load('compiler_cxx python waf_unit_test')

    env = conf.env

    conf.env.RPATH = []
    if opts['enable_rpath'] or opts['enable_build_rpath']:
        conf.env.RPATH.append('$ORIGIN/../lib')
    
    if opts['enable_rpath'] or opts['enable_install_rpath']:
        conf.env.RPATH.append('$ORIGIN/../lib')
    
    conf.env.DEFINES = ['AE_CPU=AE_INTEL', 'VCL_CAN_STATIC_CONST_INIT_FLOAT=0', 'VCL_CAN_STATIC_CONST_INIT_INT=0']
    conf.env.LINKFLAGS = ['-lm']
    # for static build
    if opts['static']: 
        conf.env.CXXFLAGS = ['-Wall', '-std=c++11', '-static-libgcc', '-static-libstdc++']
        conf.env.LINKFLAGS.extend('-static-libgcc', '-static-libstdc++')
        conf.env.STATIC_LINK = True
    else:
        conf.env.CXXFLAGS = ['-Wall', '-std=c++11']
        conf.env.STATIC_LINK = False


    if opts['profile']:
        conf.env.DEFINES.append('DEBUG=1')
        conf.env.CXXFLAGS.extend(['-Wno-unused-parameter', '-Wno-sign-compare', '-Wno-unused-local-typedefs', '-Wall', '-Wextra','-g', '-pg'])
        conf.env.LINKFLAGS.append('-pg')
    elif opts['debug']:
        conf.env.DEFINES.append('DEBUG=1')
        conf.env.CXXFLAGS.extend(['-Wno-unused-parameter', '-Wno-sign-compare', '-Wno-unused-local-typedefs', '-Wall', '-Wextra','-g'])
    elif opts['release']:
        conf.env.DEFINES.append('NDEBUG=1')
        conf.env.CXXFLAGS.extend(['-O3', '-march=core2'])
    elif opts['native']:
        conf.env.DEFINES.append('NDEBUG=1')
        conf.env.CXXFLAGS.extend(['-O3', '-march=native'])
    
    conf.check(header_name='stdio.h', features='cxx cxxprogram', mandatory=True)

        
    ############################### 
    # Library Configuration
    ############################### 
    conf.check_cfg(atleast_pkgconfig_version='0.0.0')
    conf.check_cfg(package='zlib', uselib_store='ZLIB',
                args=['--cflags', '--libs'])

def options(ctx):
    ctx.load('compiler_cxx')

    gr = ctx.get_option_group('configure options')
    
    gr.add_option('--enable-rpath', action='store_true', default = False, help = 'Set RPATH to build/install dirs')
    gr.add_option('--enable-install-rpath', action='store_true', default = False, help = 'Set RPATH to install dir only')
    gr.add_option('--enable-build-rpath', action='store_true', default = False, help = 'Set RPATH to build dir only')
    
    gr.add_option('--debug', action='store_true', default = False, help = 'Build with debug flags')
    gr.add_option('--profile', action='store_true', default = False, help = 'Build with debug and profiler flags')
    gr.add_option('--release', action='store_true', default = False, help = 'Build with tuned compiler optimizations')
    gr.add_option('--native', action='store_true', default = False, help = 'Build with highly specific compiler optimizations')
    gr.add_option('--static', action='store_true', default = False, help = "Build statically (turns off R and python)")
    
def gitversion():
    if not os.path.isdir(".git"):
        print("This does not appear to be a Git repository.")
        return
    try:
        HEAD = open(".git/HEAD");
        headtxt = HEAD.read();
        HEAD.close();

        headtxt = headtxt.split("ref: ")
        if len(headtxt) == 2:
            fname = ".git/"+headtxt[1].strip();
            master = open(fname);
            mastertxt = master.read().strip();
            master.close();
        else:
            mastertxt = headtxt[0].strip()
    
    except EnvironmentError:
        print("unable to get HEAD")
        return "unknown"
    return mastertxt

def build(bld):
    with open("lib/version.h", "w") as f:
        f.write('#define __version__ "%s"\n\n' % gitversion())
        f.close()

    bld.recurse('deps lib testing tools-WIP deps')
