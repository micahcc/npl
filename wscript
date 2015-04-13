#!/bin/env python from waflib.Utils import subprocess
import os
from waflib import Options, Node, Build, Configure
import re

out = 'build'

def configure(conf):
    conf.find_program('strip')

    join = os.path.join
    isabs = os.path.isabs
    abspath = os.path.abspath

    opts = vars(conf.options)
    conf.load('compiler_cxx compiler_c python waf_unit_test')

    env = conf.env
    conf.env.RPATH = []
    if opts['enable_rpath'] or opts['enable_build_rpath']:
        conf.env.RPATH.append('$ORIGIN/../lib')
        conf.env.RPATH.append('$ORIGIN/../deps/mathparse/')
        conf.env.RPATH.append('$ORIGIN/../deps/optimizers/lib')

    if opts['enable_rpath'] or opts['enable_install_rpath']:
        conf.env.RPATH.append('$ORIGIN/../lib')

    conf.env.LINKFLAGS = ['-lm']
    conf.env.DEFINES = ['_LARGEFILE64_SOURCE=1']
    conf.env.CXXFLAGS = ['-Wall', '-Wextra', '-std=c++11', '-Wno-sign-compare']

    conf.env.STATIC_LINK = False
    if opts['static']:
        conf.env.LIBPOST = 'Static'
    else:
        conf.env.LIBPOST = 'Dyn'

    if opts['profile']:
        conf.env.DEFINES.append('DEBUG=1')
        conf.env.CXXFLAGS.extend(['-g', '-pg'])
        conf.env.LINKFLAGS.append('-pg')
    elif opts['debug']:
        conf.env.DEFINES.append('DEBUG=1')
        conf.env.CXXFLAGS.extend(['-g'])
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
#    conf.check_cfg(package='zlib', uselib_store='ZLIB',
#                args=['--cflags', '--libs'])
    conf.check_cfg(package='fftw3', uselib_store='FFTW',
                args=['--cflags', '--libs'])
#    conf.check_cfg(package='eigen3', uselib_store='EIGEN',
#                args=['--cflags', '--libs'])

def options(ctx):
    ctx.load('compiler_cxx compiler_c waf_unit_test')

    gr = ctx.get_option_group('configure options')

    gr.add_option('--enable-rpath', action='store_true', default = False, help = 'Set RPATH to build/install dirs')
    gr.add_option('--enable-install-rpath', action='store_true', default = False, help = 'Set RPATH to install dir only')
    gr.add_option('--enable-build-rpath', action='store_true', default = False, help = 'Set RPATH to build dir only')

    gr.add_option('--debug', action='store_true', default = False, help = 'Build with debug flags')
    gr.add_option('--profile', action='store_true', default = False, help = 'Build with debug and profiler flags')
    gr.add_option('--release', action='store_true', default = False, help = 'Build with tuned compiler optimizations')
    gr.add_option('--native', action='store_true', default = False, help = 'Build with highly specific compiler optimizations')
    gr.add_option('--static', action='store_true', default = False, help = 'Statically link programs')

def build(bld):

    # recurse into other wscript files
    bld.recurse('deps lib testing tools deps scripts')

