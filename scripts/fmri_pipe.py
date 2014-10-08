#!/usr/bin/python3
import ast
import configparser
import argparse
from subprocess import call
import os
from os.path import abspath, join, isfile, isdir
from sys import exit, stderr
from shutil import copy2

npl='/home/micahc/tools/npl-3.0/'
#nplbfc=join(npl,'bin','nplBiasFieldCorrect')
nplbfc=join(npl,'bin','nplBiasCorrect')
nplmath=join(npl,'bin','nplMath')
npldistcor=join(npl,'bin','fIDistortionCorrect')

def earlier(lhs, rhs):
	if isfile(rhs):
		return os.path.getctime(lhs) < os.path.getctime(rhs)
	else:
		return False

def copy(src, target, force):
	if not earlier(src, target) or force:
		print("Copying %s -> %s" % (src, target))
		copy2(src, target)


def biascorrect(iimg, biasfield="", corrected = "", force = False):
	if not force and (not biasfield or earlier(iimg, biasfield)) and \
				(not corrected or earier(iimg, corrected)):
		print("Skipping Bias Correction of %s, output exists" % iimg)
		return 0

	if biasfield and corrected:
		ret = call([nplbfc,'-i',iimg,'-o',biasfield,'-c',corrected])
	elif biasfield:
		ret = call([nplbfc,'-i',iimg,'-o',biasfield])
	elif corrected:
		ret = call([nplbfc,'-i',iimg,'-c',corrected])

	if ret != 0:
		print("Error During Bias Field Estimation", file=stderr)
		exit(ret)
	else:
		return 0;

def average(inputs, oimg):
	if not newer(inputs, oimg) and not force:
		print("Using cached %s" % oimg)
		return 0

	lookup = 'abcdefghijklmnopqrstuvwxyz'
	cmd = [nplmath,'--out',oimg]
	eq = "("
	for i in range(len(inputs)):
		cmd.extend(['-'+lookup[i],inputs[i]])
		if i != 0:
			eq = eq + '+'
		eq = eq + lookup[i]
	eq = eq+')/'+str(len(inputs))

	ret = call(cmd)
	
	if ret != 0:
		print("Error During Averaging of %s", str(inputs), file=stderr)
		exit(ret)
	else:
		return 0;

def main(fmri, t1, t2, biasfield, biased_image, output, force, no_distortion,
		jac_weight, tps_weight, hist, knot_space, bins, parzen_size, smooth,
		cost, **extra):
	#########################################################################
	# Copy Files to Output
	#########################################################################
	t1avg = ""
	t2avg = ""
	for i in range(len(t1)):
		w = "t1_%i.nii.gz"%i
		copy(t1[i], join(output, w), force)
		t1[i] = w
	for i in range(len(t2)):
		w = "t2_%i.nii.gz"%i
		copy(t2[i], join(output, w), force)
		t2[i] = w
	if fmri: 
		copy(fmri, join(output, "fmri.nii.gz"), force)

	if biasfield: 
		copy(fmri, join(output, "fmri.nii.gz"), force)

	if biased_image: 
		copy(biased_image, join(output, "biased_image.nii.gz"), force)

	os.chdir(output)
	
	#########################################################################
	# Compute Bias Field / Bias Correct
	#########################################################################
	if biased_image:
		# estimate bias field, write
		biascorrect(iimg="biased_image.nii.gz", biasfield="biasfield.nii.gz")
	elif len(t1) > 0 and len(t2) > 0:
		# register first T2 to first T1
		mmregister(fixed = "t1_0.nii.gz", moving = "t2_0.nii.gz", 
				oimg = "t2_0_tmp.nii.gz")
		multiply(inputs = ['t1_0.nii.gz', 't2_0_tmp.nii.gz'],
				oimg = "t1_x_t2.nii.gz")
		biascorrect(iimg='t1_x_t2.nii.gz', biasfield = 'biasfield.nii.gz', 
				force = force)
	elif len(t1) > 0:
		biascorrect(iimg='t1_0.nii.gz', biasfield = 'biasfield.nii.gz')
	elif len(t2) > 0:
		biascorrect(iimg='t2_0.nii.gz', biasfield = 'biasfield.nii.gz')

#	# Apply Bias Correction
#	for ii in range(len(t1imgs)):
#		out = 't1_%i_bc.nii.gz'%ii
#		multiply(inputs = [t1imgs[ii], 'biasfield.nii.gz'], oimg = out)
#	for ii in range(len(t2imgs)):
#		out = 't2_%i_bc.nii.gz'%ii
#		multiply(inputs = [t2imgs[ii], 'biasfield.nii.gz'], oimg = out)
#	multiply(inputs = ['fmri.nii.gz', 'biasfield.nii.gz'], oimg = out)
#
#	#########################################################################
#	# Average T1 and T2 images
#	#########################################################################
#	
#	# register/average T1 images
#	sumimgs = []
#	for ii in range(len(t1imgs)):
#		if ii == 0:
#			sumimgs.append(t1imgs[ii])
#		else:
#			out = 't1_%i_reg.nii.gz' % i
#			intra_reg(fixed = t1imgs[0], moving = t1imgs[ii], oimg = out)
#			sumimgs.appent(out)
#	if len(sumimgs) > 0:
#		average(inputs=sumimgs, output='t1_avg.nii.gz')
#
#	# register/average T2 images
#	sumimgs = []
#	for ii in range(len(t2imgs)):
#		if ii == 0:
#			sumimgs.append(t2imgs[ii])
#		else:
#			out = 't2_%i_reg.nii.gz' % i
#			intra_reg(fixed = t2imgs[0], moving = t2imgs[ii], oimg = out)
#			sumimgs.appent(out)
#	if len(sumimgs) > 0:
#		average(inputs=sumimgs, output="t2_avg.nii.gz")
#
#	##########################################################################
#	# Distortion Correction
#	##########################################################################
#	if args.mconly:
#	else:
#		distcorr(fmri = 'fmri.nii.gz', t1 = 't1_avg.nii.gz', 
#				t2 = 't2_avg.nii.gz', fmap = 'est_fmap.nii.gz', 
#				knotdist = args.dc_knotdist, tps = args.dc_tps, 
#				jac = args.dc_jac, iters = args.dc_iters, 
#				
#				args.tps, args.jac, args.iters, args.
#	
#	$NPLDIR/bin/fIDistortionCorrect --input $in \
#		--t1 $ref --out $out --field-map $fieldmap $justone --debug-level 4 \
#		--knot-dist $dist --mc-only $mc --average ${mc%.nii.gz}_avg.nii.gz \
#		--motion $mcparams --bd-scale $scale \
#	 	--TPSReg $alpha_tps --JacReg $alpha_jac \
#		--bd-iters 10000 --bd-lbfgs-hist $hist \
#		--bd-grad-rescale $rescale \
#	 	$cost $optimizer $smooth --bd-bins $bins \
#		--bd-kern-rad $kern --upscale-avg $upscale \
#	 	--bd-gstop $gstop --bd-fstop $fstop 
#
#

def deparse(example, key, config, section):
	if isinstance(example, bool):
		return config.getboolean(section, key)
	elif isinstance(example, int):
		return  config.getint(section, key)
	elif isinstance(example, float):
		return  config.getfloat(section, key)
	elif isinstance(example, list):
		if len(example) > 0:
			if isinstance(example[0], int):
				return [int(v) for v in config.get(section, key).split(',')]
			elif isinstance(example[0], float):
				return  [float(v) for v in config.get(section, key).split(',')]
			else:
				return [v.strip() for v in config.get(section, key).split(',')]
		else:
			# just assume string
			return [v.strip() for v in config.get(section, key).split(',')]
	else:
		return config['io'][k]


if __name__ == "__main__":

	conf_parser = argparse.ArgumentParser(add_help = False)
	conf_parser.add_argument('-c', '--conf', help = 'Config file', metavar='*.ini')
	args, remaining_argv = conf_parser.parse_known_args()

	io = {
		'fmri' : '',
		't1' : [],
		't2' : [],
		'biasfield' : '',
		'biased_image' : '',
		'output' : ''
	}
	options = {
		'force' : False,
		'no_distortion': False,
		'jac_weight' : 0.000001,
		'tps_weight' : 0.001,
		'hist' : 5,
		'knot_space' : 12,
		'bins' : 128,
		'parzen_size' : 4,
		'smooth' : [2, 1.5, 1, .5, 0],
		'cost' : ['MI', 'MI', 'MI', 'VI', 'VI']
	}

	if args.conf:
		config = configparser.ConfigParser(
				interpolation=configparser.ExtendedInterpolation())
		config.read([args.conf])

		# merge/update defaults
		if 'io' in config.keys(): 
			tmp = dict();
			for k, v in io.items():
				if not config.has_option('io', k):
					continue
				
				tmp[k] = deparse(example=v, key=k, config=config, section='io')
			
			io = dict(list(io.items())+list(tmp.items()))
		if 'options' in config.keys(): 
			tmp = dict();
			for k, v in options.items():
				if not config.has_option('io', k):
					continue

				tmp[k] = deparse(example=v, key=k, config=config, section='option')

			options = dict(list(options.items())+list(tmp.items()))

	parser = argparse.ArgumentParser(parents = [conf_parser])

	parser.set_defaults(**dict(list(io.items())+list(options.items())))
		
	parser = argparse.ArgumentParser(description='fMRI Processing Pipeline. '
		'It is expected that all images are in roughtly the correct place so '
		'that bias-field correction can be applied without registration')

	parser.add_argument('-c', '--conf', help = 'Config file', metavar='*.ini')
	parser.add_argument('-f', '--fmri', metavar='*.nii.gz', type=str, 
			nargs='?', help='fMRI Image To Process.')
	parser.add_argument('-1', '--t1', metavar='*.nii.gz', type=str, nargs='*', 
			help = 'T1 Image(s). If multiple are given all will '
			'be registered to the first, then averaged')
	parser.add_argument('-2', '--t2', metavar='*.nii.gz', type=str, nargs='*', 
			help = 'T2 Image(s). If multiple are given all will '
			'be registered to the first, then averaged. These are used as the '
			'undistored reference rather than the T1, since correlation-based '
			'registration is easier.')
	parser.add_argument('-B', '--biasfield', metavar='*.nii.gz', type=str, 
			nargs='?', help='Bias field image. If this is given then all '
			'images will be divided by this image prior to processing.')
	parser.add_argument('-b', '--biased-image', metavar='*.nii.gz', type=str, 
			nargs='?', help='Examplar image for Bias-Field Estimation. A bias '
			'field will be estimated and applied to other images based on the '
			'bias field in this image.')

	parser.add_argument('-o', '--output', metavar='DIR', type=str, nargs=1, 
			help='Output directory. Note that existing files will be kept and '
			'used if possible to prevent recomputation')

	parser.add_argument('-F', '--force', type=bool, 
			help='Force overwriting of old output')
	parser.add_argument('--no-distortion', type=bool,
			help='Force overwriting of old output')

	parser.add_argument('--jac-weight', type=float, nargs = '?',
			help='Wieght if jacobian-regularization during distortion '
			'correction')
	parser.add_argument('--tps-weight', type=float, nargs = '?',
			help='Weight if thin-plate spline regularization during distortion '
			'correction')
	parser.add_argument('--hist', type=int, nargs = '?',
			help='Number of previous steps to keep for the LBFGS updates '
			'during distortion correction')
	parser.add_argument('--knot-space', type=float, nargs = '?',
			help='Distance between B-Spline knots during '
			'distortion correction')
	parser.add_argument('--bins', type=int, nargs = '?',
			help='Number of bins to use for distribution estimation during '
			'distortion correction')
	parser.add_argument('--parzen-size', type=int, nargs = '?',
			help='Width of parzen-window for distribution estimation during '
			'distortion correction')
	parser.add_argument('--smooth', nargs='*', type=float, 
			help='Smoothing steps to undergo for non-rigid distortion '
			'correction')
	parser.add_argument('--cost', nargs='*', type=float, 
			help='Cost function to use at each step of the non-rigid '
			'distortion correction')
	
	args = parser.parse_args(remaining_argv)

	# check if files exist
	allopts = dict(list(io.items()) + list(options.items()))

	# Check for input t1 and t2
	if not allopts['t1'] and not allopts['t2']: 
		print("Need to provide either T1 or T2 image!", file=stderr)
		exit(-1)
	
	# Check the existence of all input files
	for f in allopts['t1']:
		if not isfile(f):
			print('Error input file: %s does not exist' % f)
			exit(-1)
	for f in allopts['t2']:
		if not isfile(f):
			print('Error input file: %s does not exist' % f)
			exit(-1)
	if not allopts['output']:
		print("Need to provide an output dir (--output)!", file=stderr)
		exit(-1)
	if not isdir(allopts['output']):
		print("output dir is not a directory: %s" % output, file=stderr)
		exit(-1)
	
	if not allopts['fmri']:
		print("Need to provide an fMRI (--fmri)!", file=stderr)
		exit(-1)
	if not isfile(allopts['fmri']):
		print("Input fMRI is not a file: %s" % allopts['fmri'], file=stderr)
		exit(-1)
	
	main(**allopts)
