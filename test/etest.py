#!/bin/env python3
from difflib import unified_diff, Differ
import re
import sys

import subprocess
from subprocess import Popen, PIPE, STDOUT

import os, sys
from waflib.TaskGen import feature, before_method
from waflib.Task import compile_fun_shell
from waflib import Utils, Task, Logs, Options
import waflib
testlock = Utils.threading.Lock()

reg_act = re.compile(r"(?P<backslash>\\)|(?P<dollar>\$\$)|(?P<subst>\$\{(?P<var>\w+)(?P<code>.*?)\})", re.M)

@feature('etest')
@before_method('process_source')
def make_test(self):
	"""Create the extended test task. There can be only one extended test task by task generator."""
	
	if not getattr(self, 'cmd', None):
		return

	g_tsk = self.create_task(name = 'etest')
	
	if getattr(self, 'target', None):
		self.target= Utils.to_list(self.target)
		for x in self.target:
			if isinstance(x, str):
				g_tsk.outputs.append(self.path.find_or_declare(x))
			else:
				x.parent.mkdir() # if a node was given, create the required folders
				g_tsk.outputs.append(x)

	#Expanded search input the Data Directory
	if getattr(self, 'source', None):
		self.source = Utils.to_list(self.source)
		for i, x in enumerate(self.source):
			tmp = self.path.find_resource(x)
			try:
				g_tsk.inputs.append(tmp)
			except:
				print("Error unable to find %s, did you make that a path "
							"relative from the test declaration path?" % x)
				print("Command: ", self.cmd)
				print("Inputs: ", self.source)
				sys.exit(-1)

		# bypass the execution of process_source by setting the source to an empty list
		self.source = []

	if getattr(self, 'correct', None):
		self.correct = Utils.to_list(self.correct)
		g_tsk.correct = [];
		for i, x in enumerate(self.correct):
			tmp = self.path.find_resource(x)
			try:
				g_tsk.correct.append(tmp)
			except:
				print("Error unable to find %s, did you make that a path "
							"relative from the test declaration path?" % x)
				print("Command: ", self.cmd)
				print("Correct: ", self.correct)
				sys.exit(-1)


	for pp in ['data', 'cwd', 'cmptrg', 'regex', 'invert']:
		if getattr(self, pp, None):
			setattr(g_tsk, pp, getattr(self, pp, None));

	#parse cmd 
	def repl(match):
		g_tsk 
		g = match.group
		out = ""
		if g('dollar'): 
			out = "$"
		elif g('subst'):
			#replace with paths
			if g('var') == 'SRC':
				out = eval('g_tsk.inputs' + g('code'))
			elif g('var') == 'TGT':
				out = eval('g_tsk.outputs ' + g('code'))
			else:
				out = eval(g('var')+g('code'))
			
			# list[str] or list[Node] -> str
			tmp = ""
			if isinstance(out, list):
				# Node -> str
				for i, x in enumerate(out):
					try:
						tmp = tmp + " " + x.abspath();
					except:
						tmp = tmp + " " + str(x)
				out = tmp
			else:		
				# Node -> str
				try:
					out = str(out.abspath());
				except:
					out = str(out)
		return out
	
	g_tsk.cmd = reg_act.sub(repl, self.cmd)

class etest(Task.Task):
	"""
	Execute a extended test
	"""
	cmd = ''
	color = 'PINK'
	after = ['vnum', 'inst']
	vars = []
	data = []
	def runnable_status(self):
		"""
		Always execute the task if `waf --alltests` was used or no
                tests if ``waf --notests`` was used
		"""
		if getattr(Options.options, 'no_tests', False):
			return Task.SKIP_ME

		try:
			ret = super(etest, self).runnable_status()
		except:
			print("Hmm failure, its possible one of the \"source\" files doesn't exist")
			print(self.inputs)
			return Task.SKIP_ME

		if ret == Task.SKIP_ME:
			if getattr(Options.options, 'all_tests', False):
				return Task.RUN_ME
		return ret

	def run(self):
		"""
		Execute the test. The execution is always successful, but the results
		are stored on ``self.generator.bld.etest_results`` for postprocessing.
		"""

		if not getattr(self, 'cmd', None):
			return

		# Set up path variable from library path of all the source 
		# files (usually this will be just one executable which we compiled)
		def add_path(dct, path, var):
			dct[var] = os.pathsep.join(Utils.to_list(path) + [os.environ.get(var, '')])

		fu = os.environ.copy()
		lst = []
		for ss in self.inputs:
			snm = str(ss)
			try:
				tgen = ss.ctx.get_tgen_by_name(snm)
				basepath = ss.ctx.bldnode.abspath()
				paths = tgen.env['LIBPATH']
			except:
				paths = []
			
			for pp in paths:
				lst = lst+[os.path.join(basepath, pp)]
	
		if Utils.is_win32:
			add_path(fu, lst, 'PATH')
		elif Utils.unversioned_sys_platform() == 'darwin':
			add_path(fu, lst, 'DYLD_LIBRARY_PATH')
			add_path(fu, lst, 'LD_LIBRARY_PATH')
		else:
			add_path(fu, lst, 'LD_LIBRARY_PATH')

		#create empty outputs
		for oo in self.outputs:
			f = open(oo.abspath(), 'w')
			f.close()

		# Set up comparison of a particular target
		cmptarget = None;
		if getattr(self, 'cmptrg', None):
			cmptarget = self.outputs[0].abspath()
		
		tup = extendedtest(regex = getattr(self, 'regex', None), 
				command = getattr(self, 'cmd', None),
				invert = getattr(self, 'invert', None),
				gold = getattr(self, 'correct', None), 
				testfile = cmptarget, env = fu)
		
		self.generator.etest_result = tup

		testlock.acquire()
		try:
			bld = self.generator.bld
			Logs.debug("ut: %r", tup)
			try:
				bld.etest_results.append(tup)
			except AttributeError:
				bld.etest_results = [tup]
		finally:
			testlock.release()
		

def summary(bld):
	"""
	Display an execution summary::

		def build(bld):
			bld(features='cxx cxxprogram etest', source='main.c', target='app')
			from etest import 
			bld.add_post_fun(waf_unit_test.summary)
	"""
	lst = getattr(bld, 'etest_results', [])
	if lst:
		Logs.pprint('CYAN', 'execution summary')

		total = len(lst)
		tfail = len([x for x in lst if not x[0]])

		Logs.pprint('CYAN', '  tests that pass %d/%d' % (total-tfail, total))
		for (succ, cmd, ret, out, err) in lst:
			if succ:
				Logs.pprint('GREEN', '    %s' % ' '.join(cmd))

		Logs.pprint('CYAN', '  tests that fail %d/%d' % (tfail, total))
		for (succ, cmd, ret, out, err) in lst:
			if not succ:
				Logs.pprint('CYAN', '    %s' % ' '.join(cmd))
				Logs.pprint('YELLOW', '    %s' % out)
				Logs.pprint('RED', '    %s' % err)

def set_exit_code(bld):
	"""
	If any of the tests fail waf will exit with that exit code.
	This is useful if you have an automated build system which need
	to report on errors from the tests.
	You may use it like this:

	def build(bld):
		bld(features='cxx cxxprogram test', source='main.c', target='app')
		from waflib.Tools import waf_unit_test
		bld.add_post_fun(waf_unit_test.set_exit_code)
	"""
	lst = getattr(bld, 'etest_results', [])
	for (succ, cmd, ret, out, err) in lst:
		if succ:
			msg = []
			if out:
				msg.append('stdout:%s%s' % (os.linesep, out))
			if err:
				msg.append('stderr:%s%s' % (os.linesep, err))
			bld.fatal(os.linesep.join(msg))


def extendedtest(regex = None, gold = None, invert = False, command = 'echo', 
				testfile = None, env = None):
	# Run command, and gather the output
	if isinstance(command, str):
		command = command.strip()
		command = re.split('\s+', command)

	invert = bool(invert)

	success = True
	print("Command: ", command);
	proc = Popen(command, env = env, stdout = PIPE, stderr = PIPE)
	(stdo, stde) = out = proc.communicate()
	stdo = stdo.decode('utf-8')
	stde = stde.decode('utf-8')
	out = stdo

	if testfile:
		try:
			tfile = open(testfile, 'r')
			out = tfile.read()
			tfile.close()
		except: 
			stde = stde + '\nFile not created!: \'' + testfile + '\'\n'
			success = False
	
	#compare with regex
	if regex:
		reg = re.compile(regex, flags = re.MULTILINE | re.DOTALL)
		result = reg.search(out)
		if result is None:
			stde = stde + '\nFailed to find \'' + regex + '\'\n'
			success = False
	
	#compare output with gold standard
	if gold:
		if testfile:
			frm = testfile
		else:
			frm = 'stdout'

		for gg in gold:
			to = gg.abspath()
			goldfile = open(gg.abspath())
			gg = goldfile.read()
			goldfile.close()

			diff = False
			diffs = []
			for ll in unified_diff(out, gg, fromfile = frm, tofile = to):
				diff = True
				diffs.append(ll)

			if diff:
				stde = stde + '\nDifference from \'' + frm + '\'\n'
				stde = stde + '\n' + ' '.join(diffs)
				success = False
	
	if not gold and not regex:
		success = (proc.returncode == 0)

	return (invert^success, command, proc.returncode, stdo, stde)

def options(opt):
	"""
		Provide the ``--alltests``, ``--notests`` and ``--testcmd`` command-line options.
	"""
	opt.add_option('--notests', action='store_true', default=False, help='Exec no unit tests', dest='no_tests')
	opt.add_option('--alltests', action='store_true', default=False, help='Exec all unit tests', dest='all_tests')
	opt.add_option('--testcmd', action='store', default=False,
					help = 'Run the unit tests using the test-cmd string'
					' example "--test-cmd="valgrind --error-exitcode=1'
					' %s" to run under valgrind', dest='testcmd')

def build(bld):
	bld.add_post_fun(summary)
