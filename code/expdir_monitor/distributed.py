from subprocess import Popen, PIPE
from threading import Thread, Lock
from queue import Queue
from time import sleep
from sys import stderr, stdout
import re
import json
import shlex

max_running_machine = 10

_max_used_mem = 0.3
_max_used_gpu = 0.3
config_file = 'server_config'


class GpuChecker:
	def __init__(self, nvidia_getter, gpuid):
		self.nvidia_getter = nvidia_getter
		self.gpuid = gpuid
	
	def state_parser(self, state_str):
		result = []
		for line in state_str.split('\n'):
			# .*?(\d*)C.*\|(.*?)MiB.*?/(.*?)MiB.*?\|.*?(\d*)\%
			# .*?(\d*)C.*\|(.*?)MiB.*?/(.*?)MiB.*?\|.*?(\d*)%
			pattern = re.search('.*?(\d*)C.*\|(.*?)MiB.*?/(.*?)MiB.*?\|.*?(\d*)%', line)
			if pattern is not None:
				result.append([int(x) for x in pattern.groups()])
		if self.gpuid >= len(result):
			return None
		# assert self.gpuid < len(result), 'Parsing error or not enough gpus.'
		return result[self.gpuid]
	
	def instance_available(self, state_str):
		parse_result = self.state_parser(state_str)
		if parse_result is None: return False
		_, used_mem, total_mem, occupation = parse_result
		occupation /= 100
		return used_mem / total_mem < _max_used_mem and occupation < _max_used_gpu
	
	def check(self):
		_check_times = 3
		try:
			for _i in range(_check_times):
				assert self.instance_available(self.nvidia_getter())
				if _i < _check_times - 1:
					sleep(0.5)
		except AssertionError:
			return False
		return True
	
	def is_on(self):
		try:
			parse_result = self.state_parser(self.nvidia_getter())
			if parse_result is None:
				return False
			else:
				return True
		except Exception:
			return False


class RemoteController:
	def __init__(self, remote, gpuid, executive):
		self.remote = remote
		self.gpuid = gpuid
		self.executive = executive
		
		self.gpu_checker = GpuChecker(lambda: self.run('nvidia-smi'), self.gpuid)
		
		self._lock = Lock()
		self._occupied = False
		self._on_running = None
	
	@property
	def occupied(self):
		with self._lock:
			return self._occupied
	
	@occupied.setter
	def occupied(self, val):
		assert isinstance(val, bool), 'Occupied must be True or False, but {} received.'.format(val)
		with self._lock:
			self._occupied = val
	
	def run(self, cmd, stdin=None):
		proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
					 universal_newlines=True)
		return proc.communicate(input=stdin)[0]
	
	@property
	def gpu_state(self):
		return self.gpu_checker.check()
	
	@property
	def exe_cmd(self):
		return 'CUDA_VISIBLE_DEVICES={gpuid} python {executive}'.format(
			executive=self.executive,
			gpuid=self.gpuid
		)
	
	def check_on(self, queue):
		if not self.gpu_checker.is_on():
			if self._on_running is not None:
				queue.put(self._on_running)
				self._on_running = None
				print('Remote Error.')
			return False
		return True
	
	def remote_executer(self, idx, expdir, queue):
		self.occupied = True
		cmd = self.exe_cmd
		print('{}: {} {}'.format(self.remote, cmd, expdir), file=stdout)
		result = self.run(cmd, stdin=expdir)
		try:
			result = str(result).split('\n')
			used_time = result[-3]
			result = result[-2]
			assert result.startswith('valid performance: ') and used_time.startswith('running time: '), \
				'Invalid return: %s, %s' % (used_time, result)
			used_time = used_time[len('running time: '):]
			used_time = float(used_time) / 60  # minutes
			result = result[len('valid performance: '):]
			result = float(result)
			queue.put([idx, (result, used_time)])
			print('{}th task: {} is successfully executed, result is {}, using {} min.'.
				  format(idx, expdir, result, used_time), file=stdout)
		except Exception:
			queue.put([idx, expdir])
			print('{}th task: {} fails, with return: {}.'.format(idx, expdir, result), file=stdout)
		self.occupied = False
	
	def execute(self, idx, expdir, queue):
		if self.occupied or not self.gpu_state:
			queue.put([idx, expdir])
		else:
			self._on_running = [idx, expdir]
			thr = Thread(target=self.remote_executer, args=(idx, expdir, queue))
			thr.start()
			self.occupied = True
			self._on_running = None


class ClusterController:
	def __init__(self, config_list):
		self.cluster = [RemoteController(*config) for config in config_list]
		self._pt = 0
	
	def choice(self, queue):
		remotes_available, occupy_num = self.get_available(queue)
		while occupy_num >= max_running_machine:
			sleep(0.5)
			remotes_available, occupy_num = self.get_available(queue)
		while not remotes_available[self._pt]:
			self._pt = (self._pt + 1) % len(self.cluster)
		choose_remote = self.cluster[self._pt]
		self._pt = (self._pt + 1) % len(self.cluster)
		return choose_remote
		# return random.choice(self.cluster)
	
	def get_available(self, queue):
		remotes_available = [False] * len(self.cluster)
		occupy_num = len(self.cluster)
		for _i, remote in enumerate(self.cluster):
			if not remote.check_on(queue):
				occupy_num -= 1
				continue
			if not remote.occupied:
				remotes_available[_i] = True
				occupy_num -= 1
		return remotes_available, occupy_num
	
	def execute(self, idx, expdir, queue):
		self.choice(queue).execute(idx, expdir, queue)


def run_tasks(config_list, expdir_list):
	controller = ClusterController(config_list)
	result_list = [None for _ in expdir_list]
	
	queue = Queue()
	for idx, expdir in enumerate(expdir_list):
		queue.put([idx, expdir])
	
	remained = len(result_list)
	while remained > 0:
		idx, val = queue.get()
		if isinstance(val, str):
			# expdir, need to execute
			controller.execute(idx, val, queue)
		elif isinstance(val, tuple):
			# result, need to be put in result_list
			result_list[idx] = val
			remained -= 1
	return result_list


def run(task_list):
	with open(config_file, 'r') as f:
		config_list = json.load(f)
	expdir_list = [expdir for expdir, *_ in task_list]
	result_list = run_tasks(config_list, expdir_list)
	for idx, _ in enumerate(task_list):
		task_list[idx].append(result_list[idx])
