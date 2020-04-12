from src.rubiks.utils import seedsetter
from shutil import rmtree

class MainTest:
	@classmethod
	def setup_test(cls):
		seedsetter()

	@classmethod
	def teardown_test(cls):
		rmtree('local_tests', onerror=self.ignore_absentee)

	@staticmethod
	def ignore_absentee(func, path, exc_inf):
		except_instance = exc_inf[1]
		if isinstance(except_instance, FileNotFoundError): return
		raise except_instance
