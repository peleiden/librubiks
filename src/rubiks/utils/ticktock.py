from datetime import datetime
from time import perf_counter

def get_timestamp(for_file=False):
	# Returns a timestamp
	# File name friendly format if for_file
	if for_file:
		return "-".join(str(datetime.now()).split(".")[0].split(":")).replace(" ", "_")
	else:
		return str(datetime.now())
	

class TickTock:

	_start: float
	_sections = {}
	_section_depth = 0
	_units = {"ms": 1000, "s": 1, "m": 1/60}

	def tick(self):
		self._start = perf_counter()
		return self._start
	
	def tock(self, with_print = False):
		end = perf_counter()
		passed_time = end - self._start
		return passed_time

	def section(self, name: str):
		if name not in self._sections:
			self._sections[name] = {"tt": TickTock(), "elapsed": 0, "n": 0, "depth": self._section_depth}
		self._section_depth += 1
		self._sections[name]["n"] += 1
		self._sections[name]["tt"].tick()
	
	def end_section(self, name: str):
		dt = self._sections[name]["tt"].tock()
		self._sections[name]["elapsed"] += dt
		self._section_depth -= 1
	
	@staticmethod
	def thousand_seps(numstr: str or float or int) -> str:
		decs = str(numstr)
		rest = ""
		if "." in decs:
			rest = decs[decs.index("."):]
			decs = decs[:decs.index(".")]
		for i in range((len(decs)-1)//3):
			idx = len(decs) - i * 3 - 3
			decs = decs[:idx] + "," + decs[idx:]  # FIXME
		return decs + rest
	
	@classmethod
	def stringify_time(cls, dt: float, unit="ms"):
		str_ = f"{dt*cls._units[unit]:.3f} {unit}"
		return cls.thousand_seps(str_)

	def get_sections(self):
		# Returns data parts of sections
		return {kw: v for kw, v in self._sections.items() if kw != "tt"}

	def get_section_times(self):
		return {kw: v["elapsed"] for kw, v in self._sections.items()}
	
	def stringify_sections(self, unit="s"):
		# Returns pretty sections
		strs = [["Execution times", "Total time", "Hits", "Avg. time"]]
		for kw, v in self._sections.items():
			strs.append([
				"- " * v["depth"] + kw,
				self.stringify_time(v["elapsed"], unit),
				self.thousand_seps(v["n"]),
				self.stringify_time(v["elapsed"] / v["n"], "ms")
			])
		for i in range(len(strs[0])):
			length = max(len(strs[j][i]) for j in range(len(strs)))
			for j in range(len(strs)):
				if i == 0:
					strs[j][i] += " " * (length - len(strs[j][i]))
				else:
					strs[j][i] = " " * (length - len(strs[j][i])) + strs[j][i]
		for i in range(len(strs)):
			strs[i] = " | ".join(strs[i])
		return "\n".join(strs)
	
	def __str__(self):
		return self.stringify_sections("s")



