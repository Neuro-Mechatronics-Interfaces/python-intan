"""Lightweight package exports for intan.applications.

Make imports lazy/guarded so importing the package doesn't pull heavy
dependencies (tensorflow, scipy, pandas) during CLI or test runs.
"""
try:
	from ._emg_viewer import EMGViewer
except Exception:
	EMGViewer = None

try:
	from ._launcher import launch_emg_viewer, launch_emg_trial_selector
except Exception:
	def launch_emg_viewer(*a, **k):
		raise RuntimeError('launch_emg_viewer is unavailable (import failed)')
	def launch_emg_trial_selector(*a, **k):
		raise RuntimeError('launch_emg_trial_selector is unavailable (import failed)')

try:
	from ._emg_trial_selector import EMGTrialSelector
except Exception:
	EMGTrialSelector = None