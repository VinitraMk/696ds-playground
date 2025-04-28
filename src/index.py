import subprocess

scripts = ["src/query_set_generation/query_generator.py", "src/query_set_generation/answer_generator.py", "src/query_set_generation/groundings_generator.py", "src/query_set_generation/reasonings_generator.py"]

for script in scripts:
    subprocess.run(["python", script], env={"PYTHONPATH": "/work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/playground"})