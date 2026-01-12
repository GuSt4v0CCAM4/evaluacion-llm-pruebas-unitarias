import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

files_path = os.path.join(BASE_DIR, "files.txt")
seeds_path = os.path.join(BASE_DIR, "seeds.txt")
evosuite_jar = os.path.join(ROOT_DIR, "tools/evosuite/evosuite-1.0.6.jar")
projects_dir = os.path.join(ROOT_DIR, "projetos")

with open(files_path, "r") as dados:
    for line in dados:
        line = line.strip()
        prj, clazz = line.split(":")

        project_path = os.path.join(projects_dir, prj)

        with open(seeds_path, "r") as seeds:
            for seed in seeds:
                seed = seed.strip()

                cmd = (
                    f"cd {project_path} && "
                    f"mvn compile && "
                    f"java -jar {evosuite_jar} "
                    f"-generateSuite "
                    f"-seed {seed} "
                    f"-Dsearch_budget=60 "
                    f"-Dcriterion=branch "
                    f"-Dstopping_condition=MaxTime "
                    f"-target target/classes "
                    f"-class {clazz} "
                    f"-Duse_separate_classloader=false"
                )

                os.system(cmd)


#import sys
#import os

#if len(sys.argv) < 1:
#	print("error: gera-evo.py")
#	print("Example: gera-evo.py")
#	sys.exit(1)

#dados = open('files.txt', 'r')

#for x in dados:
#	x = x.strip()
#	info = x.split(':')
#	prj = info[0]
#	clazz = info[1]

#	seeds = open('seeds.txt', 'r')
	
#	for seed in seeds:
#		className = clazz.split('.')
#		seed = seed.strip()
#		cmd = "cd " + "../projetos/" + prj + ";mvn compile; java -jar ../../tools/evosuite/evosuite-1.0.6.jar -generateSuite -seed " + seed + " -Dsearch_budget=60 -Dcriterion=branch -Dstopping_condition=MaxTime -target target/classes -class "+ clazz + " -Duse_separate_classloader=false; ../../scripts/rename-evo-tc.sh " + className[1] + " " + seed
#		os.system(cmd)
#	seeds.close()

#dados.close()

