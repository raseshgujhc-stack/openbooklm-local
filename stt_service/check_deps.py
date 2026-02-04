import pkg_resources

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

for req in requirements:
    try:
        dist = pkg_resources.get_distribution(req.split('==')[0].split('>=')[0].split('<=')[0])
        print(f"{dist.key}=={dist.version}")
        for dep in dist.requires():
            print(f"  -> {dep}")
            if 'av' in str(dep).lower():
                print(f"     WARNING: Depends on av!")
    except:
        print(f"{req} - not installed")
