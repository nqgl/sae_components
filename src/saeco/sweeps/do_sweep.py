def do_sweep(purge_after=True, in_cmd=None):
    from saeco.sweeps import Sweeper
    import sys

    swfpath = sys.argv[0]
    print("args:", sys.argv)
    print(swfpath)
    thispath = "/".join(swfpath.split("/")[:-1])
    filename = swfpath.split("/")[-1]
    thispath = thispath.split("/sae_components/")[-1]
    # sw.start_agent()
    options = ["run", "rand", "sweep", "yes"]
    sweep_or_run = -1
    while sweep_or_run not in options:
        sweep_or_run = in_cmd or input(f"sweep? ({', '.join(options)}):")

    sw = Sweeper(thispath, module_name=filename)
    if sweep_or_run == "rand":
        sw.rand_run_no_agent()
        return
    sw.initialize_sweep()
    if sweep_or_run == "run":
        sw.start_agent()
    elif sweep_or_run in ("sweep", "yes"):
        n = input("create instances?")
        try:
            n = int(n)
        except ValueError:
            n = False
        from ezpod import Pods, RunProject, RunFolder

        pods = Pods.All()
        if n:
            pods.make_new_pods(n)

        pods.sync()
        pods.setup()
        print("running!")
        pods.runpy(
            f"src/saeco/sweeps/sweeper.py {thispath} --module-name {filename}",
            purge_after=purge_after,
            challenge_file="src/saeco/sweeps/challenge.py",
        )
    else:
        raise ValueError(f"Invalid input: {sweep_or_run}")
