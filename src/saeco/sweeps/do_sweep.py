def do_sweep(purge_after=True):
    from saeco.sweeps import Sweeper
    import sys

    swfpath = sys.argv[0]
    print(swfpath)
    thispath = "/".join(swfpath.split("/")[:-1])
    filename = swfpath.split("/")[-1]
    thispath = thispath.split("/sae_components/")[-1]
    sw = Sweeper(thispath, module_name=filename)
    # sw.start_agent()
    sw.initialize_sweep()
    sweep_or_run = input("sweep?")
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
        )
    else:
        raise ValueError(f"Invalid input: {sweep_or_run}")
