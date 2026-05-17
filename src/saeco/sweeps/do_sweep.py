def do_sweep(purge_after=True, in_cmd=None):
    """Entry point for a sweep script: prompt for and dispatch a run mode.

    Call this at the bottom of a script that builds a (possibly swept)
    ``RunConfig``. It asks (or takes ``in_cmd``) what to do:

      - ``rand``  — run one randomly-sampled config locally
      - ``run``   — start a sweep agent (pull configs and run them)
      - ``sweep`` / ``yes`` — initialize the sweep and optionally spin up
        remote pods (requires the ``remote`` extra) to run the grid
      - ``new …`` — initialize a fresh sweep first, then do the above
    """
    import sys

    from saeco.sweeps.sweeper import Sweeper

    swfpath = sys.argv[0]
    print("args:", sys.argv)
    print(swfpath)
    thispath = "/".join(swfpath.split("/")[:-1])
    filename = swfpath.split("/")[-1]
    thispath = thispath.split("/sae_components/")[-1]
    options = ["run", "rand", "sweep", "yes"]
    sweep_or_run = -1
    sw = Sweeper(thispath, module_name=filename)
    while sweep_or_run not in options:
        sweep_or_run = in_cmd or input(f"sweep? ({', '.join(options)}):")
        if sweep_or_run[:3] == "new":
            print("initializing new sweep")
            sw.initialize_sweep()
            sweep_or_run = " ".join(sweep_or_run.split(" ")[1:])
    if sweep_or_run == "rand":
        sw.rand_run_no_agent()
        return
    if sweep_or_run == "run":
        sw.start_agent()
    elif sweep_or_run in ("sweep", "yes"):
        sw.initialize_sweep()
        n = input("create instances?")
        try:
            n = int(n)
        except ValueError:
            n = False
        from ezpod import Pods

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
