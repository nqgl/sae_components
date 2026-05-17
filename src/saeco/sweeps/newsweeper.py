import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from attrs import define, field

from saeco.architecture import ArchitectureBase
from saeco.architecture.arch_reload_info import ArchRef
from saeco.mlog import mlog
from saeco.sweeps.sweep_data import SweepData
from sweepable import SweepableConfig

from .sweep_runner import SweepRunner

if TYPE_CHECKING:
    from ezpod import Pods


T = TypeVar("T", default=SweepableConfig)


@define
class SweepManager:
    arch: ArchitectureBase
    sweep_data: SweepData | None = field(default=None)
    sweep_data_path: Path | None = field(default=None)
    ezpod_group: str | None = field(default=None)

    @property
    def cfg(self):
        return self.arch.run_cfg

    def initialize_sweep(
        self, project=None, custom_sweep=True, run_type_str: str = "sweep"
    ):
        assert self.sweep_data is None
        assert self.sweep_data_path is None
        if self.cfg.is_concrete() and not custom_sweep:
            raise ValueError(
                "tried to initialize sweep on a config with no swept fields"
            )
        if custom_sweep:
            sweep_id = run_type_str
            # import datetime

            # sweep_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            # if run_type_str:
            #     sweep_id = f"{run_type_str}_{sweep_id}"
        else:
            assert not run_type_str
            sweep_id = mlog.create_sweep(self.cfg.sweep_info_tree, project=project)
        self.sweep_data = SweepData.from_arch_and_id(
            self.arch, sweep_id, project=project
        )
        self.sweep_data_path = self.sweep_data.save(None)

    def rand_run_no_agent(self, project: str | None = None):
        if not self.sweep_data:
            self.initialize_sweep(
                custom_sweep=True, run_type_str="rand", project=project
            )
        sweeprunner = SweepRunner(self.sweep_data)
        return sweeprunner.run_random_instance()

    def run_local_inst_by_index(self, index: int):
        if not self.sweep_data:
            self.initialize_sweep(custom_sweep=True, run_type_str="indexed_single")

        cfg: SweepableConfig = self.sweep_data.root_arch_ref.config
        cfg_nodes = cfg.sweep_info_tree
        cfg_i = cfg_nodes.select_instance_by_index(index)
        inst_cfg = cfg.from_selective_sweep(cfg_i)
        cfg_hash = inst_cfg.get_hash()
        sweeprunner = SweepRunner(
            self.sweep_data, sweep_index=index, sweep_hash=cfg_hash
        )
        return sweeprunner.start_sweep_agent()  # is this right?

    def local_sweep(self):
        raise AssertionError("method needs update on sweeprunner init")
        arch_ref = ArchRef.from_arch(self.arch)
        sweeprunner = SweepRunner(arch_ref, self.sweep_data.sweep_id)
        return sweeprunner.start_sweep_agent()

    def local_custom_sweep(self):
        # root = self.sweep_data.root_arch_ref.config
        root = self.arch.run_cfg
        root_swept = root.sweep_info_tree
        n = root_swept.swept_combinations_count_including_vars()
        for i in range(n):
            cfg = root.from_selective_sweep(root_swept.select_instance_by_index(i))
            cfg_hash = cfg.get_hash()
            runner = SweepRunner.from_sweepdata(
                self.sweep_data, sweep_index=i, sweep_hash=cfg_hash
            )
            runner.start_sweep_agent()

    def load_architecture(self, path: Path): ...

    def get_worker_run_command(self, extra_args: str = ""):
        path = (
            self.sweep_data_path
            if not self.sweep_data_path.is_absolute()
            else self.sweep_data_path.relative_to(Path.cwd())
        )
        return f"-m saeco.sweeps.sweeprunner_cli {path} {extra_args}"

    def get_worker_run_commands_for_manual_sweep(self, suffix: str = ""):
        root = self.arch.run_cfg
        root_swept = root.sweep_info_tree
        variants = []
        for i in range(root_swept.swept_combinations_count_including_vars()):
            cfg = root.from_selective_sweep(root_swept.select_instance_by_index(i))
            variants.append((i, cfg.get_hash()))
        return [
            self.get_worker_run_command(f"--sweep-index {i} --sweep-hash {h} {suffix}")
            for i, h in variants
        ]

    def get_worker_run_commands_for_manual_random_sweep(self, n: int, suffix: str = ""):
        root = self.arch.run_cfg
        root_swept = root.sweep_info_tree
        combos = root_swept.swept_combinations_count_including_vars()
        assert n <= combos
        import random

        sweep_ids = random.sample(range(combos), n)
        variants = []
        for i in sweep_ids:
            cfg = root.from_selective_sweep(root_swept.select_instance_by_index(i))
            variants.append((i, cfg.get_hash()))
        return [
            self.get_worker_run_command(f"--sweep-index {i} --sweep-hash {h} {suffix}")
            for i, h in variants
        ]

    def run_sweep_on_pods(
        self,
        new_pods=None,
        purge_after=True,
        challenge_file="src/saeco/sweeps/challenge.py",
        keep_after=False,
    ):
        with self.created_pods(new_pods, keep=keep_after) as pods:
            pods.runpy(
                self.get_worker_run_command(),
                purge_after=purge_after,
                challenge_file=challenge_file,
            )

    def run_sweep_on_pods_with_monitoring(
        self,
        new_pods=None,
        purge_after=True,
        challenge_file=None,
        # challenge_file="src/saeco/sweeps/challenge.py",
        keep_after=False,
        setup_min=None,
    ):
        with self.created_pods(new_pods, keep=keep_after, setup_min=setup_min) as pods:
            print("running on remotes")
            pods.runpy_with_monitor(
                self.get_worker_run_command(),
                purge_after=purge_after,
                challenge_file=challenge_file,
            )

    def run_manual_sweep_with_monitoring(
        self,
        total_pods=None,
        new_pods=None,
        purge_after=True,
        keep_after=False,
        setup_min=None,
        prefix_vars=None,
    ):
        with self.created_pods(
            total_pods, create_n=new_pods, keep=keep_after, setup_min=setup_min
        ) as pods:
            print("running on remotes")
            pods.runpy_with_monitor(
                self.get_worker_run_commands_for_manual_sweep(),
                purge_after=purge_after,
                challenge_file=None,
                prefix_vars=prefix_vars,
            )

    @contextmanager
    def created_pods(
        self,
        num_pods=None,
        create_n=None,
        keep=False,
        setup_min=None,
        skip_setup=False,
    ) -> Generator["Pods"]:
        from ezpod import Pods

        if num_pods is None:
            num_pods = int(input("Enter number of pods: "))

        if create_n is None:
            create_n = num_pods
            do_purge = False
        else:
            do_purge = True

        pods = Pods.All(group=self.ezpod_group).get_alive()
        pods.make_new_pods(create_n)
        pods.EZPOD_MIN_COMPLETE_TO_CONTINUE = setup_min
        pods.sync()
        if not skip_setup:
            pods.setup()
        if do_purge:
            pods.prune(
                n=num_pods,
                challenge_file="-c 'print(2)'",
                stop_after_n_complete=num_pods,
            )
            pods = Pods.All(group=self.ezpod_group).get_alive()

        try:
            yield pods
        finally:
            if not keep:
                try:
                    pods.purge()
                except Exception as e:
                    try:
                        print("pod purge failed, retrying")
                        time.sleep(10)
                        Pods.All(group=self.ezpod_group).purge()
                    except Exception as e:
                        print(
                            "failed twice to purge subset of pods, purging all pods in "
                            "10 seconds"
                        )
                        time.sleep(10)
                        Pods.All().purge()
                        raise e
                    raise e

    @contextmanager
    def existing_pods(
        self,
        keep=True,
    ) -> Generator["Pods"]:
        from ezpod import Pods

        pods = Pods.All(group=self.ezpod_group)
        pods.sync()
        try:
            yield pods
        finally:
            if not keep:
                try:
                    pods.purge()
                except Exception as e:
                    try:
                        print("pod purge failed, retrying")
                        time.sleep(10)
                        Pods.All(group=self.ezpod_group).purge()
                    except Exception as e:
                        print(
                            "failed twice to purge subset of pods, purging all pods in "
                            "10 seconds"
                        )
                        time.sleep(10)
                        Pods.All().purge()
                        raise e
                    raise e
