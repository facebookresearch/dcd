# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import datetime
import json
import logging
import os
import time
from typing import Dict

import numpy as np


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None
    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(
        self,
        xpid: str = None,
        xp_args: dict = None,
        rootdir: str = "~/logs",
        symlink_to_latest: bool = True,
        seeds=None,
    ):
        if not xpid:
            # Make unique id.
            xpid = "{proc}_{unixtime}".format(
                proc=os.getpid(), unixtime=int(time.time())
            )
        self.xpid = xpid
        self._tick = 0

        # Metadata gathering.
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # We need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other unwanted side-effects).
        self.metadata["args"] = copy.deepcopy(xp_args)
        self.metadata["xpid"] = self.xpid

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")

        train_full_distribution = xp_args.get('train_full_distribution', False)
        seed_buffer_size = xp_args.get('level_replay_seed_buffer_size', 0)
        self.record_seed_diffs = \
            train_full_distribution and seed_buffer_size > 0

        self.seeds = None
        if not self.record_seed_diffs and seeds:
            self.seeds = [str(seed) for seed in seeds]

        # To stdout handler.
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # To file handler.
        self.basepath = os.path.join(rootdir, self.xpid)
        if not os.path.exists(self.basepath):
            self._logger.info("Creating log directory: %s", self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info("Found log directory: %s", self.basepath)

        if symlink_to_latest:
            # Add 'latest' as symlink unless it exists and is no symlink.
            symlink = os.path.join(rootdir, "latest")
            try:
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(self.basepath, symlink)
                    self._logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                # os.remove() or os.symlink() raced. Don't do anything.
                pass

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            logs="{base}/logs.csv".format(base=self.basepath),
            fields="{base}/fields.csv".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
            level_weights="{base}/level_weights.csv".format(base=self.basepath),
            level_seeds="{base}/level_seeds.csv".format(base=self.basepath),
            final_test_eval="{base}/final_test_eval.csv".format(base=self.basepath)
        )

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning(
                "Path to meta file already exists. " "Not overriding meta."
            )
        else:
            self._save_metadata()

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]):
            self._logger.warning(
                "Path to message file already exists. " "New data will be appended."
            )

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info("Saving logs data to %s", self.paths["logs"])
        self._logger.info("Saving logs' fields to %s", self.paths["fields"])
        self.fieldnames = ["_tick", "_time"]
        self.final_test_eval_fieldnames = ['num_test_seeds', 'mean_episode_return', 'median_episode_return']
        self.level_seeds_fieldnames = ['new_seeds', 'new_seed_indices']
        if os.path.exists(self.paths["logs"]):
            self._logger.warning(
                "Path to log file already exists. " "New data will be appended."
            )
            # Override default fieldnames.
            with open(self.paths["fields"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if len(lines) > 0:
                    self.fieldnames = lines[-1]
            # Override default tick: use the last tick from the logs file plus 1.
            with open(self.paths["logs"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                # Need at least two lines in order to read the last tick:
                # the first is the csv header and the second is the first line
                # of data.
                if len(lines) > 1:
                    self._tick = int(lines[-1][0]) + 1

        self._fieldfile = open(self.paths["fields"], "a")
        self._fieldwriter = csv.writer(self._fieldfile)
        self._logfile = open(self.paths["logs"], "a")
        self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)
        self._levelweightsfile = open(self.paths["level_weights"], "a")
        self._levelweightswriter = csv.writer(self._levelweightsfile)
        self._levelseedsfile = open(self.paths["level_seeds"], "a")
        self._levelseedswriter = csv.DictWriter(self._levelseedsfile, fieldnames=self.level_seeds_fieldnames)
        self._finaltestfile = open(self.paths["final_test_eval"], "a")
        self._finaltestwriter = csv.DictWriter(self._finaltestfile, fieldnames=self.final_test_eval_fieldnames)

        if self.seeds and not self.record_seed_diffs:
            self._levelweightsfile.write("# %s\n" % ",".join(self.seeds))
            self._levelweightsfile.flush()

        self._finaltestwriter.writeheader()
        self._finaltestfile.flush()

    def log(self, to_log: Dict, tick: int = None, verbose: bool = False) -> None:
        if tick is not None:
            raise NotImplementedError
        else:
            to_log["_tick"] = self._tick
            self._tick += 1
        to_log["_time"] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            self._fieldwriter.writerow(self.fieldnames)
            self._logger.info("Updated log fields: %s", self.fieldnames)

        if to_log["_tick"] == 0:
            self._logfile.write("# %s\n" % ",".join(self.fieldnames))

        if verbose:
            self._logger.info(
                "LOG | %s",
                ", ".join(["{}: {}".format(k, to_log[k]) for k in sorted(to_log)]),
            )

        self._logwriter.writerow(to_log)
        self._logfile.flush()

    def log_level_weights(self, weights, seeds=None):
        if self.record_seed_diffs:
            if self.seeds is None:
                self.seeds = seeds.copy()
                level_seed_log = {
                    'new_seeds': " ".join([str(s) for s in self.seeds]),
                    'new_seed_indices': " ".join([str(i) for i in range(len(self.seeds))]),
                }
            else:
                new_seed_indices = np.nonzero(self.seeds - seeds)[0]
                new_seeds = seeds[new_seed_indices]
                self.seeds = seeds.copy()
                level_seed_log = {
                    'new_seeds': " ".join([str(s) for s in new_seeds]),
                    'new_seed_indices': " ".join([str(i) for i in new_seed_indices]),
                }
            self._levelseedswriter.writerow(level_seed_log)
            self._levelseedsfile.flush()

        self._levelweightswriter.writerow(weights)
        self._levelweightsfile.flush()

    def log_final_test_eval(self, to_log):
        self._finaltestwriter.writerow(to_log)
        self._finaltestfile.flush()

    def close(self, successful: bool = True) -> None:
        self.metadata["date_end"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        self.metadata["successful"] = successful
        self._save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()

    def _save_metadata(self) -> None:
        with open(self.paths["meta"], "w") as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)

    def latest_tick(self):
        with open(self.paths["logs"], "r") as logsfile:
            csvreader = csv.reader(logsfile)
            for row in csvreader:
                pass
            if row:
                return int(row[0])
            else:
                return 0
