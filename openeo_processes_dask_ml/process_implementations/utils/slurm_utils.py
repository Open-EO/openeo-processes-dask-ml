import os.path
import re
import subprocess
import time
from pathlib import Path

from openeo_processes_dask_ml.process_implementations.constants import (
    SLURM_ML_CONFIG_PATH,
)
from openeo_processes_dask_ml.process_implementations.exceptions import SLURMException


def get_config() -> list[str]:
    config = {}
    with open(SLURM_ML_CONFIG_PATH) as configfile:
        for i, line in enumerate(configfile):
            configline = line.split("=")
            if len(configline) != 2:
                raise ValueError(
                    f"SLURM Config file invalid. "
                    f'Only one key-value pair per line is allowed, seperated by "=". '
                    f"Error in line {i+1}."
                )
            key, value = configline
            config[key] = value

    check_config(config)
    config_list = [f"--{k}={config[k]}" for k in config]
    return config_list


def check_config(config: dict):
    required_values = ["time", "cpus-per-task", "mem", "partition"]

    for req_value in required_values:
        if req_value not in config:
            raise ValueError(
                f"Required SLURM config parameter {req_value} not in configfile"
            )


class SLURMJob:
    def __init__(self, input_dir: Path):
        self._input_dir = input_dir
        self._job_id = None

    @property
    def _job_info_file(self) -> Path:
        return self._input_dir / "slurm_job.info"

    @property
    def created(self) -> bool:
        return os.path.exists(self._job_info_file)

    @property
    def job_id(self):
        if self._job_id is not None:
            return self._job_id

        if self.created:
            job_id = self._read_job_file()
            self._job_id = job_id
            return job_id

        raise SLURMException("Slurm Job has not been created yet.")

    def create_job(self, script_lines: list[str]):
        slurm_config = get_config()

        command = ["sbatch"]
        command.extend(slurm_config)

        script = "; ".join(script_lines)
        command.append(f"--wrap={script}")

        s = subprocess.run(command, capture_output=True)

        if s.returncode != 0:
            raise SLURMException(
                f"Something went wrong submitting the job to SLURM.\n"
                f"{s.stdout.decode()}\n"
                f"{s.stderr.decode()}"
            )

        # If submitted successfully, the return string should be:
        # e.g.: "Submitted batch job 10552218"
        slurm_string = s.stdout.decode()
        pattern = r"\d+"
        job_id = re.search(pattern, slurm_string).group()

        self._write_job_file(job_id)
        self._job_id = job_id

    def wait_till_finnished(self, poll_interval: int = 10):
        """
        Wait till slurm job has finnished
        :param poll_interval: Job polling interval in seconds
        :return: Job finnish status: True if successful, False if failed
        """

        while not self.job_finnished:
            time.sleep(poll_interval)

    @property
    def job_finnished(self) -> bool:
        """
        :return: True if job has finnished, False if it is pending or running
        """
        command = ["squeue", f"--job={self.job_id}"]
        s = subprocess.run(command, capture_output=True)

        if s.returncode != 0:
            raise SLURMException(
                f"Something went wrong retrieving SLURM Job status.\n"
                f"{s.stdout.decode()}\n"
                f"{s.stderr.decode()}"
            )

        return not self.job_id in s.stdout.decode()

    def _write_job_file(self, job_id: str):
        with open(self._job_info_file, "w") as file:
            file.write(f"jobid={job_id}")

    def _read_job_file(self) -> str:
        with open(self._job_info_file) as file:
            line = file.readline().strip()
        return line.split("=")[1]
