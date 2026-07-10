import os
import json
import time
import logging
import subprocess
import tempfile
from typing import Any, Callable
from abc import ABC, abstractmethod
from pipeline.config import ComputeConfig

logger = logging.getLogger(__name__)


class ComputeBackend(ABC):
    @abstractmethod
    def submit_job(self, job_spec: dict) -> str:
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> dict:
        pass

    @abstractmethod
    def wait_for_job(self, job_id: str, timeout: float = None) -> dict:
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        pass

    @abstractmethod
    def get_available_resources(self) -> dict:
        pass


class SlurmBackend(ComputeBackend):
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.account = config.slurm_account
        self.partition = config.slurm_partition

    def submit_job(self, job_spec: dict) -> str:
        script = self._generate_slurm_script(job_spec)
        script_path = tempfile.mktemp(suffix=".slurm", dir="/tmp")
        with open(script_path, "w") as f:
            f.write(script)

        cmd = ["sbatch"]
        if self.account:
            cmd.extend(["--account", self.account])
        cmd.append(script_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Slurm submit failed: {result.stderr}")

        job_id = result.stdout.strip().split()[-1]
        logger.info("Slurm job %s submitted", job_id)
        return job_id

    def _generate_slurm_script(self, job_spec: dict) -> str:
        script = f"""#!/bin/bash
#SBATCH --job-name={job_spec.get('name', 'longevity-screen')}
#SBATCH --partition={self.partition}
#SBATCH --nodes={job_spec.get('nodes', 1)}
#SBATCH --ntasks-per-node={job_spec.get('tasks_per_node', 4)}
#SBATCH --cpus-per-task={self.config.cpus_per_worker}
#SBATCH --gres=gpu:{self.config.gpus_per_worker}
#SBATCH --mem={self.config.memory_per_worker_gb}G
#SBATCH --time={job_spec.get('time', '24:00:00')}
#SBATCH --output={job_spec.get('output', '/tmp/longevity_%j.out')}
#SBATCH --error={job_spec.get('error', '/tmp/longevity_%j.err')}

module load cuda/12.1 python/3.11
source /opt/longevity/bin/activate

{job_spec.get('command', 'python -m pipeline.main')}
"""
        return script

    def get_job_status(self, job_id: str) -> dict:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State,Elapsed,MaxRSS", "--noheader"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = lines[0].split()
            return {
                "state": parts[0] if parts else "UNKNOWN",
                "elapsed": parts[1] if len(parts) > 1 else "0:00",
                "max_rss": parts[2] if len(parts) > 2 else "0",
            }
        return {"state": "UNKNOWN"}

    def wait_for_job(self, job_id: str, timeout: float = None) -> dict:
        start = time.time()
        while True:
            status = self.get_job_status(job_id)
            state = status.get("state", "UNKNOWN")
            if state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                return status
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
            time.sleep(30)

    def cancel_job(self, job_id: str) -> bool:
        result = subprocess.run(["scancel", job_id], capture_output=True)
        return result.returncode == 0

    def get_available_resources(self) -> dict:
        result = subprocess.run(
            ["sinfo", "-p", self.partition, "--format=CPUs,GPU,Memory,State", "--noheader"],
            capture_output=True, text=True
        )
        nodes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split()
                nodes.append({
                    "cpus": int(parts[0]) if parts[0] else 0,
                    "gpus": int(parts[1]) if parts[1] else 0,
                    "memory_gb": int(parts[2].rstrip("G")) if parts[2] else 0,
                    "state": parts[3] if len(parts) > 3 else "unknown",
                })
        return {"partition": self.partition, "nodes": nodes}


class KubernetesBackend(ComputeBackend):
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.namespace = config.k8s_namespace
        self.image = config.k8s_image

    def submit_job(self, job_spec: dict) -> str:
        job_name = job_spec.get("name", f"longevity-{int(time.time())}")
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": self.namespace},
            "spec": {
                "parallelism": job_spec.get("parallelism", 1),
                "completions": job_spec.get("completions", 1),
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "longevity-worker",
                            "image": self.image,
                            "command": job_spec.get("command", ["python", "-m", "pipeline.main"]).split(),
                            "resources": {
                                "requests": {
                                    "cpu": str(self.config.cpus_per_worker),
                                    "memory": f"{self.config.memory_per_worker_gb}Gi",
                                    "nvidia.com/gpu": str(self.config.gpus_per_worker),
                                },
                                "limits": {
                                    "cpu": str(self.config.cpus_per_worker),
                                    "memory": f"{self.config.memory_per_worker_gb}Gi",
                                    "nvidia.com/gpu": str(self.config.gpus_per_worker),
                                },
                            },
                        }],
                        "restartPolicy": "Never",
                    },
                },
                "backoffLimit": 3,
            },
        }

        manifest_path = tempfile.mktemp(suffix=".yaml", dir="/tmp")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        result = subprocess.run(
            ["kubectl", "apply", "-f", manifest_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"K8s submit failed: {result.stderr}")
        return job_name

    def get_job_status(self, job_id: str) -> dict:
        result = subprocess.run(
            ["kubectl", "get", "job", job_id, "-n", self.namespace, "-o", "json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"state": "UNKNOWN"}
        job = json.loads(result.stdout)
        conditions = job.get("status", {}).get("conditions", [])
        if conditions:
            latest = conditions[-1]
            return {"state": latest.get("type", "UNKNOWN"), "message": latest.get("message", "")}
        return {"state": "PENDING"}

    def wait_for_job(self, job_id: str, timeout: float = None) -> dict:
        start = time.time()
        while True:
            status = self.get_job_status(job_id)
            if status.get("state") in ["Complete", "Failed"]:
                return status
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"K8s job {job_id} timed out")
            time.sleep(15)

    def cancel_job(self, job_id: str) -> bool:
        result = subprocess.run(
            ["kubectl", "delete", "job", job_id, "-n", self.namespace],
            capture_output=True
        )
        return result.returncode == 0

    def get_available_resources(self) -> dict:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"nodes": []}
        nodes_data = json.loads(result.stdout)
        nodes = []
        for node in nodes_data.get("items", []):
            status = node.get("status", {})
            capacity = status.get("capacity", {})
            nodes.append({
                "name": node["metadata"]["name"],
                "cpus": int(capacity.get("cpu", 0)),
                "memory_gb": int(capacity.get("memory", "0Gi".rstrip("Gi"))) if "Gi" in capacity.get("memory", "") else 0,
                "gpus": int(capacity.get("nvidia.com/gpu", 0)),
            })
        return {"nodes": nodes}


class AWSBatchBackend(ComputeBackend):
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.region = config.aws_region
        self.job_queue = config.aws_job_queue
        self.job_definition = config.aws_job_definition

    def submit_job(self, job_spec: dict) -> str:
        job_name = job_spec.get("name", f"longevity-{int(time.time())}")
        cmd = [
            "aws", "batch", "submit-job",
            "--job-name", job_name,
            "--job-queue", self.job_queue,
            "--job-definition", self.job_definition,
            "--region", self.region,
        ]

        container_overrides = {
            "vcpus": self.config.cpus_per_worker,
            "memory": self.config.memory_per_worker_gb * 1024,
        }
        if self.config.gpus_per_worker > 0:
            container_overrides["resourceRequirements"] = [{
                "type": "GPU",
                "value": str(self.config.gpus_per_worker),
            }]

        if "command" in job_spec:
            container_overrides["command"] = job_spec["command"].split()

        cmd.extend(["--container-overrides", json.dumps(container_overrides)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"AWS Batch submit failed: {result.stderr}")

        response = json.loads(result.stdout)
        job_id = response["jobId"]
        logger.info("AWS Batch job %s submitted", job_id)
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        result = subprocess.run(
            ["aws", "batch", "describe-jobs", "--jobs", job_id, "--region", self.region],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"state": "UNKNOWN"}
        jobs = json.loads(result.stdout).get("jobs", [])
        if jobs:
            job = jobs[0]
            return {
                "state": job.get("status", "UNKNOWN"),
                "started_at": job.get("startedAt"),
                "stopped_at": job.get("stoppedAt"),
            }
        return {"state": "UNKNOWN"}

    def wait_for_job(self, job_id: str, timeout: float = None) -> dict:
        start = time.time()
        while True:
            status = self.get_job_status(job_id)
            if status.get("state") in ["SUCCEEDED", "FAILED"]:
                return status
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"AWS Batch job {job_id} timed out")
            time.sleep(30)

    def cancel_job(self, job_id: str) -> bool:
        result = subprocess.run(
            ["aws", "batch", "cancel-job", "--job-id", job_id, "--region", self.region],
            capture_output=True
        )
        return result.returncode == 0

    def get_available_resources(self) -> dict:
        result = subprocess.run(
            ["aws", "batch", "describe-job-queues", "--job-queues", self.job_queue, "--region", self.region],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {"compute_environments": []}
        queues = json.loads(result.stdout).get("jobQueues", [])
        return {"queues": queues}


class RayBackend(ComputeBackend):
    def __init__(self, config: ComputeConfig):
        self.config = config
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        try:
            import ray
            if not ray.is_initialized():
                ray.init(
                    address="auto",
                    num_cpus=self.config.num_workers * self.config.cpus_per_worker,
                    num_gpus=self.config.num_workers * self.config.gpus_per_worker,
                    ignore_reinit_error=True,
                )
            self._initialized = True
            logger.info("Ray cluster initialized: %s", ray.cluster_resources())
        except Exception as e:
            logger.warning("Ray initialization failed: %s. Using local mode.", e)
            import ray
            ray.init(ignore_reinit_error=True)
            self._initialized = True

    def submit_job(self, job_spec: dict) -> str:
        self.initialize()
        import ray
        func = job_spec.get("function")
        args = job_spec.get("args", ())
        kwargs = job_spec.get("kwargs", {})
        remote = ray.remote(func)
        ref = remote.remote(*args, **kwargs)
        return str(ref)

    def get_job_status(self, job_id: str) -> dict:
        import ray
        try:
            ref = ray.ObjectRef(bytes.fromhex(job_id))
            ready, _ = ray.wait([ref], timeout=0)
            if ready:
                return {"state": "COMPLETED"}
            return {"state": "RUNNING"}
        except Exception:
            return {"state": "UNKNOWN"}

    def wait_for_job(self, job_id: str, timeout: float = None) -> dict:
        import ray
        ref = ray.ObjectRef(bytes.fromhex(job_id))
        result = ray.get(ref, timeout=timeout)
        return {"state": "COMPLETED", "result": result}

    def cancel_job(self, job_id: str) -> bool:
        return True

    def get_available_resources(self) -> dict:
        import ray
        if self._initialized:
            return ray.cluster_resources()
        return {"cpus": os.cpu_count(), "gpus": 0}

    def map_parallel(self, func: Callable, items: list, **kwargs) -> list:
        self.initialize()
        import ray
        batch_size = kwargs.get("batch_size", self.config.batch_size)
        num_cpus = kwargs.get("num_cpus", 1)
        num_gpus = kwargs.get("num_gpus", 0)

        @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
        def process_batch(batch):
            return [func(item) for item in batch]

        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        futures = [process_batch.remote(batch) for batch in batches]
        results = ray.get(futures)
        return [item for batch in results for item in batch]


class SupercomputerInterface:
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.backend = self._create_backend()

    def _create_backend(self) -> ComputeBackend:
        if self.config.backend == "slurm":
            return SlurmBackend(self.config)
        elif self.config.backend == "kubernetes":
            return KubernetesBackend(self.config)
        elif self.config.backend == "aws_batch":
            return AWSBatchBackend(self.config)
        elif self.config.backend == "ray":
            return RayBackend(self.config)
        else:
            logger.warning("Unknown backend '%s', falling back to Ray", self.config.backend)
            return RayBackend(self.config)

    def submit_screening_job(self, pipeline_stage: str, input_data: dict) -> str:
        job_spec = {
            "name": f"longevity-{pipeline_stage}-{int(time.time())}",
            "command": f"python -m pipeline.screening.{pipeline_stage} --config /tmp/config.json",
            "nodes": max(1, self.config.num_workers // 8),
            "tasks_per_node": 8,
            "time": "48:00:00",
        }
        return self.backend.submit_job(job_spec)

    def wait_and_collect(self, job_id: str, timeout: float = 86400) -> dict:
        return self.backend.wait_for_job(job_id, timeout=timeout)

    def submit_parallel_screening(
        self, stage: str, candidates: list, filter_fn: Callable
    ) -> list:
        if isinstance(self.backend, RayBackend):
            return self.backend.map_parallel(
                filter_fn, candidates,
                batch_size=self.config.batch_size,
                num_cpus=self.config.cpus_per_worker,
                num_gpus=self.config.gpus_per_worker,
            )
        else:
            results = []
            for i in range(0, len(candidates), self.config.batch_size):
                batch = candidates[i:i + self.config.batch_size]
                job_spec = {
                    "name": f"{stage}-batch-{i}",
                    "command": f"python -c 'import json,sys; from pipeline.screening.{stage} import filter; print(json.dumps(filter(json.loads(sys.stdin.read()))))'",
                }
                job_id = self.backend.submit_job(job_spec)
                status = self.backend.wait_for_job(job_id)
                if status.get("state") == "COMPLETED":
                    results.extend(status.get("result", []))
            return results

    def get_cluster_status(self) -> dict:
        return self.backend.get_available_resources()
