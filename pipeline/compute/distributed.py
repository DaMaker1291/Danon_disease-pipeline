import os
import time
import logging
import pickle
import hashlib
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    HAS_MPI = True
except (ImportError, RuntimeError):
    HAS_MPI = False
    MPI = None


class MPIController:
    def __init__(self):
        if HAS_MPI and MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_master = self.rank == 0
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_master = True

    def broadcast(self, data: Any) -> Any:
        if self.comm is None:
            return data
        return self.comm.bcast(data, root=0)

    def scatter(self, data: list) -> Any:
        if self.comm is None:
            return data
        chunks = self._chunk_list(data, self.size)
        return self.comm.scatter(chunks, root=0)

    def gather(self, data: Any) -> list:
        if self.comm is None:
            return [data]
        return self.comm.gather(data, root=0)

    def reduce(self, data: Any, op: str = "sum") -> Any:
        if self.comm is None:
            return data
        mpi_op = {
            "sum": MPI.SUM,
            "max": MPI.MAX,
            "min": MPI.MIN,
            "prod": MPI.PROD,
        }.get(op, MPI.SUM)
        return self.comm.reduce(data, op=mpi_op, root=0)

    def allreduce(self, data: Any, op: str = "sum") -> Any:
        if self.comm is None:
            return data
        mpi_op = {
            "sum": MPI.SUM,
            "max": MPI.MAX,
            "min": MPI.MIN,
        }.get(op, MPI.SUM)
        return self.comm.allreduce(data, op=mpi_op)

    def allgather(self, data: Any) -> list:
        if self.comm is None:
            return [data]
        return self.comm.allgather(data)

    def _chunk_list(self, data: list, n: int) -> list:
        chunk_size = (len(data) + n - 1) // n
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    def barrier(self):
        if self.comm is not None:
            self.comm.Barrier()

    def log(self, message: str):
        logger.info("[Rank %d/%d] %s", self.rank, self.size, message)


class DistributedFilter:
    def __init__(self, mpi_controller: MPIController = None):
        self.mpi = mpi_controller or MPIController()

    def parallel_filter(self, filter_fn: Callable, candidates: list, threshold: float) -> list:
        if not self.mpi.is_master:
            return []

        local_candidates = self.mpi.scatter(candidates)

        passed = []
        for batch in (local_candidates if isinstance(local_candidates, list) else [local_candidates]):
            if filter_fn(batch, threshold):
                passed.append(batch)

        all_passed = self.mpi.gather(passed)

        result = [item for sublist in all_passed if sublist for item in sublist]
        logger.info(
            "Distributed filter: %d/%d candidates passed",
            len(result), len(candidates)
        )
        return result

    def map_reduce(
        self,
        map_fn: Callable,
        reduce_fn: Callable,
        data: list,
        initial_value: Any = None,
    ) -> Any:
        local_data = self.mpi.scatter(data)
        local_results = [map_fn(item) for item in local_data]
        local_reduced = reduce_fn(local_results)
        global_result = self.mpi.allreduce(local_reduced)
        return global_result


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, stage: str, data: Any, metadata: dict = None):
        timestamp = int(time.time())
        filename = f"{stage}_{timestamp}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "stage": stage,
            "timestamp": timestamp,
            "data": data,
            "metadata": metadata or {},
        }

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)

        checksum = self._compute_checksum(filepath)
        meta_path = filepath + ".meta"
        with open(meta_path, "w") as f:
            import json
            json.dump({
                "stage": stage,
                "timestamp": timestamp,
                "checksum": checksum,
                "size_bytes": os.path.getsize(filepath),
            }, f)

        logger.info("Checkpoint saved: %s (%d bytes)", filepath, os.path.getsize(filepath))
        return filepath

    def load_checkpoint(self, filepath: str) -> dict:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def find_latest_checkpoint(self, stage: str) -> str:
        import glob
        pattern = os.path.join(self.checkpoint_dir, f"{stage}_*.pkl")
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def _compute_checksum(self, filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
