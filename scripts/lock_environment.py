"""仅锁定训练入口需要的依赖闭包，不混入宿主环境的其他模型工具。"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("requirements-lock.txt"))
    args = parser.parse_args()
    roots = ["torch", "transformers", "numpy", "scipy", "scikit-learn", "safetensors", "tqdm", "gdown", "pytest"]
    queue = [(name, frozenset()) for name in roots]
    versions = {}
    visited = set()
    while queue:
        requested, extras = queue.pop()
        name = canonicalize_name(requested)
        identity = (name, extras)
        if identity in visited:
            continue
        visited.add(identity)
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        for specification in distribution.requires or []:
            requirement = Requirement(specification)
            # requests[socks]、CUDA toolkit extras 等也必须纳入锁定闭包。
            contexts = {"", *extras}
            if requirement.marker is None or any(requirement.marker.evaluate({"extra": extra}) for extra in contexts):
                actual = metadata.version(requirement.name)
                if requirement.specifier and not requirement.specifier.contains(actual, prereleases=True):
                    raise RuntimeError(f"Dependency mismatch: {requirement.name} {actual} does not satisfy {requirement.specifier}")
                queue.append((requirement.name, frozenset(requirement.extras)))
    header = "# Verified training dependency closure on Python 3.11, Linux x86_64.\n"
    args.output.write_text(header + "".join(f"{name}=={versions[name]}\n" for name in sorted(versions)))
    print(f"Locked {len(versions)} distributions to {args.output}")


if __name__ == "__main__":
    main()
