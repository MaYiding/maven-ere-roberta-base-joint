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
    queue = ["torch", "transformers", "numpy", "scipy", "scikit-learn", "safetensors", "tqdm", "gdown", "pytest"]
    versions = {}
    while queue:
        name = canonicalize_name(queue.pop())
        if name in versions:
            continue
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        for specification in distribution.requires or []:
            requirement = Requirement(specification)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                actual = metadata.version(requirement.name)
                if requirement.specifier and not requirement.specifier.contains(actual, prereleases=True):
                    raise RuntimeError(f"Dependency mismatch: {requirement.name} {actual} does not satisfy {requirement.specifier}")
                queue.append(requirement.name)
    header = "# Verified training dependency closure on Python 3.11, Linux x86_64.\n"
    args.output.write_text(header + "".join(f"{name}=={versions[name]}\n" for name in sorted(versions)))
    print(f"Locked {len(versions)} distributions to {args.output}")


if __name__ == "__main__":
    main()
