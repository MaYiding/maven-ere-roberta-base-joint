"""可训练、保存和独立加载的 MAVEN-ERE 四任务联合模型。"""

from transformers import AutoConfig, AutoModel

from .configuration_maven_ere import MavenEreConfig
from .modeling_maven_ere import MavenEreModel, MavenEreOutput

AutoConfig.register(MavenEreConfig.model_type, MavenEreConfig, exist_ok=True)
AutoModel.register(MavenEreConfig, MavenEreModel, exist_ok=True)

__all__ = ["MavenEreConfig", "MavenEreModel", "MavenEreOutput"]
