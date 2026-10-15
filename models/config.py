from dataclasses import dataclass

@dataclass(frozen=True)
class Configuration:
    title: str = "Full CRAFT"
    height_supervision: bool = True

VARIANTS = {"G3": Configuration()}
