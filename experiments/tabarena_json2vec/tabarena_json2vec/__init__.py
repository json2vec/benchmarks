__all__ = ["JSON2VecTabArenaModel"]


def __getattr__(name: str):
    if name == "JSON2VecTabArenaModel":
        from tabarena_json2vec.ag_model import JSON2VecTabArenaModel

        return JSON2VecTabArenaModel
    raise AttributeError(name)
